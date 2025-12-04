"""
ReACT-Drug: Chemistry Module
Molecular Fragmentation, Reaction Templates, and Reward Calculation
"""

import pickle
import numpy as np
from collections import deque
from pathlib import Path
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors, AllChem
from rdkit.Chem import BRICS, Recap
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdChemReactions

# SA Score
from rdkit.Chem import RDConfig
import sys
import os
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

try:
    from rdchiral.main import rdchiralRunText
    RDCHIRAL_AVAILABLE = True
except ImportError:
    RDCHIRAL_AVAILABLE = False
    print("⚠️ rdchiral not available, using RDKit fallback")

from .config import CONFIG


class MolecularFragmenter:
    """Molecular fragmentation using BRICS and RECAP methods"""
    
    def __init__(self):
        self.min_size = CONFIG["fragmentation"]["min_fragment_size"]
        self.max_size = CONFIG["fragmentation"]["max_fragment_size"]
        self.methods = CONFIG["fragmentation"]["fragmentation_methods"]
        self.max_per_mol = CONFIG["fragmentation"]["max_fragments_per_molecule"]
        print("🧩 Molecular Fragmenter initialized")
    
    def fragment_molecule(self, smiles, parent_id=None):
        """Fragment a single molecule"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        all_fragments = []
        for method in self.methods:
            try:
                if method == "brics":
                    fragments = self._fragment_brics(mol, smiles, parent_id)
                elif method == "recap":
                    fragments = self._fragment_recap(mol, smiles, parent_id)
                else:
                    continue
                all_fragments.extend(fragments)
            except Exception:
                continue
        
        # Filter and deduplicate
        valid = self._filter_fragments(all_fragments)
        if CONFIG["fragmentation"]["remove_duplicates"]:
            valid = self._deduplicate(valid)
        
        # Limit count
        if len(valid) > self.max_per_mol:
            valid = sorted(valid, key=lambda x: abs(x['num_heavy_atoms'] - 
                          (self.min_size + self.max_size) / 2))
            valid = valid[:self.max_per_mol]
        
        return valid
    
    def _fragment_brics(self, mol, parent_smiles, parent_id):
        """Fragment using BRICS"""
        fragments = []
        try:
            brics_frags = BRICS.BRICSDecompose(mol)
            for i, frag_smi in enumerate(brics_frags):
                clean_smi = self._clean_fragment(frag_smi)
                if clean_smi:
                    frag_mol = Chem.MolFromSmiles(clean_smi)
                    if frag_mol:
                        fragments.append({
                            'smiles': Chem.MolToSmiles(frag_mol, canonical=True),
                            'parent_smiles': parent_smiles,
                            'parent_id': parent_id,
                            'method': 'brics',
                            'fragment_id': f"{parent_id}_brics_{i}" if parent_id else f"brics_{i}",
                            'num_heavy_atoms': frag_mol.GetNumHeavyAtoms(),
                        })
        except Exception:
            pass
        return fragments
    
    def _fragment_recap(self, mol, parent_smiles, parent_id):
        """Fragment using RECAP"""
        fragments = []
        try:
            recap_tree = Recap.RecapDecompose(mol)
            recap_frags = recap_tree.GetLeaves()
            for i, (frag_smi, _) in enumerate(recap_frags.items()):
                clean_smi = self._clean_fragment(frag_smi)
                if clean_smi:
                    frag_mol = Chem.MolFromSmiles(clean_smi)
                    if frag_mol:
                        fragments.append({
                            'smiles': Chem.MolToSmiles(frag_mol, canonical=True),
                            'parent_smiles': parent_smiles,
                            'parent_id': parent_id,
                            'method': 'recap',
                            'fragment_id': f"{parent_id}_recap_{i}" if parent_id else f"recap_{i}",
                            'num_heavy_atoms': frag_mol.GetNumHeavyAtoms(),
                        })
        except Exception:
            pass
        return fragments
    
    def _clean_fragment(self, smiles):
        """Remove dummy atoms from fragment"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            edit_mol = Chem.RWMol(mol)
            atoms_to_remove = [atom.GetIdx() for atom in edit_mol.GetAtoms() 
                              if atom.GetAtomicNum() == 0]
            
            for idx in sorted(atoms_to_remove, reverse=True):
                edit_mol.RemoveAtom(idx)
            
            clean_mol = edit_mol.GetMol()
            if clean_mol and clean_mol.GetNumAtoms() > 0:
                Chem.SanitizeMol(clean_mol)
                clean_mol = Chem.RemoveHs(clean_mol)
                return Chem.MolToSmiles(clean_mol, canonical=True)
        except Exception:
            pass
        return None
    
    def _filter_fragments(self, fragments):
        """Filter fragments by size and validity"""
        valid = []
        for frag in fragments:
            try:
                num_heavy = frag['num_heavy_atoms']
                if num_heavy < self.min_size or num_heavy > self.max_size:
                    continue
                
                mol = Chem.MolFromSmiles(frag['smiles'])
                if mol is None:
                    continue
                
                Chem.SanitizeMol(mol)
                mw = Descriptors.MolWt(mol)
                if mw < 50 or mw > 400:
                    continue
                
                if len(Chem.GetMolFrags(mol)) > 1:
                    continue
                
                valid.append(frag)
            except Exception:
                continue
        return valid
    
    def _deduplicate(self, fragments):
        """Remove duplicate fragments"""
        seen = set()
        unique = []
        for frag in fragments:
            if frag['smiles'] not in seen:
                seen.add(frag['smiles'])
                unique.append(frag)
        return unique
    
    def fragment_collection(self, molecules, show_progress=True):
        """Fragment a collection of molecules"""
        all_fragments = []
        iterator = tqdm(molecules, desc="Fragmenting") if show_progress else molecules
        
        for mol_data in iterator:
            smiles = mol_data['smiles']
            parent_id = mol_data.get('source', mol_data.get('protein_id', 'unknown'))
            fragments = self.fragment_molecule(smiles, parent_id)
            all_fragments.extend(fragments)
        
        return all_fragments


class ReactionTemplateLibrary:
    """ChEMBL-derived reaction template library"""
    
    def __init__(self, templates_file=None):
        self.templates_file = templates_file or CONFIG["data"]["templates_file"]
        self.templates = []
        
        print("🧪 Initializing Reaction Template Library...")
        self._load_templates()
    
    def _load_templates(self):
        """Load templates from file"""
        template_path = Path(self.templates_file)
        
        if not template_path.exists():
            print(f"❌ Template file not found: {template_path}")
            print("   Please run 'python scripts/generate_templates.py' first")
            return
        
        print(f"📂 Loading templates from {template_path}")
        with open(template_path, 'rb') as f:
            loaded = pickle.load(f)
        
        # Process templates
        print("🔧 Processing templates...")
        for template in tqdm(loaded, desc="Initializing"):
            try:
                rxn = rdChemReactions.ReactionFromSmarts(template['smarts'])
                if rxn and rxn.GetNumReactantTemplates() > 0:
                    template['reaction'] = rxn
                    template['num_reactants'] = rxn.GetNumReactantTemplates()
                    self.templates.append(template)
            except Exception:
                continue
        
        print(f"✅ Loaded {len(self.templates)} reaction templates")
    
    def get_applicable_templates(self, smiles, max_templates=None):
        """Get templates applicable to a given molecule"""
        if max_templates is None:
            max_templates = CONFIG["templates"]["max_templates_per_step"]
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            Chem.SanitizeMol(mol)
            
            applicable = []
            for template in self.templates:
                rxn = template['reaction']
                reactant_template = rxn.GetReactantTemplate(0)
                if mol.HasSubstructMatch(reactant_template):
                    applicable.append(template)
            
            # Sort by frequency
            applicable.sort(key=lambda x: x.get('frequency', 0), reverse=True)
            return applicable[:max_templates]
        except Exception:
            return []
    
    def apply_template(self, smiles, template):
        """Apply a reaction template to a molecule"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or template['num_reactants'] != 1:
                return None
            
            # Try rdchiral first for stereo-awareness
            if RDCHIRAL_AVAILABLE:
                products = rdchiralRunText(template['smarts'], smiles, combine_enantiomers=True)
                for prod_smi in products:
                    prod_mol = Chem.MolFromSmiles(prod_smi)
                    if prod_mol:
                        Chem.SanitizeMol(prod_mol)
                        canonical = Chem.MolToSmiles(prod_mol, canonical=True)
                        if Chem.MolFromSmiles(canonical):
                            return canonical
            
            # Fallback to RDKit
            rxn = template['reaction']
            products = rxn.RunReactants((mol,))
            if products and products[0]:
                prod_mol = products[0][0]
                Chem.SanitizeMol(prod_mol)
                return Chem.MolToSmiles(prod_mol, canonical=True)
            
        except Exception:
            pass
        return None


class MultiObjectiveRewardCalculator:
    """Multi-objective reward calculation"""
    
    def __init__(self, vina_docker=None):
        self.objectives = CONFIG["reward"]["objectives"]
        self.weights = CONFIG["reward"]["weights"]
        self.vina_docker = vina_docker
        
        self.calculators = {
            "binding_affinity": self._calc_binding_affinity,
            "drug_likeness": self._calc_drug_likeness,
            "synthetic_accessibility": self._calc_sa,
            "novelty": self._calc_novelty,
            "selectivity": self._calc_selectivity,
            "synthesizability": self._calc_synthesizability,
        }
        
        self.history = {obj: deque(maxlen=1000) for obj in self.objectives}
        self.stats = {obj: {"mean": 0.0, "std": 1.0} for obj in self.objectives}
        
        print("🎯 Multi-Objective Reward Calculator initialized")
    
    def calculate_reward(self, smiles, additional_info=None):
        """Calculate multi-objective reward"""
        if additional_info is None:
            additional_info = {}
        
        obj_values = {}
        for obj in self.objectives:
            if obj in self.calculators:
                try:
                    value = self.calculators[obj](smiles, additional_info)
                    obj_values[obj] = value
                    self.history[obj].append(value)
                except Exception:
                    obj_values[obj] = 0.0
            else:
                obj_values[obj] = 0.0
        
        # Normalize and scalarize
        normalized = self._normalize(obj_values)
        reward = sum(self.weights[i] * normalized.get(obj, 0) 
                    for i, obj in enumerate(self.objectives))
        reward /= sum(self.weights) if sum(self.weights) > 0 else 1
        
        return reward, obj_values
    
    def _calc_binding_affinity(self, smiles, info):
        """Calculate binding affinity using Vina"""
        if self.vina_docker:
            score = self.vina_docker.dock_molecule(smiles)
            if score is not None:
                return max(0.0, min(1.0, (-score) / 15.0))
        return 0.0
    
    def _calc_drug_likeness(self, smiles, info):
        """Calculate QED score"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return QED.qed(mol) if mol else 0.0
        except Exception:
            return 0.0
    
    def _calc_sa(self, smiles, info):
        """Calculate synthetic accessibility"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            sa_score = sascorer.calculateScore(mol)
            return max(0.0, 1.0 - sa_score / 10.0)
        except Exception:
            return 0.0
    
    def _calc_novelty(self, smiles, info):
        """Calculate novelty score"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            num_atoms = mol.GetNumAtoms()
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            num_hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [1, 6])
            return min(1.0, (num_atoms + num_rings * 2 + num_hetero) / 50.0)
        except Exception:
            return 0.0
    
    def _calc_selectivity(self, smiles, info):
        """Calculate selectivity score based on drug-likeness criteria"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            scores = [
                1.0 if mw <= 500 else max(0.0, 1.0 - (mw - 500) / 300),
                1.0 if logp <= 5 else max(0.0, 1.0 - (logp - 5) / 3),
                1.0 if hbd <= 5 else max(0.0, 1.0 - (hbd - 5) / 5),
                1.0 if hba <= 10 else max(0.0, 1.0 - (hba - 10) / 10),
            ]
            return np.mean(scores)
        except Exception:
            return 0.5
    
    def _calc_synthesizability(self, smiles, info):
        """Calculate synthesizability score"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            num_atoms = mol.GetNumAtoms()
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            num_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            
            penalty = (num_atoms - 10) * 0.02 + num_rings * 0.1 + num_chiral * 0.15
            return max(0.0, 1.0 - penalty)
        except Exception:
            return 0.5
    
    def _normalize(self, obj_values):
        """Normalize objective values"""
        import torch
        normalized = {}
        for obj, value in obj_values.items():
            stats = self.stats[obj]
            norm_val = (value - stats["mean"]) / max(stats["std"], 1e-6)
            normalized[obj] = torch.sigmoid(torch.tensor(norm_val)).item()
        return normalized
    
    def update_stats(self):
        """Update running statistics"""
        for obj in self.objectives:
            if len(self.history[obj]) > 10:
                values = list(self.history[obj])
                self.stats[obj]["mean"] = np.mean(values)
                self.stats[obj]["std"] = max(np.std(values), 1e-6)