"""
ReACT-Drug: Molecular RL Environment
Gym-like environment for molecular generation
"""

import random
import hashlib
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.MolStandardize import rdMolStandardize

from .config import CONFIG
from .chemistry import MolecularFragmenter, ReactionTemplateLibrary, MultiObjectiveRewardCalculator
from .docking import VinaDocker


class MolecularEnvironment:
    """SMILES-based molecular RL environment"""
    
    def __init__(self, target_protein_sequence, protein_database, 
                 esm2_encoder, chemberta_encoder,
                 target_protein_pdb_file=None, binding_site_center=None):
        
        self.target_sequence = target_protein_sequence
        self.protein_database = protein_database
        self.esm2_encoder = esm2_encoder
        self.chemberta_encoder = chemberta_encoder
        self.target_id = hashlib.md5(target_protein_sequence.encode()).hexdigest()[:8]
        
        # Initialize docking
        self.vina_docker = None
        if target_protein_pdb_file and binding_site_center:
            try:
                self.vina_docker = VinaDocker(target_protein_pdb_file, binding_site_center)
                if not self.vina_docker.check_availability():
                    print("⚠️ Vina not available, docking disabled")
                    self.vina_docker = None
            except Exception as e:
                print(f"⚠️ Failed to initialize Vina: {e}")
        
        # Initialize components
        self.fragmenter = MolecularFragmenter()
        self.template_library = ReactionTemplateLibrary()
        
        # Get starting molecules
        self.starting_molecules = self._get_starting_molecules()
        
        # Initialize reward calculator
        self.reward_calculator = MultiObjectiveRewardCalculator(vina_docker=self.vina_docker)
        
        # State tracking
        self.current_smiles = None
        self.current_embedding = None
        self.possible_actions = []
        self.step_count = 0
        self.max_steps = CONFIG["discovery"]["max_steps_per_episode"]
        self.episode_molecules = set()
        self.episode_discoveries = []
        
        print(f"🧪 Molecular Environment initialized")
        print(f"   Target ID: {self.target_id}")
        print(f"   Vina docking: {'✅' if self.vina_docker else '❌'}")
        print(f"   Templates: {len(self.template_library.templates)}")
        print(f"   Starting molecules: {len(self.starting_molecules)}")
    
    def _neutralize_smiles(self, smiles):
        """Neutralize a SMILES string"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            un = rdMolStandardize.Uncharger()
            unmol = un.uncharge(mol)
            return Chem.MolToSmiles(unmol, canonical=True)
        except Exception:
            return smiles
    
    def _get_starting_molecules(self):
        """Get starting molecules from similar proteins"""
        print("🔍 Finding similar proteins using ESM-2...")
        
        similar_proteins = self.esm2_encoder.find_similar_proteins(
            self.target_sequence,
            self.protein_database,
            top_k=CONFIG["esm2"]["similarity_top_k"]
        )
        
        if not similar_proteins:
            print("❌ No similar proteins found, using defaults")
            return [
                {'smiles': 'c1ccccc1', 'source': 'default', 'similarity': 0.0},
                {'smiles': 'c1ccncc1', 'source': 'default', 'similarity': 0.0},
            ]
        
        print(f"✅ Found {len(similar_proteins)} similar proteins")
        
        # Collect ligands
        parent_molecules = []
        for prot in similar_proteins:
            pid = prot['protein_id']
            if pid in self.protein_database and 'ligand_smiles' in self.protein_database[pid]:
                parent_molecules.append({
                    'smiles': self.protein_database[pid]['ligand_smiles'],
                    'source': pid,
                    'similarity': prot['similarity']
                })
        
        print(f"📚 Collected {len(parent_molecules)} parent ligands")
        
        # Fragment if enabled
        if CONFIG["fragmentation"]["use_fragmentation"] and parent_molecules:
            print("🧩 Fragmenting parent molecules...")
            fragments = self.fragmenter.fragment_collection(parent_molecules)
            print(f"✅ Generated {len(fragments)} fragments")
            
            fragment_mols = []
            for frag in fragments:
                parent_sim = next((m['similarity'] for m in parent_molecules 
                                  if m['source'] == frag['parent_id']), 0.0)
                fragment_mols.append({
                    'smiles': frag['smiles'],
                    'source': frag['parent_id'],
                    'similarity': parent_sim,
                    'fragment_info': {
                        'method': frag['method'],
                        'num_heavy_atoms': frag['num_heavy_atoms'],
                    }
                })
            
            if CONFIG["fragmentation"]["keep_parent_molecules"]:
                starting = parent_molecules + fragment_mols
            else:
                starting = fragment_mols
        else:
            starting = parent_molecules
        
        # Deduplicate and limit
        seen = set()
        unique = []
        for mol in starting:
            if mol['smiles'] not in seen:
                seen.add(mol['smiles'])
                unique.append(mol)
        
        unique.sort(key=lambda x: -x['similarity'])
        max_starting = CONFIG["smiles_rl"]["starting_molecules_per_target"]
        
        return unique[:max_starting]
    
    def reset(self, starting_idx=None, episode=0):
        """Reset environment for new episode"""
        self.step_count = 0
        self.episode_molecules = set()
        self.episode_discoveries = []
        
        if not self.starting_molecules:
            raise ValueError("No starting molecules available")
        
        if starting_idx is None:
            starting_idx = random.randint(0, len(self.starting_molecules) - 1)
        
        starting_mol = self.starting_molecules[starting_idx % len(self.starting_molecules)]
        self.current_smiles = self._neutralize_smiles(starting_mol['smiles'])
        self.episode_molecules.add(self.current_smiles)
        
        # Get initial state and actions
        self.possible_actions = self._get_possible_actions(self.current_smiles)
        self.current_embedding = self.chemberta_encoder.encode_smiles([self.current_smiles])[0]
        
        print(f"🎯 Episode {episode}: Starting with {self.current_smiles}")
        print(f"   Actions available: {len(self.possible_actions)}")
        
        return self.current_embedding, self.possible_actions
    
    def step(self, action_idx):
        """Take a step in the environment"""
        self.step_count += 1
        
        if not self.possible_actions or action_idx >= len(self.possible_actions):
            reward = -0.5
            done = self.step_count >= self.max_steps
            info = {'invalid_action': True, 'smiles': self.current_smiles}
            self.possible_actions = self._get_possible_actions(self.current_smiles)
            return self.current_embedding, reward, done, info, self.possible_actions
        
        action = self.possible_actions[action_idx]
        new_smiles = action['smiles']
        
        # Check revisit penalty
        revisit_penalty = -0.5 if new_smiles in self.episode_molecules else 0.0
        self.episode_molecules.add(new_smiles)
        
        self.current_smiles = new_smiles
        self.current_embedding = self.chemberta_encoder.encode_smiles([new_smiles])[0]
        
        # Calculate reward
        reward, obj_values = self.reward_calculator.calculate_reward(new_smiles)
        reward += revisit_penalty
        
        # Check discovery
        is_discovery = self._is_discovery(obj_values)
        if is_discovery:
            self.episode_discoveries.append({
                'smiles': new_smiles,
                'objectives': obj_values,
                'step': self.step_count
            })
        
        done = self.step_count >= self.max_steps
        self.possible_actions = self._get_possible_actions(self.current_smiles)
        
        info = {
            'smiles': self.current_smiles,
            'is_discovery': is_discovery,
            'objective_values': obj_values,
            'template': action.get('template', 'N/A'),
        }
        
        return self.current_embedding, reward, done, info, self.possible_actions
    
    def _is_valid_molecule(self, smiles):
        """Check molecule validity"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            result = Chem.SanitizeMol(mol, catchErrors=True)
            if result != Chem.SanitizeFlags.SANITIZE_NONE:
                return False
            
            mw = Descriptors.MolWt(mol)
            if not (150 < mw < 650):
                return False
            
            if QED.qed(mol) < 0.2:
                return False
            
            return True
        except Exception:
            return False
    
    def _get_possible_actions(self, smiles):
        """Get all valid molecular transformations"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []
        
        reactant_heavy = mol.GetNumHeavyAtoms()
        products = {}
        
        templates = self.template_library.get_applicable_templates(
            smiles, max_templates=CONFIG["smiles_rl"]["action_space_size"]
        )
        
        for template in templates:
            try:
                prod_smi = self.template_library.apply_template(smiles, template)
                if prod_smi and prod_smi != smiles:
                    prod_smi = self._neutralize_smiles(prod_smi)
                    prod_mol = Chem.MolFromSmiles(prod_smi)
                    
                    if prod_mol:
                        result = Chem.SanitizeMol(prod_mol, catchErrors=True)
                        if result != Chem.SanitizeFlags.SANITIZE_NONE:
                            continue
                        
                        if len(Chem.GetMolFrags(prod_mol)) > 1:
                            continue
                        
                        if prod_mol.GetNumHeavyAtoms() < reactant_heavy * 0.7:
                            continue
                        
                        canonical = Chem.MolToSmiles(prod_mol, canonical=True)
                        if self._is_valid_molecule(canonical) and canonical not in products:
                            products[canonical] = f"SMARTS:{template['smarts']}"
            except Exception:
                continue
        
        return [{'smiles': s, 'template': t} for s, t in products.items()]
    
    def _is_discovery(self, obj_values):
        """Check if molecule meets discovery criteria"""
        if not obj_values:
            return False
        
        criteria = CONFIG["discovery"]["success_criteria"]
        binding_ok = obj_values.get('binding_affinity', 0) >= 0.7
        drug_ok = obj_values.get('drug_likeness', 0) >= criteria.get('qed_score', 0.5)
        sa_ok = obj_values.get('synthetic_accessibility', 0) >= 0.5
        
        return binding_ok and drug_ok and sa_ok