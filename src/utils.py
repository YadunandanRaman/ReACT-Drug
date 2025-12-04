"""
ReACT-Drug: Utility Functions
Data loading and helper functions
"""

import pickle
import tarfile
import random
import requests
import numpy as np
import torch
from pathlib import Path
from io import StringIO
from tqdm import tqdm

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from rdkit import Chem

from .config import CONFIG, RAW_DATA_DIR, PROCESSED_DIR


def set_seeds(seed=None):
    """Set random seeds for reproducibility"""
    if seed is None:
        seed = CONFIG["general"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🎲 Random seeds set to {seed}")


def check_dependencies():
    """Check all required dependencies"""
    print("🔍 Checking dependencies...")
    
    deps = {
        'PyTorch': 'torch',
        'RDKit': 'rdkit',
        'BioPython': 'Bio',
        'Transformers': 'transformers',
        'Scikit-learn': 'sklearn',
        'OpenBabel': 'openbabel',
    }
    
    missing = []
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"   {name}: ✅")
        except ImportError:
            print(f"   {name}: ❌")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️ Missing: {', '.join(missing)}")
        return False
    return True


def fetch_pdb_structure(pdb_id):
    """Fetch PDB structure from RCSB"""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        print(f"✅ Fetched PDB {pdb_id}")
        return response.text
    except Exception as e:
        print(f"❌ Failed to fetch {pdb_id}: {e}")
        return None


def extract_sequence_from_pdb(pdb_content):
    """Extract protein sequence from PDB content"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', StringIO(pdb_content))
        ppb = PPBuilder()
        polypeptides = ppb.build_peptides(structure)
        
        if polypeptides:
            longest = max(polypeptides, key=len)
            return str(longest.get_sequence())
    except Exception as e:
        print(f"❌ Error extracting sequence: {e}")
    return None


def extract_binding_site(pdb_content, ligand_name="LIG"):
    """Extract binding site center from ligand coordinates"""
    try:
        lines = pdb_content.split('\n')
        coords = []
        
        for line in lines:
            if line.startswith('HETATM') and ligand_name in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
        
        if coords:
            center = np.mean(coords, axis=0)
            return center.tolist()
    except Exception as e:
        print(f"❌ Error extracting binding site: {e}")
    return None


class PDBbindDataLoader:
    """PDBbind dataset loader"""
    
    def __init__(self):
        self.raw_dir = Path(RAW_DATA_DIR)
        self.processed_dir = Path(PROCESSED_DIR)
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        print("📚 PDBbind Data Loader initialized")
    
    def load_binding_data(self):
        """Load binding affinity data from PDBbind index"""
        index_patterns = [
            "INDEX_refined_data.2020",
            "INDEX_general_PL_data.2020",
            "index/INDEX_refined.2020",
        ]
        
        index_file = None
        for pattern in index_patterns:
            potential = self.raw_dir / pattern
            if potential.exists():
                index_file = potential
                break
            for found in self.raw_dir.rglob(pattern):
                if found.is_file():
                    index_file = found
                    break
            if index_file:
                break
        
        if index_file is None:
            print("❌ PDBbind index file not found")
            return None
        
        print(f"📖 Loading from {index_file}")
        binding_data = {}
        
        with open(index_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 4:
                    pdb_id = parts[0].lower()
                    try:
                        pk_value = float(parts[3])
                        RT_ln10 = 1.366
                        affinity = -RT_ln10 * pk_value
                        
                        if CONFIG["data"]["min_binding_affinity"] <= affinity <= CONFIG["data"]["max_binding_affinity"]:
                            binding_data[pdb_id] = {
                                'affinity': affinity,
                                'raw_data': parts
                            }
                    except (ValueError, IndexError):
                        continue
        
        print(f"✅ Loaded {len(binding_data)} binding entries")
        return binding_data
    
    def load_structures(self, binding_data, max_structures=None):
        """Load protein-ligand structures"""
        if max_structures is None:
            max_structures = len(binding_data)
        
        print(f"🔬 Loading structures (max: {max_structures})...")
        complexes = []
        processed = 0
        
        for pdb_id, binding_info in tqdm(binding_data.items(), desc="Loading"):
            if processed >= max_structures:
                break
            
            try:
                # Find files
                pdb_file, sdf_file = None, None
                search_dirs = [
                    self.raw_dir / pdb_id,
                    self.raw_dir / "refined-set" / pdb_id,
                    self.raw_dir / "v2020" / pdb_id,
                ]
                
                for search_dir in search_dirs:
                    if not search_dir.exists():
                        continue
                    
                    for pdb_name in [f"{pdb_id}_protein.pdb", f"{pdb_id}.pdb", "protein.pdb"]:
                        potential = search_dir / pdb_name
                        if potential.exists():
                            pdb_file = potential
                            break
                    
                    for sdf_name in [f"{pdb_id}_ligand.sdf", f"{pdb_id}.sdf", "ligand.sdf"]:
                        potential = search_dir / sdf_name
                        if potential.exists():
                            sdf_file = potential
                            break
                    
                    if pdb_file and sdf_file:
                        break
                
                if not pdb_file or not sdf_file:
                    continue
                
                # Extract sequence
                sequence = self._extract_sequence(pdb_file)
                if not sequence:
                    continue
                
                # Extract ligand SMILES
                smiles = self._extract_smiles(sdf_file)
                if not smiles:
                    continue
                
                complexes.append({
                    'pdb_id': pdb_id,
                    'protein_sequence': sequence,
                    'ligand_smiles': smiles,
                    'binding_affinity': binding_info['affinity'],
                    'pdb_file': str(pdb_file),
                    'sdf_file': str(sdf_file)
                })
                processed += 1
                
            except Exception:
                continue
        
        print(f"✅ Loaded {len(complexes)} complexes")
        return complexes
    
    def _extract_sequence(self, pdb_file):
        """Extract protein sequence from PDB file"""
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', str(pdb_file))
            ppb = PPBuilder()
            polypeptides = ppb.build_peptides(structure)
            
            if polypeptides:
                longest = max(polypeptides, key=len)
                seq = str(longest.get_sequence())
                if len(seq) <= CONFIG["data"]["max_protein_size"]:
                    return seq
        except Exception:
            pass
        return None
    
    def _extract_smiles(self, sdf_file):
        """Extract ligand SMILES from SDF file"""
        try:
            supplier = Chem.SDMolSupplier(str(sdf_file))
            for mol in supplier:
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles and len(smiles) <= CONFIG["data"]["max_ligand_size"] * 2:
                        if Chem.MolFromSmiles(smiles):
                            return smiles
        except Exception:
            pass
        return None
    
    def extract_pdbbind(self):
        """Extract PDBbind archive"""
        archive = self.raw_dir / CONFIG["data"]["pdbbind_file"]
        if not archive.exists():
            print(f"❌ Archive not found: {archive}")
            return False
        
        print(f"📦 Extracting {archive}...")
        try:
            with tarfile.open(archive, 'r:gz') as tar:
                tar.extractall(path=self.raw_dir)
            print("✅ Extraction complete")
            return True
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def save_processed(self, data, filename):
        """Save processed data"""
        filepath = self.processed_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Saved to {filepath}")
    
    def load_processed(self, filename):
        """Load processed data"""
        filepath = self.processed_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"📂 Loaded from {filepath}")
            return data
        return None