"""
ReACT-Drug: Configuration and Hyperparameters
"""

import torch
from pathlib import Path

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory structure
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TEMPLATES_DIR = DATA_DIR / "templates"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
TEMP_DIR = BASE_DIR / "temp_docking"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DIR, TEMPLATES_DIR, MODELS_DIR, RESULTS_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

CONFIG = {
    # Data configuration
    "data": {
        "pdbbind_url": "http://www.pdbbind.org.cn/download/",
        "pdbbind_file": "PDBbind_v2020_refined.tar.gz",
        "raw_dir": str(RAW_DATA_DIR),
        "processed_dir": str(PROCESSED_DIR),
        "templates_dir": str(TEMPLATES_DIR),
        "templates_file": str(TEMPLATES_DIR / "drug_templates.pkl"),
        "min_binding_affinity": -12.0,
        "max_binding_affinity": 0.0,
        "max_protein_size": 1000,
        "max_ligand_size": 50,
    },
    
    # ESM-2 protein encoder
    "esm2": {
        "model_name": "facebook/esm2_t33_650M_UR50D",
        "max_length": 1024,
        "embedding_dim": 1280,
        "use_cached_embeddings": True,
        "batch_size": 8,
        "device": DEVICE,
        "similarity_top_k": 20,
        "min_similarity_threshold": 0.6,
    },
    
    # ChemBERTa molecular encoder
    "chemberta": {
        "model_name": "seyonec/ChemBERTa-zinc-base-v1",
        "max_length": 512,
        "embedding_dim": 768,
        "use_cached_embeddings": True,
        "batch_size": 32,
        "device": DEVICE,
    },
    
    # Reaction templates (ChEMBL-derived)
    "templates": {
        "max_templates_per_step": 1000,
        "min_template_frequency": 5,
    },
    
    # Molecular fragmentation
    "fragmentation": {
        "use_fragmentation": True,
        "min_fragment_size": 5,
        "max_fragment_size": 25,
        "fragmentation_methods": ["brics", "recap"],
        "remove_duplicates": True,
        "keep_parent_molecules": False,
        "max_fragments_per_molecule": 20,
    },
    
    # SMILES-based RL
    "smiles_rl": {
        "max_smiles_length": 100,
        "action_space_size": 1000,
        "starting_molecules_per_target": 100,
        "stereo_aware": True,
    },
    
    # AutoDock Vina
    "vina": {
        "executable": "vina",
        "obabel_executable": "/usr/bin/obabel",
        "search_space_size": 20,
        "exhaustiveness": 16,
        "num_poses": 1,
        "timeout": 300,
        "temp_dir": str(TEMP_DIR),
    },
    
    # Multi-objective reward
    "reward": {
        "objectives": [
            "binding_affinity",
            "drug_likeness", 
            "synthetic_accessibility",
            "novelty",
            "selectivity",
            "synthesizability"
        ],
        "weights": [1.0, 0.1, 0.1, 0.35, 0.05, 0.05],
        "curriculum_learning": False,
    },
    
    # PPO training
    "training": {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "ppo_epochs": 10,
        "clip_ratio": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
    },
    
    # Discovery
    "discovery": {
        "episodes": 500,
        "max_steps_per_episode": 15,
        "success_criteria": {
            "binding_affinity": -8.5,
            "qed_score": 0.5,
            "sa_score": 4.0,
        },
    },
    
    # General
    "general": {
        "device": DEVICE,
        "seed": 42,
        "checkpoint_interval": 50,
        "models_dir": str(MODELS_DIR),
        "results_dir": str(RESULTS_DIR),
    }
}