# ReACT-Drug: Reaction-Template Guided Reinforcement Learning for *de novo* Drug Design

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete reinforcement learning pipeline for *de novo* molecular drug design using reaction-template guided molecular transformations.

## Overview

ReACT-Drug is an end-to-end RL framework that combines deep representation learning with multi-objective molecular optimization. Given a target protein, the system:

1. **Identifies similar proteins** using ESM-2 embeddings from the PDBbind database
2. **Fragments known ligands** using BRICS/RECAP to create starting molecules
3. **Grows molecules** using ChEMBL-derived reaction templates
4. **Optimizes** for binding affinity (AutoDock Vina), drug-likeness (QED), synthetic accessibility, and novelty


## Features

- 🧬 **ESM-2** protein embeddings for similarity search
- 💊 **ChemBERTa** molecular embeddings for state representation
- 🧪 **ChEMBL-derived** reaction templates for chemically valid transformations
- 🔬 **AutoDock Vina** integration for binding affinity scoring
- 🤖 **PPO** with dynamic action spaces for multi-objective optimization
- 🧩 **Fragment-based** starting state generation

## Installation
```bash
# Clone repository
git clone https://github.com/YadunandanRaman/ReACT-Drug.git
cd ReACT-Drug

# Create conda environment
conda create -n react-drug python=3.10
conda activate react-drug

# Install dependencies
pip install -r requirements.txt

# Install AutoDock Vina (for docking)
conda install -c conda-forge autodock-vina

# Install Open Babel
conda install -c conda-forge openbabel
```

## Directory Structure
```
ReACT-Drug/
├── data/                 # Create this directory (see Data Preparation)
│   ├── raw/              # PDBbind tar.gz and PDB files
│   ├── processed/        # Processed binding data (auto-generated)
│   └── templates/        # Place drug_templates.pkl here
├── src/
│   ├── __init__.py
│   ├── config.py         # Hyperparameters and paths
│   ├── encoders.py       # ESM-2 and ChemBERTa models
│   ├── chemistry.py      # Fragmentation, Templates, Rewards
│   ├── docking.py        # AutoDock Vina wrapper
│   ├── environment.py    # Gym-like RL Environment
│   ├── agent.py          # PPO Actor-Critic Agent
│   └── utils.py          # Data loading helpers
├── scripts/
│   └── generate_templates.py  # ChEMBL MMP extraction
├── models/               # Saved model checkpoints
├── results/              # Discovery results
├── main.py               # Entry point
├── requirements.txt
└── README.md
```

## Data Preparation

> **Note:** The `data/` directory is not included in this repository due to file size and licensing restrictions. You must create it and download the required files manually.

### Step 1: Create Data Directories
```bash
mkdir -p data/raw data/processed data/templates
```

### Step 2: Download PDBbind Dataset

The PDBbind dataset is required for training. Due to licensing requirements, please download directly from the official source.

**For Academic Use:**
1. Visit: https://www.pdbbind-plus.org.cn/download
2. Register for a free academic account
3. Download: `PDBbind_v2020_refined.tar.gz`
4. Place in `data/raw/`

**For Commercial Use:**
1. Visit: https://www.pdbbind-plus.org.cn/download
2. Contact PDBbind for commercial licensing information

### Step 3: Download Reaction Templates

Download the pre-computed ChEMBL-derived reaction templates:

📥 **[Download drug_templates.pkl](https://drive.google.com/file/d/1Q1LCFhAhu873oj5ubJ_wcC63VcxuMPSy/view?usp=share_link)**

Place the downloaded file in `data/templates/`

**Alternative:** Generate templates yourself using ChEMBL data:
```bash
python scripts/generate_templates.py
```

### Verify Setup

After completing the above steps, your data directory should look like:
```
data/
├── raw/
│   └── PDBbind_v2020_refined.tar.gz
├── processed/
│   └── (empty - will be auto-generated on first run)
└── templates/
    └── drug_templates.pkl
```

## Usage

### Basic Usage
```bash
python main.py --pdb_id 4nc3 --episodes 200
```

### Advanced Options
```bash
python main.py \
    --pdb_id 6lu7 \
    --binding_site "10.5,20.3,30.1" \
    --episodes 500 \
    --max_structures 10000 \
    --seed 42
```

### Using Local PDB File
```bash
python main.py \
    --pdb_file path/to/protein.pdb \
    --binding_site "x,y,z" \
    --episodes 300
```

## Configuration

All hyperparameters can be modified in `src/config.py`:
```python
CONFIG = {
    "discovery": {
        "episodes": 500,
        "max_steps_per_episode": 15,
        "success_criteria": {
            "binding_affinity": -8.5,  # kcal/mol
            "qed_score": 0.5,
            "sa_score": 4.0,
        },
    },
    "training": {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "ppo_epochs": 10,
        "clip_ratio": 0.2,
    },
    # ... more options
}
```

