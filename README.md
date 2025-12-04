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

![Pipeline](docs/pipeline.png)

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
git clone https://github.com/yourusername/ReACT-Drug.git
cd ReACT-Drug

# Create conda environment
conda create -n react-drug python=3.10
conda activate react-drug

# Install dependencies
pip install -r requirements.txt

# Install AutoDock Vina (optional, for docking)
conda install -c conda-forge autodock-vina

# Install Open Babel
conda install -c conda-forge openbabel
```

## Directory Structure
```
ReACT-Drug/
├── data/
│   ├── raw/              # PDBbind tar.gz and PDB files
│   ├── processed/        # Processed binding data
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

### 1. PDBbind Dataset

Download the PDBbind v2020 refined set:
```bash
# Download from http://www.pdbbind.org.cn/download/
# Place PDBbind_v2020_refined.tar.gz in data/raw/
```

### 2. Reaction Templates

Generate ChEMBL-derived templates:
```bash
# First, create the ChEMBL MMP database
# Download ChEMBL molecules and process with mmpdb

# Then run template extraction
python scripts/generate_templates.py
```

Or download pre-computed templates from [releases].

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

## Results

Example results on benchmark targets:

| Target | Mean Binding (kcal/mol) | Best Binding | Avg QED | High-Affinity Count |
|--------|------------------------|--------------|---------|---------------------|
| 5-HT1B | -9.3 | -11.2 | 0.32 | 45 |
| 5-HT2B | -9.2 | -10.8 | 0.35 | 38 |
| ACM2   | -10.3 | -11.5 | 0.28 | 62 |
| AKT1   | -9.1 | -10.4 | 0.41 | 35 |
| DRD2   | -9.9 | -11.1 | 0.33 | 52 |

## Citation

If you use ReACT-Drug in your research, please cite:
```bibtex
@article{yadunandan2025react,
  title={ReACT-Drug: Reaction-Template Guided Reinforcement Learning for de novo Drug Design},
  author={Yadunandan, R and Ghosh, Nimisha},
  year={2025}
}```
  
