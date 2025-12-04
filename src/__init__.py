"""
ReACT-Drug: Reaction-Template Guided Reinforcement Learning for de novo Drug Design

A complete RL pipeline for molecular drug discovery using:
- ESM-2 for protein embeddings
- ChemBERTa for molecular encoding
- ChEMBL-derived reaction templates
- AutoDock Vina for binding affinity calculation
- PPO for multi-objective optimization
"""

__version__ = "1.0.0"
__author__ = "R Yadunandan, Nimisha Ghosh"

from .config import CONFIG, DEVICE
from .encoders import ESM2ProteinEncoder, ChemBERTaSmilesEncoder
from .chemistry import MolecularFragmenter, ReactionTemplateLibrary, MultiObjectiveRewardCalculator
from .docking import VinaDocker
from .environment import MolecularEnvironment
from .agent import PPOAgent, PPONetwork, PPOBuffer
from .utils import PDBbindDataLoader, set_seeds, check_dependencies