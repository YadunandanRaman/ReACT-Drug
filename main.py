#!/usr/bin/env python
"""
ReACT-Drug: Reaction-Template Guided Reinforcement Learning for de novo Drug Design

Main entry point for training and molecular discovery.

Usage:
    python main.py --pdb_id 4nc3 --binding_site -16.210,-15.874,5.523 --episodes 200
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from src.config import CONFIG, DEVICE, RESULTS_DIR, MODELS_DIR
from src.encoders import ESM2ProteinEncoder, ChemBERTaSmilesEncoder
from src.environment import MolecularEnvironment
from src.agent import PPOAgent
from src.utils import (
    set_seeds, 
    check_dependencies, 
    PDBbindDataLoader,
    fetch_pdb_structure,
    extract_sequence_from_pdb
)


class ReACTDrugDiscovery:
    """Main discovery system"""
    
    def __init__(self, target_pdb_file, target_sequence, binding_site_center):
        self.target_pdb_file = target_pdb_file
        self.target_sequence = target_sequence
        self.binding_site_center = binding_site_center
        
        # Initialize components
        self.data_loader = PDBbindDataLoader()
        print("🧬 Initializing ESM-2 encoder...")
        self.esm2_encoder = ESM2ProteinEncoder()
        print("💊 Initializing ChemBERTa encoder...")
        self.chemberta_encoder = ChemBERTaSmilesEncoder()
        
        # Data
        self.protein_database = {}
        self.training_data = None
        
        # RL components
        self.env = None
        self.agent = None
        
        # Tracking
        self.discoveries = []
        self.performance_history = {
            'rewards': [],
            'episodes': [],
            'discoveries_per_episode': [],
            'template_usage': defaultdict(int),
        }
        
        print("🏭 ReACT-Drug Discovery System initialized")
    
    def load_training_data(self, max_structures=5000, force_reload=False):
        """Load PDBbind training data"""
        print("📚 Loading training data...")
        
        processed_file = "training_data.pkl"
        if not force_reload:
            existing = self.data_loader.load_processed(processed_file)
            if existing:
                self.training_data = existing
                self._build_protein_database()
                return True
        
        # Load binding data
        binding_data = self.data_loader.load_binding_data()
        if binding_data is None:
            print("⚠️ Attempting to extract PDBbind archive...")
            self.data_loader.extract_pdbbind()
            binding_data = self.data_loader.load_binding_data()
            if binding_data is None:
                return False
        
        # Load structures
        complexes = self.data_loader.load_structures(binding_data, max_structures)
        if not complexes:
            return False
        
        self.training_data = complexes
        self.data_loader.save_processed(complexes, processed_file)
        self._build_protein_database()
        
        return True
    
    def _build_protein_database(self):
        """Build protein database from training data"""
        print("🏗️ Building protein database...")
        for complex_data in self.training_data:
            pdb_id = complex_data['pdb_id']
            self.protein_database[pdb_id] = {
                'sequence': complex_data['protein_sequence'],
                'ligand_smiles': complex_data['ligand_smiles'],
                'binding_affinity': complex_data['binding_affinity'],
                'pdb_file': complex_data['pdb_file'],
            }
        print(f"✅ Database: {len(self.protein_database)} entries")
    
    def initialize_rl(self):
        """Initialize RL environment and agent"""
        print("🤖 Initializing RL components...")
        
        self.env = MolecularEnvironment(
            target_protein_sequence=self.target_sequence,
            protein_database=self.protein_database,
            esm2_encoder=self.esm2_encoder,
            chemberta_encoder=self.chemberta_encoder,
            target_protein_pdb_file=self.target_pdb_file,
            binding_site_center=self.binding_site_center
        )
        
        state_dim = CONFIG["chemberta"]["embedding_dim"]
        self.agent = PPOAgent(state_dim, self.chemberta_encoder)
        
        print("✅ RL components ready")
        return True
    
    def run_discovery(self, num_episodes=None):
        """Run molecular discovery"""
        if num_episodes is None:
            num_episodes = CONFIG["discovery"]["episodes"]
        
        print(f"\n🚀 Starting discovery for {num_episodes} episodes...")
        print(f"   Max steps per episode: {CONFIG['discovery']['max_steps_per_episode']}")
        print(f"   Templates available: {len(self.env.template_library.templates)}")
        
        for episode in range(num_episodes):
            self._run_episode(episode)
            
            if episode % 25 == 0 and episode > 0:
                self._report_progress(episode)
            
            if episode % CONFIG["general"]["checkpoint_interval"] == 0 and episode > 0:
                self._save_checkpoint(episode)
        
        self._final_analysis()
    
    def _run_episode(self, episode):
        """Run single episode"""
        state, actions = self.env.reset(episode=episode)
        episode_reward = 0
        episode_discoveries = []
        
        print(f"\n--- Episode {episode:03d} ---")
        print(f"Starting: {self.env.current_smiles}")
        
        for step in range(CONFIG["discovery"]["max_steps_per_episode"]):
            action_idx, log_prob, value, context = self.agent.select_action(state, actions)
            
            if action_idx is None:
                print("   No valid actions, terminating")
                break
            
            next_state, reward, done, info, next_actions = self.env.step(action_idx)
            
            # Log step
            template = info.get('template', 'N/A')[:50]
            obj_vals = info.get('objective_values', {})
            print(f"   Step {step+1:02d} | R: {reward:.3f} | "
                  f"Bind: {obj_vals.get('binding_affinity', 0):.3f} | "
                  f"QED: {obj_vals.get('drug_likeness', 0):.3f}")
            
            if template != 'N/A':
                self.performance_history['template_usage'][template] += 1
            
            self.agent.store_experience(state, action_idx, reward, value, log_prob, done, context)
            episode_reward += reward
            
            if info.get('is_discovery', False):
                discovery = {
                    'smiles': info['smiles'],
                    'objectives': obj_vals,
                    'episode': episode,
                    'step': step,
                    'binding_raw': -obj_vals.get('binding_affinity', 0) * 15.0
                }
                episode_discoveries.append(discovery)
                self.discoveries.append(discovery)
                print(f"   🔬 DISCOVERY! {info['smiles'][:50]}...")
            
            state = next_state
            actions = next_actions
            
            if done:
                break
        
        # Update agent
        if len(self.agent.buffer.states) >= CONFIG["training"]["batch_size"]:
            losses = self.agent.update(episode)
            if losses:
                print(f"   Update: Policy={losses.get('policy_loss', 0):.3f}")
        
        # Track
        self.performance_history['rewards'].append(episode_reward)
        self.performance_history['episodes'].append(episode)
        self.performance_history['discoveries_per_episode'].append(len(episode_discoveries))
    
    def _report_progress(self, episode):
        """Report progress"""
        print(f"\n📊 Progress Report - Episode {episode}")
        print("=" * 60)
        print(f"Total Discoveries: {len(self.discoveries)}")
        
        if self.discoveries:
            recent = sorted(self.discoveries[-10:], key=lambda x: x.get('binding_raw', 0))[:3]
            print("Recent High-Affinity:")
            for d in recent:
                print(f"   {d['smiles'][:40]}... ({d['binding_raw']:.2f} kcal/mol)")
        
        if self.performance_history['rewards']:
            avg = np.mean(self.performance_history['rewards'][-50:])
            print(f"Avg Reward (last 50): {avg:.3f}")

        print("=" * 60)
    
    def _save_checkpoint(self, episode):
        """Save checkpoint"""
        checkpoint = {
            'episode': episode,
            'discoveries': self.discoveries,
            'performance_history': dict(self.performance_history),
            'config': CONFIG,
        }
        
        checkpoint_path = RESULTS_DIR / f"checkpoint_ep_{episode}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Save agent
        agent_path = MODELS_DIR / f"agent_ep_{episode}.pt"
        self.agent.save(agent_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _final_analysis(self):
        """Final analysis and reporting"""
        print(f"\n{'='*80}")
        print("🎯 FINAL DISCOVERY ANALYSIS")
        print("=" * 80)
        
        print(f"\n📊 Summary:")
        print(f"   Total Discoveries: {len(self.discoveries)}")
        print(f"   Total Episodes: {len(self.performance_history['rewards'])}")
        
        if self.discoveries:
            binding_scores = [d.get('binding_raw', 0) for d in self.discoveries]
            qed_scores = [d.get('objectives', {}).get('drug_likeness', 0) for d in self.discoveries]
            
            print(f"\n📈 Binding Affinity Statistics:")
            print(f"   Average: {np.mean(binding_scores):.3f} kcal/mol")
            print(f"   Best: {min(binding_scores):.3f} kcal/mol")
            print(f"   Std Dev: {np.std(binding_scores):.3f}")
            
            print(f"\n💊 Drug-likeness (QED) Statistics:")
            print(f"   Average: {np.mean(qed_scores):.3f}")
            print(f"   Best: {max(qed_scores):.3f}")
            
            # Count high-affinity discoveries
            high_affinity = [d for d in self.discoveries if d.get('binding_raw', 0) <= -8.5]
            ultra_high = [d for d in self.discoveries if d.get('binding_raw', 0) <= -10.0]
            
            print(f"\n🏆 High-Affinity Discoveries:")
            print(f"   ≤ -8.5 kcal/mol: {len(high_affinity)}")
            print(f"   ≤ -10.0 kcal/mol: {len(ultra_high)}")
            
            # Top 10 discoveries
            print(f"\n🥇 Top 10 Discoveries by Binding Affinity:")
            sorted_discoveries = sorted(self.discoveries, key=lambda x: x.get('binding_raw', 999))
            
            for i, d in enumerate(sorted_discoveries[:10]):
                obj = d.get('objectives', {})
                print(f"   {i+1:2d}. {d['smiles']}")
                print(f"       Binding: {d['binding_raw']:.2f} kcal/mol | "
                      f"QED: {obj.get('drug_likeness', 0):.3f} | "
                      f"SA: {obj.get('synthetic_accessibility', 0):.3f}")
        
        # Template usage
        if self.performance_history['template_usage']:
            print(f"\n🧪 Top 10 Most Used Templates:")
            top_templates = sorted(
                self.performance_history['template_usage'].items(),
                key=lambda x: x[1], reverse=True
            )[:10]
            
            for i, (template, count) in enumerate(top_templates):
                print(f"   {i+1:2d}. {template[:60]}... ({count} uses)")
        
        # Performance
        if self.performance_history['rewards']:
            print(f"\n📉 Training Performance:")
            print(f"   Initial Avg Reward (first 50): {np.mean(self.performance_history['rewards'][:50]):.3f}")
            print(f"   Final Avg Reward (last 50): {np.mean(self.performance_history['rewards'][-50:]):.3f}")
            
            discovery_rate = sum(1 for x in self.performance_history['discoveries_per_episode'] if x > 0)
            print(f"   Episodes with Discoveries: {discovery_rate}/{len(self.performance_history['episodes'])}")
        
        # Save final results
        results = {
            'discoveries': self.discoveries,
            'performance_history': dict(self.performance_history),
            'config': CONFIG,
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'total_discoveries': len(self.discoveries),
                'high_affinity_count': len([d for d in self.discoveries if d.get('binding_raw', 0) <= -8.5]),
                'avg_binding': np.mean([d.get('binding_raw', 0) for d in self.discoveries]) if self.discoveries else 0,
                'avg_qed': np.mean([d.get('objectives', {}).get('drug_likeness', 0) for d in self.discoveries]) if self.discoveries else 0,
            }
        }
        
        results_path = RESULTS_DIR / f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to JSON-serializable
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, defaultdict):
                return dict(obj)
            else:
                return obj
        
        with open(results_path, 'w') as f:
            json.dump(make_serializable(results), f, indent=2)
        
        print(f"\n📄 Results saved: {results_path}")
        print("=" * 80)
        print("✅ ReACT-Drug Discovery Complete!")
        print("=" * 80)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ReACT-Drug: Reaction-Template Guided RL for Drug Design"
    )
    
    parser.add_argument(
        "--pdb_id", type=str, default="4nc3",
        help="PDB ID of target protein (default: 4nc3)"
    )
    
    parser.add_argument(
        "--pdb_file", type=str, default=None,
        help="Path to local PDB file (overrides --pdb_id)"
    )
    
    parser.add_argument(
        "--binding_site", type=str, default="-16.210,-15.874,5.523",
        help="Binding site center as x,y,z (default: -16.210,-15.874,5.523)"
    )
    
    parser.add_argument(
        "--episodes", type=int, default=200,
        help="Number of discovery episodes (default: 200)"
    )
    
    parser.add_argument(
        "--max_structures", type=int, default=5000,
        help="Max PDBbind structures to load (default: 5000)"
    )
    
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--force_reload", action="store_true",
        help="Force reload of training data"
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    print("=" * 80)
    print("🧬 ReACT-Drug: Reaction-Template Guided RL for de novo Drug Design")
    print("=" * 80)
    print("   🧬 ESM-2 for protein similarity search")
    print("   💊 ChemBERTa for molecular encoding")
    print("   🧪 ChEMBL-derived reaction templates")
    print("   🔬 AutoDock Vina for binding affinity")
    print("   🤖 PPO for multi-objective optimization")
    print("=" * 80)
    
    args = parse_args()
    
    # Set seeds
    set_seeds(args.seed)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Missing dependencies. Please install requirements.")
        return
    
    # Parse binding site
    binding_site = [float(x) for x in args.binding_site.split(",")]
    print(f"\n🎯 Target Configuration:")
    print(f"   PDB ID: {args.pdb_id}")
    print(f"   Binding Site: {binding_site}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Device: {DEVICE}")
    
    # Get target protein
    if args.pdb_file and Path(args.pdb_file).exists():
        target_pdb_file = args.pdb_file
        with open(target_pdb_file, 'r') as f:
            pdb_content = f.read()
    else:
        print(f"\n📥 Fetching PDB structure {args.pdb_id}...")
        pdb_content = fetch_pdb_structure(args.pdb_id)
        if pdb_content is None:
            print("❌ Failed to fetch PDB structure")
            return
        
        # Save PDB file
        from src.config import RAW_DATA_DIR
        target_pdb_file = RAW_DATA_DIR / f"{args.pdb_id}.pdb"
        with open(target_pdb_file, 'w') as f:
            f.write(pdb_content)
        print(f"✅ Saved to {target_pdb_file}")
    
    # Extract sequence
    print("🧬 Extracting protein sequence...")
    target_sequence = extract_sequence_from_pdb(pdb_content)
    if target_sequence is None:
        print("❌ Failed to extract protein sequence")
        return
    print(f"✅ Sequence length: {len(target_sequence)} residues")
    
    # Initialize discovery system
    print("\n🏭 Initializing discovery system...")
    discovery = ReACTDrugDiscovery(
        target_pdb_file=str(target_pdb_file),
        target_sequence=target_sequence,
        binding_site_center=binding_site
    )
    
    # Load training data
    print("\n📚 Loading training data...")
    if not discovery.load_training_data(
        max_structures=args.max_structures,
        force_reload=args.force_reload
    ):
        print("❌ Failed to load training data")
        print("\nPlease ensure PDBbind data is available:")
        print(f"  1. Download from: {CONFIG['data']['pdbbind_url']}")
        print(f"  2. Place archive in: {RAW_DATA_DIR}")
        return
    
    # Initialize RL components
    print("\n🤖 Initializing RL components...")
    if not discovery.initialize_rl():
        print("❌ Failed to initialize RL components")
        return
    
    # Check templates
    if len(discovery.env.template_library.templates) == 0:
        print("\n⚠️ No reaction templates loaded!")
        print("Please run: python scripts/generate_templates.py")
        print("Or place drug_templates.pkl in data/templates/")
        return
    
    # Run discovery
    print("\n🚀 Starting molecular discovery...")
    try:
        discovery.run_discovery(num_episodes=args.episodes)
    except KeyboardInterrupt:
        print("\n⚡ Discovery interrupted by user")
        discovery._final_analysis()
    
    print("\n✅ ReACT-Drug completed successfully!")


if __name__ == "__main__":
    main()