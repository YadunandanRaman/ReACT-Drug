#!/usr/bin/env python
"""
ReACT-Drug: ChEMBL MMP Template Extraction Script

This script extracts drug-relevant reaction templates from the ChEMBL database
using Matched Molecular Pair (MMP) analysis.

Prerequisites:
1. Download ChEMBL molecules and process with mmpdb:
   - Download ChEMBL SDF or use chembl_webresource_client
   - Run: mmpdb fragment chembl_molecules.smi -o chembl_fragmented.csv
   - Run: mmpdb index chembl_fragmented.csv -o chembl_mmpdb.db

2. Place chembl_mmpdb.db in the data/templates/ directory

Usage:
    python scripts/generate_templates.py
"""

import pickle
import sqlite3
from pathlib import Path
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TEMPLATES_DIR


class MMPTemplateExtractor:
    """Extract drug-relevant transformation templates from ChEMBL MMP database"""
    
    def __init__(self, mmpdb_path):
        self.conn = sqlite3.connect(mmpdb_path)
        self.transformations = {}
    
    def _parse_transformation(self, tx):
        """Parse and validate transformation SMILES"""
        try:
            mol1 = Chem.MolFromSmiles(tx['mol1_smiles'])
            core = Chem.MolFromSmiles(tx['core_smiles'])
            var1 = Chem.MolFromSmiles(tx['var_smiles1'])
            var2 = Chem.MolFromSmiles(tx['var_smiles2'])
            if all([mol1, core, var1, var2]):
                return mol1, core, var1, var2
        except Exception:
            pass
        return None
    
    def _is_drug_relevant(self, tx, max_heavy_change=10, max_ratio=0.5, min_core_atoms=6):
        """Filter for drug-relevant transformations"""
        parsed = self._parse_transformation(tx)
        if not parsed:
            return False
        
        mol1, core, var1, var2 = parsed
        
        var1_atoms = var1.GetNumHeavyAtoms()
        var2_atoms = var2.GetNumHeavyAtoms()
        
        if var1_atoms > max_heavy_change or var2_atoms > max_heavy_change:
            return False
        
        mol1_atoms = mol1.GetNumHeavyAtoms()
        ratio = max(var1_atoms, var2_atoms) / mol1_atoms if mol1_atoms > 0 else 0
        if ratio > max_ratio:
            return False
        
        if core.GetNumHeavyAtoms() < min_core_atoms:
            return False
        
        return True
    
    def extract_and_filter(self):
        """Extract all transformation rules with filtering"""
        print("📊 Extracting transformation rules from database...")
        
        rule_query = """
        SELECT
            from_rule.smiles AS from_smarts,
            to_rule.smiles AS to_smarts,
            COUNT(*) AS frequency
        FROM rule
        JOIN rule_smiles AS from_rule ON rule.from_smiles_id = from_rule.id
        JOIN rule_smiles AS to_rule ON rule.to_smiles_id = to_rule.id
        GROUP BY from_smarts, to_smarts
        HAVING frequency >= 1
        ORDER BY frequency DESC
        """
        
        cursor = self.conn.execute(rule_query)
        all_rules = cursor.fetchall()
        print(f"   Found {len(all_rules)} raw rules")
        
        drug_relevant_rules = {}
        
        for from_smarts, to_smarts, frequency in tqdm(all_rules, desc="Filtering"):
            # Get example for validation
            example_query = """
            SELECT
                c.clean_smiles,
                cs.smiles
            FROM pair AS p
            JOIN compound AS c ON p.compound1_id = c.id
            JOIN constant_smiles AS cs ON p.constant_id = cs.id
            JOIN rule_environment AS re ON p.rule_environment_id = re.id
            JOIN rule AS r ON re.rule_id = r.id
            JOIN rule_smiles AS from_s ON r.from_smiles_id = from_s.id
            JOIN rule_smiles AS to_s ON r.to_smiles_id = to_s.id
            WHERE from_s.smiles = ? AND to_s.smiles = ?
            LIMIT 1
            """
            
            example_cursor = self.conn.execute(example_query, (from_smarts, to_smarts))
            representative = example_cursor.fetchone()
            
            if representative:
                mol1_smiles, core_smiles = representative
                tx_data = {
                    'mol1_smiles': mol1_smiles,
                    'core_smiles': core_smiles,
                    'var_smiles1': from_smarts,
                    'var_smiles2': to_smarts,
                }
                
                if self._is_drug_relevant(tx_data):
                    rule_smiles = f"{from_smarts}>>{to_smarts}"
                    drug_relevant_rules[rule_smiles] = {
                        'from_smarts': from_smarts,
                        'to_smarts': to_smarts,
                        'frequency': frequency
                    }
        
        self.transformations = drug_relevant_rules
        print(f"✅ Filtered to {len(self.transformations)} drug-relevant transformations")
        return self.transformations
    
    def convert_to_templates(self):
        """Convert transformations to RDKit reaction templates"""
        print("🧪 Converting to reaction templates...")
        templates = []
        
        for rule_smiles, trans in tqdm(self.transformations.items(), desc="Creating"):
            try:
                reaction_smarts = f"{trans['from_smarts']}>>{trans['to_smarts']}"
                rxn = AllChem.ReactionFromSmarts(reaction_smarts)
                
                if rxn and rxn.GetNumReactantTemplates() > 0:
                    templates.append({
                        'smarts': reaction_smarts,
                        'from_fragment': trans['from_smarts'],
                        'to_fragment': trans['to_smarts'],
                        'frequency': trans['frequency'],
                    })
            except Exception:
                continue
        
        print(f"✅ Created {len(templates)} reaction templates")
        return templates


def main():
    print("=" * 60)
    print("ReACT-Drug: ChEMBL MMP Template Extraction")
    print("=" * 60)
    
    # Check for database
    db_path = TEMPLATES_DIR / "chembl_mmpdb.db"
    
    if not db_path.exists():
        print(f"\n❌ Database not found at: {db_path}")
        print("\nTo create the database:")
        print("1. Download ChEMBL molecules")
        print("2. Run: mmpdb fragment chembl_molecules.smi -o fragmented.csv")
        print("3. Run: mmpdb index fragmented.csv -o chembl_mmpdb.db")
        print(f"4. Place chembl_mmpdb.db in {TEMPLATES_DIR}")
        return
    
    # Extract templates
    extractor = MMPTemplateExtractor(str(db_path))
    extractor.extract_and_filter()
    templates = extractor.convert_to_templates()
    
    # Save
    output_file = TEMPLATES_DIR / "drug_templates.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(templates, f)
    
    print(f"\n💾 Saved {len(templates)} templates to {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()