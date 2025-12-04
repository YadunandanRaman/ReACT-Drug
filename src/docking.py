"""
ReACT-Drug: AutoDock Vina Docking Integration
"""

import subprocess
import hashlib
from pathlib import Path

from openbabel import pybel

from .config import CONFIG


class VinaDocker:
    """AutoDock Vina docking interface"""
    
    def __init__(self, receptor_pdb_file, binding_site_center, binding_site_size=20.0):
        self.receptor_pdb_file = Path(receptor_pdb_file)
        self.binding_site_center = binding_site_center
        self.binding_site_size = binding_site_size
        self.temp_dir = Path(CONFIG["vina"]["temp_dir"])
        self.temp_dir.mkdir(exist_ok=True)
        
        self.receptor_pdbqt = None
        self._prepare_receptor()
        
        print(f"🔬 Vina Docker initialized")
        print(f"   Receptor: {self.receptor_pdbqt}")
        print(f"   Binding site: {self.binding_site_center}")
    
    def _prepare_receptor(self):
        """Prepare receptor PDBQT using Pybel"""
        self.receptor_pdbqt = self.temp_dir / "receptor.pdbqt"
        try:
            mol = next(pybel.readfile("pdb", str(self.receptor_pdb_file)))
            mol.addh()
            mol.calccharges(model='gasteiger')
            mol.write("pdbqt", str(self.receptor_pdbqt), overwrite=True, opt={"r": ""})
            
            if self.receptor_pdbqt.exists() and self.receptor_pdbqt.stat().st_size > 0:
                print("✅ Receptor prepared successfully")
            else:
                raise RuntimeError("Failed to create PDBQT file")
        except Exception as e:
            self.receptor_pdbqt = None
            print(f"❌ Receptor preparation failed: {e}")
            raise
    
    def check_availability(self):
        """Check if Vina executable is available"""
        try:
            result = subprocess.run(
                [CONFIG["vina"]["executable"], "--help"],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def dock_molecule(self, smiles):
        """Dock a molecule and return binding affinity"""
        if self.receptor_pdbqt is None:
            return None
        
        mol_id = hashlib.md5(smiles.encode()).hexdigest()[:8]
        ligand_pdbqt = self.temp_dir / f"ligand_{mol_id}.pdbqt"
        output_pdbqt = self.temp_dir / f"output_{mol_id}.pdbqt"
        
        # Prepare ligand
        if not self._prepare_ligand(smiles, ligand_pdbqt):
            return None
        
        # Run docking
        score = self._run_docking(ligand_pdbqt, output_pdbqt)
        
        # Cleanup
        self._cleanup([ligand_pdbqt, output_pdbqt])
        
        return score
    
    def _prepare_ligand(self, smiles, output_file):
        """Prepare ligand PDBQT from SMILES"""
        try:
            mol = pybel.readstring("smi", smiles)
            mol.addh()
            mol.make3D(forcefield='mmff94', steps=500)
            mol.calccharges(model='gasteiger')
            mol.write("pdbqt", str(output_file), overwrite=True)
            return output_file.exists() and output_file.stat().st_size > 0
        except Exception as e:
            print(f"❌ Ligand prep failed: {e}")
            return False
    
    def _run_docking(self, ligand_pdbqt, output_pdbqt):
        """Run Vina docking"""
        try:
            cmd = [
                CONFIG["vina"]["executable"],
                "--receptor", str(self.receptor_pdbqt),
                "--ligand", str(ligand_pdbqt),
                "--out", str(output_pdbqt),
                "--center_x", str(self.binding_site_center[0]),
                "--center_y", str(self.binding_site_center[1]),
                "--center_z", str(self.binding_site_center[2]),
                "--size_x", str(self.binding_site_size),
                "--size_y", str(self.binding_site_size),
                "--size_z", str(self.binding_site_size),
                "--exhaustiveness", str(CONFIG["vina"]["exhaustiveness"]),
                "--num_modes", str(CONFIG["vina"]["num_poses"]),
                "--seed", "42"
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, 
                timeout=CONFIG["vina"]["timeout"]
            )
            
            if result.returncode == 0:
                return self._parse_output(result.stdout)
            return None
        except Exception as e:
            print(f"❌ Docking error: {e}")
            return None
    
    def _parse_output(self, vina_output):
        """Parse Vina output for best score"""
        for line in vina_output.split('\n'):
            if line.strip() and line.strip().split()[0].isdigit():
                try:
                    return float(line.strip().split()[1])
                except Exception:
                    continue
        return None
    
    def _cleanup(self, files):
        """Clean up temporary files"""
        for f in files:
            if isinstance(f, Path) and f.exists():
                f.unlink(missing_ok=True)