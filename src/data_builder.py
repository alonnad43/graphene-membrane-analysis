# data_builder.py

"""
Builds LAMMPS-compatible atomic structures for GO, rGO, and hybrid membranes.

Creates realistic molecular representations based on empirical data from Phase 1.
Focuses on physical structure generation without chemical reaction modeling.
"""

import numpy as np
import os
from properties import LAMMPS_PARAMETERS

class LAMMPSDataBuilder:
    """
    Builds atomic structures for LAMMPS simulations.
    """
    
    def __init__(self):
        self.params = LAMMPS_PARAMETERS
        self.atom_types = {
            'C': 1,    # Carbon in graphene
            'O': 2,    # Oxygen in GO
            'H': 3,    # Hydrogen in GO/water
            'OW': 4,   # Water oxygen
            'HW': 5    # Water hydrogen
        }
    
    def create_membrane_structure(self, membrane):
        """
        Create atomic structure for a membrane based on its properties.
        
        Args:
            membrane: Membrane object from Phase 1
            
        Returns:
            np.array: Atomic structure [atom_id, type, x, y, z]
        """
        # Small system for demonstration (5x5 unit cells)
        nx, ny = 5, 5
        layer_z = 0.0
        
        # Create base graphene structure
        atoms = self.create_graphene_sheet(nx, ny, layer_z)
        
        # Add functional groups for GO
        if "GO" in membrane.name:
            atoms = self.add_go_functional_groups(atoms, oxidation_ratio=0.2)
        
        # Add water layer above membrane
        atoms_with_water = self.add_water_layer(atoms, water_thickness=15.0)
        
        return atoms_with_water
    
    def create_graphene_sheet(self, nx=5, ny=5, layer_z=0.0):
        """
        Create a graphene sheet with hexagonal lattice.
        
        Args:
            nx, ny (int): Number of unit cells
            layer_z (float): Z-coordinate of the layer
        
        Returns:
            np.array: Carbon atom coordinates
        """
        # Graphene lattice parameters
        a = 2.46  # Lattice constant (Angstrom)
        atoms = []
        atom_id = 1
        
        for i in range(nx):
            for j in range(ny):
                # First carbon atom in unit cell
                x1 = i * a + (j % 2) * a/2
                y1 = j * a * np.sqrt(3)/2
                atoms.append([atom_id, self.atom_types['C'], x1, y1, layer_z])
                atom_id += 1
                
                # Second carbon atom in unit cell
                x2 = x1 + a/2
                y2 = y1
                atoms.append([atom_id, self.atom_types['C'], x2, y2, layer_z])
                atom_id += 1
        
        return np.array(atoms)
    
    def add_go_functional_groups(self, carbon_atoms, oxidation_ratio=0.2):
        """
        Add functional groups to convert graphene to GO.
        
        Args:
            carbon_atoms (np.array): Base carbon structure
            oxidation_ratio (float): Fraction of carbons with functional groups
        
        Returns:
            np.array: Extended structure with functional groups
        """
        atoms = carbon_atoms.copy().tolist()
        atom_id = int(carbon_atoms[:, 0].max()) + 1
        
        # Randomly select carbons for oxidation
        n_carbons = len(carbon_atoms)
        n_oxidized = int(n_carbons * oxidation_ratio)
        oxidized_indices = np.random.choice(n_carbons, n_oxidized, replace=False)
        
        for idx in oxidized_indices:
            carbon = carbon_atoms[idx]
            cx, cy, cz = carbon[2], carbon[3], carbon[4]
            
            # Add hydroxyl group (-OH) above carbon
            if np.random.random() < 0.7:  # 70% hydroxyl groups
                # Oxygen above carbon
                atoms.append([atom_id, self.atom_types['O'], cx, cy, cz + 1.4])
                atom_id += 1
                # Hydrogen above oxygen
                atoms.append([atom_id, self.atom_types['H'], cx, cy, cz + 2.0])
                atom_id += 1
            else:  # 30% epoxy groups (simplified as oxygen bridge)
                atoms.append([atom_id, self.atom_types['O'], cx + 0.7, cy, cz + 1.2])
                atom_id += 1
        
        return np.array(atoms)
    
    def add_water_layer(self, membrane_atoms, water_thickness=15.0):
        """
        Add water molecules above the membrane.
        
        Args:
            membrane_atoms (np.array): Membrane structure
            water_thickness (float): Thickness of water layer (Angstrom)
        
        Returns:
            np.array: Structure with water
        """
        # Get membrane bounds
        x_min, x_max = membrane_atoms[:, 2].min(), membrane_atoms[:, 2].max()
        y_min, y_max = membrane_atoms[:, 3].min(), membrane_atoms[:, 3].max()
        z_max = membrane_atoms[:, 4].max()
        
        # Estimate number of water molecules
        area = (x_max - x_min) * (y_max - y_min)
        volume = area * water_thickness
        n_water = int(volume * self.params['water_density'] * 0.033)  # Approximate
        
        water_atoms = []
        atom_id = int(membrane_atoms[:, 0].max()) + 1
        
        # Add water molecules randomly in the layer
        for i in range(min(n_water, 50)):  # Limit for demo
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            z = np.random.uniform(z_max + 3, z_max + water_thickness)
            
            # Water oxygen
            water_atoms.append([atom_id, self.atom_types['OW'], x, y, z])
            atom_id += 1
            
            # Two water hydrogens
            for j in range(2):
                hx = x + np.random.uniform(-1, 1)
                hy = y + np.random.uniform(-1, 1)
                hz = z + np.random.uniform(-0.5, 0.5)
                water_atoms.append([atom_id, self.atom_types['HW'], hx, hy, hz])
                atom_id += 1
        
        # Combine membrane and water
        if water_atoms:
            return np.vstack([membrane_atoms, np.array(water_atoms)])
        else:
            return membrane_atoms
    
    def write_lammps_data(self, atoms, filename):
        """
        Write atomic structure to LAMMPS data file.
        
        Args:
            atoms (np.array): Atomic coordinates
            filename (str): Output filename
        """
        # Calculate simulation box
        margin = 5.0
        x_min, x_max = atoms[:, 2].min() - margin, atoms[:, 2].max() + margin
        y_min, y_max = atoms[:, 3].min() - margin, atoms[:, 3].max() + margin
        z_min, z_max = atoms[:, 4].min() - margin, atoms[:, 4].max() + margin
        
        with open(filename, 'w') as f:
            f.write("# LAMMPS data file for GO/rGO membrane simulation\n\n")
            f.write(f"{len(atoms)} atoms\n")
            f.write(f"{len(self.atom_types)} atom types\n\n")
            
            f.write(f"{x_min:.6f} {x_max:.6f} xlo xhi\n")
            f.write(f"{y_min:.6f} {y_max:.6f} ylo yhi\n")
            f.write(f"{z_min:.6f} {z_max:.6f} zlo zhi\n\n")
            
            f.write("Masses\n\n")
            f.write(f"1 {self.params['carbon_mass']:.3f}   # Carbon\n")
            f.write(f"2 {self.params['oxygen_mass']:.3f}   # Oxygen\n")
            f.write(f"3 {self.params['hydrogen_mass']:.3f}   # Hydrogen\n")
            f.write(f"4 {self.params['water_ow_mass']:.3f}   # Water O\n")
            f.write(f"5 {self.params['water_hw_mass']:.3f}   # Water H\n\n")
            
            f.write("Atoms\n\n")
            for atom in atoms:
                atom_id = int(atom[0])
                atom_type = int(atom[1])
                x, y, z = atom[2], atom[3], atom[4]
                f.write(f"{atom_id} {atom_type} {x:.6f} {y:.6f} {z:.6f}\n")
