# data_builder.py

"""
Builds LAMMPS-compatible atomic structures for GO, rGO, and hybrid membranes.

Creates realistic molecular representations based on empirical data from Phase 1.
Focuses on physical structure generation without chemical reaction modeling.
"""

import numpy as np
import os
import json
from properties import LAMMPS_PARAMETERS

class LAMMPSDataBuilder:
    """
    Builds atomic structures for LAMMPS simulations.
    """
    
    def __init__(self, forcefield_path="data/forcefield_params.json"):
        with open(forcefield_path, "r") as f:
            self.forcefield = json.load(f)
        self.atom_types = {k: v["id"] for k, v in self.forcefield["atom_types"].items()}
        self.bond_types = {k: v["id"] for k, v in self.forcefield["bond_types"].items()}
        self.angle_types = {k: v["id"] for k, v in self.forcefield["angle_types"].items()}
        self.atom_params = self.forcefield["atom_types"]
        self.bond_params = self.forcefield["bond_types"]
        self.angle_params = self.forcefield["angle_types"]
    
    def create_membrane_structure(self, membrane):
        """
        Create atomic structure for a membrane based on its properties.
        
        Args:
            membrane: Membrane object from Phase 1
            
        Returns:
            np.array: Atomic structure [atom_id, type, x, y, z]
        """
        nx, ny = 5, 5
        layer_z = 0.0
        
        # Create base graphene structure
        atoms = self.create_graphene_sheet(nx, ny, layer_z)
        
        # Add functional groups for GO
        if "GO" in membrane.name:
            atoms = self.add_go_functional_groups(atoms, oxidation_ratio=0.2)
        
        # Add PEI branches if requested
        if getattr(membrane, 'pei_branches', False):
            atoms = self.add_pei_branches(atoms, pei_density=getattr(membrane, 'pei_density', 0.1))
        
        # Add ions if requested
        if hasattr(membrane, 'ions') and membrane.ions:
            atoms = self.add_ions(atoms, membrane.ions)
        
        # Add contaminants if requested
        if hasattr(membrane, 'contaminants') and membrane.contaminants:
            atoms = self.add_contaminants(atoms, membrane.contaminants)
        
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
    
    def add_pei_branches(self, atoms, pei_density=0.1):
        """
        Add PEI (polyethylenimine) branches to the membrane structure.
        
        Args:
            atoms (np.array): Current atom array
            pei_density (float): Fraction of carbons to functionalize
        
        Returns:
            np.array: Extended atom array
        """
        # Stub: randomly add N_PEI, C_PEI, H_PEI near surface
        atoms = atoms.tolist()
        n_carbons = sum(1 for a in atoms if a[1] == self.atom_types['C'])
        n_pei = int(n_carbons * pei_density)
        pei_indices = np.random.choice([i for i,a in enumerate(atoms) if a[1]==self.atom_types['C']], n_pei, replace=False)
        atom_id = int(max(a[0] for a in atoms)) + 1
        for idx in pei_indices:
            c = atoms[idx]
            x, y, z = c[2], c[3], c[4]
            # Add N_PEI
            atoms.append([atom_id, self.atom_types['N_PEI'], x, y, z+1.5])
            atom_id += 1
            # Add C_PEI
            atoms.append([atom_id, self.atom_types['C_PEI'], x+0.7, y, z+1.7])
            atom_id += 1
            # Add H_PEI
            atoms.append([atom_id, self.atom_types['H_PEI'], x+0.7, y, z+2.2])
            atom_id += 1
        return np.array(atoms)

    def add_ions(self, atoms, ions):
        """
        Add ions (e.g., Na+, Cl-, Pb2+, Cd2+) to the simulation box.
        
        Args:
            atoms (np.array): Current atom array
            ions (dict): {ion_type: count}
        
        Returns:
            np.array: Extended atom array
        """
        atoms = atoms.tolist()
        atom_id = int(max(a[0] for a in atoms)) + 1
        # Place ions randomly above membrane
        z_max = max(a[4] for a in atoms)
        x_vals = [a[2] for a in atoms]
        y_vals = [a[3] for a in atoms]
        for ion_type, count in ions.items():
            for _ in range(count):
                x = np.random.uniform(min(x_vals), max(x_vals))
                y = np.random.uniform(min(y_vals), max(y_vals))
                z = np.random.uniform(z_max+2, z_max+10)
                if ion_type in self.atom_types:
                    atoms.append([atom_id, self.atom_types[ion_type], x, y, z])
                    atom_id += 1
        return np.array(atoms)

    def add_contaminants(self, atoms, contaminants):
        """
        Add contaminant molecules (e.g., BPA, pharmaceuticals, dyes) to the simulation box.
        
        Args:
            atoms (np.array): Current atom array
            contaminants (dict): {contaminant_type: count}
        
        Returns:
            np.array: Extended atom array
        """
        atoms = atoms.tolist()
        atom_id = int(max(a[0] for a in atoms)) + 1
        z_max = max(a[4] for a in atoms)
        x_vals = [a[2] for a in atoms]
        y_vals = [a[3] for a in atoms]
        for cont_type, count in contaminants.items():
            for _ in range(count):
                x = np.random.uniform(min(x_vals), max(x_vals))
                y = np.random.uniform(min(y_vals), max(y_vals))
                z = np.random.uniform(z_max+2, z_max+10)
                if cont_type in self.atom_types:
                    atoms.append([atom_id, self.atom_types[cont_type], x, y, z])
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
        n_water = int(volume * LAMMPS_PARAMETERS['water_density'] * 0.033)  # Approximate
        
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
    
    def get_atom_charge(self, atom_type):
        for k, v in self.atom_params.items():
            if v["id"] == atom_type:
                return v.get("charge", 0.0)
        return 0.0

    def get_dihedral_type(self, atom_types):
        """
        Given a tuple of four atom type IDs, return a string key representing the dihedral type (e.g., 'C-C-C-C').
        """
        id_to_symbol = {v['id']: k for k, v in self.forcefield['atom_types'].items()}
        return '-'.join([id_to_symbol[tid] for tid in atom_types])

    def write_lammps_data(self, atoms, filename):
        """
        Write atomic structure to LAMMPS data file in the format:
        atom-ID molecule-ID atom-type charge x y z
        Also writes Bonds, Angles, and Dihedrals for water and GO molecules.
        """
        # Calculate simulation box
        margin = 5.0
        x_min, x_max = atoms[:, 2].min() - margin, atoms[:, 2].max() + margin
        y_min, y_max = atoms[:, 3].min() - margin, atoms[:, 3].max() + margin
        z_min, z_max = atoms[:, 4].min() - margin, atoms[:, 4].max() + margin
        
        # Identify water and GO bonds/angles
        water_oxygens = [atom for atom in atoms if int(atom[1]) == 4]
        water_hydrogens = [atom for atom in atoms if int(atom[1]) == 5]
        go_oxygens = [atom for atom in atoms if int(atom[1]) == 2]
        go_hydrogens = [atom for atom in atoms if int(atom[1]) == 3]
        carbons = [atom for atom in atoms if int(atom[1]) == 1]
        bonds = []
        angles = []
        dihedrals = []
        dihedral_types_set = set()
        dihedral_type_map = {}
        bond_id = 1
        angle_id = 1
        dihedral_id = 1
        # Bond/angle type mapping:
        # 1: C-C, 2: C-O, 3: O-H (GO), 4: OW-HW (water)
        # Angle types: 1: C-C-C, 2: C-C-O, 3: C-O-H, 4: HW-OW-HW
        # Dihedral types: 1: C-C-C-C, 2: C-C-C-O, 3: C-C-O-H, etc.
        # Water bonds/angles
        for i, o_atom in enumerate(water_oxygens):
            o_id = int(o_atom[0])
            h_candidates = sorted(water_hydrogens, key=lambda h: (h[2]-o_atom[2])**2 + (h[3]-o_atom[3])**2 + (h[4]-o_atom[4])**2)
            h1_id = int(h_candidates[0][0])
            h2_id = int(h_candidates[1][0])
            bonds.append([bond_id, 4, o_id, h1_id])  # OW-HW
            bond_id += 1
            bonds.append([bond_id, 4, o_id, h2_id])
            bond_id += 1
            angles.append([angle_id, 4, h1_id, o_id, h2_id])  # HW-OW-HW
            angle_id += 1
            water_hydrogens = [h for h in water_hydrogens if int(h[0]) not in (h1_id, h2_id)]
        # GO bonds/angles (simple: C-C, C-O, O-H, C-C-C, C-C-O, C-O-H)
        # C-C bonds
        for i, c1 in enumerate(carbons):
            for j, c2 in enumerate(carbons):
                if i < j:
                    dist = np.linalg.norm(np.array(c1[2:5]) - np.array(c2[2:5]))
                    if abs(dist - 1.44) < 0.1:
                        bonds.append([bond_id, 1, int(c1[0]), int(c2[0])])
                        bond_id += 1
        # C-O and O-H bonds (GO)
        for o in go_oxygens:
            # Find closest carbon
            c = min(carbons, key=lambda c: np.linalg.norm(np.array(c[2:5]) - np.array(o[2:5])))
            if np.linalg.norm(np.array(c[2:5]) - np.array(o[2:5])) < 1.5:
                bonds.append([bond_id, 2, int(c[0]), int(o[0])])
                bond_id += 1
            # If hydroxyl, find attached H
            h = [h for h in go_hydrogens if abs(h[2]-o[2])<0.2 and abs(h[3]-o[3])<0.2]
            if h:
                bonds.append([bond_id, 3, int(o[0]), int(h[0][0])])
                bond_id += 1
        # Angles for GO (C-C-C, C-C-O, C-O-H)
        for c1 in carbons:
            for c2 in carbons:
                for c3 in carbons:
                    if len({c1[0],c2[0],c3[0]})==3:
                        d1 = np.linalg.norm(np.array(c1[2:5])-np.array(c2[2:5]))
                        d2 = np.linalg.norm(np.array(c2[2:5])-np.array(c3[2:5]))
                        if abs(d1-1.44)<0.1 and abs(d2-1.44)<0.1:
                            angles.append([angle_id, 1, int(c1[0]), int(c2[0]), int(c3[0])])
                            angle_id += 1
        for c1 in carbons:
            for c2 in carbons:
                for o in go_oxygens:
                    if len({c1[0],c2[0],o[0]})==3:
                        d1 = np.linalg.norm(np.array(c1[2:5])-np.array(c2[2:5]))
                        d2 = np.linalg.norm(np.array(c2[2:5])-np.array(o[2:5]))
                        if abs(d1-1.44)<0.1 and abs(d2-1.42)<0.2:
                            angles.append([angle_id, 2, int(c1[0]), int(c2[0]), int(o[0])])
                            angle_id += 1
        for c in carbons:
            for o in go_oxygens:
                for h in go_hydrogens:
                    if len({c[0],o[0],h[0]})==3:
                        d1 = np.linalg.norm(np.array(c[2:5])-np.array(o[2:5]))
                        d2 = np.linalg.norm(np.array(o[2:5])-np.array(h[2:5]))
                        if abs(d1-1.42)<0.2 and abs(d2-0.96)<0.2:
                            angles.append([angle_id, 3, int(c[0]), int(o[0]), int(h[0])])
                            angle_id += 1
        # Dihedrals for graphene/GO (robust: detect all unique types)
        for c1 in carbons:
            for c2 in carbons:
                for c3 in carbons:
                    for c4 in carbons + go_oxygens + go_hydrogens:
                        if len({c1[0],c2[0],c3[0],c4[0]})==4:
                            d1 = np.linalg.norm(np.array(c1[2:5])-np.array(c2[2:5]))
                            d2 = np.linalg.norm(np.array(c2[2:5])-np.array(c3[2:5]))
                            d3 = np.linalg.norm(np.array(c3[2:5])-np.array(c4[2:5]))
                            if abs(d1-1.44)<0.1 and abs(d2-1.44)<0.1 and abs(d3-1.44)<0.1:
                                atom_types = (int(c1[1]), int(c2[1]), int(c3[1]), int(c4[1]))
                                dihedral_type_key = self.get_dihedral_type(atom_types)
                                if dihedral_type_key not in dihedral_type_map:
                                    dihedral_type_map[dihedral_type_key] = len(dihedral_type_map) + 1
                                dihedral_type_id = dihedral_type_map[dihedral_type_key]
                                dihedrals.append([dihedral_id, dihedral_type_id, int(c1[0]), int(c2[0]), int(c3[0]), int(c4[0])])
                                dihedral_id += 1
        # Write header with dihedrals
        with open(filename, 'w') as f:
            f.write("# LAMMPS data file for GO/rGO membrane simulation\n\n")
            f.write(f"{len(atoms)} atoms\n")
            f.write(f"{len(self.atom_types)} atom types\n")
            f.write(f"{len(bonds)} bonds\n")
            f.write(f"{len(self.bond_types)} bond types\n")
            f.write(f"{len(angles)} angles\n")
            f.write(f"{len(self.angle_types)} angle types\n")
            f.write(f"{len(dihedrals)} dihedrals\n")
            f.write(f"{len(dihedral_type_map)} dihedral types\n\n")
            f.write(f"{atoms[:,2].min()-5:.6f} {atoms[:,2].max()+5:.6f} xlo xhi\n")
            f.write(f"{atoms[:,3].min()-5:.6f} {atoms[:,3].max()+5:.6f} ylo yhi\n")
            f.write(f"{atoms[:,4].min()-5:.6f} {atoms[:,4].max()+5:.6f} zlo zhi\n\n")
            f.write("Masses\n\n")
            for atom_type, params in self.atom_params.items():
                mass = params["mass"]
                f.write(f"{params['id']} {mass:.6f}   # {atom_type}\n")
            f.write("Atoms\n\n")
            for atom in atoms:
                atom_id = int(atom[0])
                molecule_id = 1
                atom_type = int(atom[1])
                charge = self.get_atom_charge(atom_type)
                x, y, z = atom[2], atom[3], atom[4]
                f.write(f"{atom_id} {molecule_id} {atom_type} {charge:.5f} {x:.6f} {y:.6f} {z:.6f}\n")
            if bonds:
                f.write("\nBonds\n\n")
                for b in bonds:
                    f.write(f"{b[0]} {b[1]} {b[2]} {b[3]}\n")
            if angles:
                f.write("\nAngles\n\n")
                for a in angles:
                    f.write(f"{a[0]} {a[1]} {a[2]} {a[3]} {a[4]}\n")
            if dihedrals:
                f.write("\nDihedrals\n\n")
                for d in dihedrals:
                    f.write(f"{d[0]} {d[1]} {d[2]} {d[3]} {d[4]} {d[5]}\n")
            # Optionally, write a mapping of dihedral type IDs to type keys for reference
            if dihedral_type_map:
                f.write("\n# Dihedral type mapping:\n")
                for k, v in dihedral_type_map.items():
                    f.write(f"# {v}: {k}\n")
