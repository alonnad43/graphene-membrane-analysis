# data_builder.py

"""
Builds LAMMPS-compatible atomic structures for GO, rGO, and hybrid membranes.

Creates realistic molecular representations based on empirical data from Phase 1.
Focuses on physical structure generation without chemical reaction modeling.
"""

import numpy as np
import os
import json
from src.properties import LAMMPS_PARAMETERS

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
        Create atomic structure for a membrane based on its properties, using advanced structure generation.
        
        Args:
            membrane: Membrane object from Phase 1
            
        Returns:
            np.array: Atomic structure [atom_id, type, x, y, z]
        """
        # Use multilayer if thickness > 1 layer (assume 3.4 Å per layer)
        n_layers = max(1, int(round(membrane.thickness_nm * 10 / 3.4)))
        nx, ny = 5, 5
        # Build multilayer graphene/GO
        atoms = self.create_multilayer_graphene(nx, ny, n_layers=n_layers, interlayer=3.4)
        # Add edge capping
        atoms = self.cap_edges_with_hydrogen(atoms)
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
        atoms = self.add_water_layer(atoms, water_thickness=15.0)
        
        # Add small random displacement to all atoms
        atoms = self.randomize_positions(atoms, max_disp=0.05)
        
        return atoms
    
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
        
        # Smarter placement: only add O/H if no overlap with existing atoms (min_dist = 1.2 Å)
        min_dist = 1.2
        for idx in oxidized_indices:
            carbon = carbon_atoms[idx]
            cx, cy, cz = carbon[2], carbon[3], carbon[4]
            # Add hydroxyl group (-OH) above carbon
            if np.random.random() < 0.7:  # 70% hydroxyl groups
                ox, oy, oz = cx, cy, cz + 1.4
                hx, hy, hz = cx, cy, cz + 2.0
                # Check for overlaps before adding O
                if all(np.linalg.norm(np.array([ox, oy, oz]) - np.array(a[2:5])) >= min_dist for a in atoms):
                    atoms.append([atom_id, self.atom_types['O'], ox, oy, oz])
                    atom_id += 1
                    # Check for overlaps before adding H
                    if all(np.linalg.norm(np.array([hx, hy, hz]) - np.array(a[2:5])) >= min_dist for a in atoms):
                        atoms.append([atom_id, self.atom_types['H'], hx, hy, hz])
                        atom_id += 1
            else:  # 30% epoxy groups (simplified as oxygen bridge)
                ox, oy, oz = cx + 0.7, cy, cz + 1.2
                if all(np.linalg.norm(np.array([ox, oy, oz]) - np.array(a[2:5])) >= min_dist for a in atoms):
                    atoms.append([atom_id, self.atom_types['O'], ox, oy, oz])
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
        Add water molecules above the membrane, avoiding overlaps with membrane and other water atoms.
        """
        x_min, x_max = membrane_atoms[:, 2].min(), membrane_atoms[:, 2].max()
        y_min, y_max = membrane_atoms[:, 3].min(), membrane_atoms[:, 3].max()
        z_max = membrane_atoms[:, 4].max()
        area = (x_max - x_min) * (y_max - y_min)
        volume = area * water_thickness
        n_water = int(volume * LAMMPS_PARAMETERS['water_density'] * 0.033)
        water_atoms = []
        atom_id = int(membrane_atoms[:, 0].max()) + 1
        min_dist = 2.5
        for i in range(min(n_water, 50)):
            tries = 0
            while tries < 20:
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                z = np.random.uniform(z_max + 3, z_max + water_thickness)
                pos = np.array([x, y, z])
                # Check distance to all membrane atoms and existing water oxygens
                if all(np.linalg.norm(pos - a[2:5]) >= min_dist for a in membrane_atoms) and \
                   all(np.linalg.norm(pos - a[2:5]) >= min_dist for a in water_atoms if a[1] == self.atom_types['OW']):
                    water_atoms.append([atom_id, self.atom_types['OW'], x, y, z])
                    atom_id += 1
                    # Add two hydrogens (randomly placed nearby)
                    for j in range(2):
                        tries_h = 0
                        while tries_h < 10:
                            hx = x + np.random.uniform(-1, 1)
                            hy = y + np.random.uniform(-1, 1)
                            hz = z + np.random.uniform(-0.5, 0.5)
                            hpos = np.array([hx, hy, hz])
                            if all(np.linalg.norm(hpos - a[2:5]) >= min_dist for a in membrane_atoms) and \
                               all(np.linalg.norm(hpos - a[2:5]) >= min_dist for a in water_atoms):
                                water_atoms.append([atom_id, self.atom_types['HW'], hx, hy, hz])
                                atom_id += 1
                                break
                            tries_h += 1
                    break
                tries += 1
        if water_atoms:
            return np.vstack([membrane_atoms, np.array(water_atoms)])
        else:
            return membrane_atoms
    
    def create_multilayer_graphene(self, nx=5, ny=5, n_layers=2, interlayer=3.4):
        """
        Create a multilayer graphene/GO stack with realistic interlayer spacing.
        Args:
            nx, ny (int): Number of unit cells per layer
            n_layers (int): Number of layers
            interlayer (float): Interlayer distance (Å)
        Returns:
            np.array: All atom coordinates
        """
        all_atoms = []
        atom_id = 1
        for l in range(n_layers):
            z = l * interlayer
            layer_atoms = self.create_graphene_sheet(nx, ny, z)
            # Offset atom IDs for each layer
            for atom in layer_atoms:
                all_atoms.append([atom_id, atom[1], atom[2], atom[3], atom[4]])
                atom_id += 1
        return np.array(all_atoms)

    def cap_edges_with_hydrogen(self, atoms, bond_cutoff=1.7):
        """
        Add hydrogens to edge carbons (carbons with <3 neighbors).
        Args:
            atoms (np.array): Atom array
            bond_cutoff (float): Max C–C bond length
        Returns:
            np.array: Atom array with edge hydrogens
        """
        atoms = atoms.tolist()
        carbons = [a for a in atoms if a[1] == self.atom_types['C']]
        edge_carbons = []
        for c in carbons:
            c_id, _, cx, cy, cz = c
            neighbors = 0
            for c2 in carbons:
                if c2[0] != c_id and np.linalg.norm(np.array([cx, cy, cz]) - np.array(c2[2:5])) < bond_cutoff:
                    neighbors += 1
            if neighbors < 3:
                edge_carbons.append(c)
        atom_id = int(max(a[0] for a in atoms)) + 1
        for c in edge_carbons:
            cx, cy, cz = c[2], c[3], c[4]
            # Place H 1.1 Å above the edge carbon (simple model)
            atoms.append([atom_id, self.atom_types['H'], cx, cy, cz + 1.1])
            atom_id += 1
        return np.array(atoms)

    def randomize_positions(self, atoms, max_disp=0.05):
        """
        Add a small random displacement to all atom positions.
        Args:
            atoms (np.array): Atom array
            max_disp (float): Maximum displacement (Å)
        Returns:
            np.array: Displaced atom array
        """
        atoms = atoms.copy()
        atoms[:, 2:5] += np.random.uniform(-max_disp, max_disp, atoms[:, 2:5].shape)
        return atoms

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
        margin = 15.0  # Increased from 10.0 to 15.0 for extra safety
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
        # Write header with correct counts
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
            f.write(f"{atoms[:,2].min()-margin:.6f} {atoms[:,2].max()+margin:.6f} xlo xhi\n")
            f.write(f"{atoms[:,3].min()-margin:.6f} {atoms[:,3].max()+margin:.6f} ylo yhi\n")
            f.write(f"{atoms[:,4].min()-margin:.6f} {atoms[:,4].max()+margin:.6f} zlo zhi\n\n")
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
        return
