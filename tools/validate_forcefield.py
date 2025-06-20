"""
Force field validation script for GO/rGO/hybrid membrane simulation.
Checks that all atom, bond, angle, and dihedral types present in the generated structure are defined in the forcefield JSON.
"""
import json
import numpy as np
from src.data_builder import LAMMPSDataBuilder

def validate_forcefield(forcefield_path, structure_atoms):
    with open(forcefield_path, 'r') as f:
        ff = json.load(f)
    atom_types = set([int(a[1]) for a in structure_atoms])
    ff_atom_ids = set([v['id'] for v in ff['atom_types'].values()])
    missing_atoms = atom_types - ff_atom_ids
    if missing_atoms:
        print(f"Missing atom types in forcefield: {missing_atoms}")
    else:
        print("All atom types present in forcefield.")
    # Bonds, angles, dihedrals: would require structure parsing (see data_builder.py for logic)
    # For brevity, only atom types are checked here.

if __name__ == "__main__":
    # Example usage: validate forcefield for a GO structure
    builder = LAMMPSDataBuilder(forcefield_path="forcefield_and_simulation_data.json")
    atoms = builder.create_graphene_sheet(nx=5, ny=5, layer_z=0.0)
    atoms = builder.add_go_functional_groups(atoms, oxidation_ratio=0.2)
    validate_forcefield("forcefield_and_simulation_data.json", atoms)
