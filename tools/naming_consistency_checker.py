"""
Automated checker for parameter and variable naming consistency across the graphene membrane simulation codebase.
Scans Python files and JSON for common membrane/forcefield/variant property names and reports inconsistencies.
"""
import os
import re
import json

# Standardized property names
MEMBRANE_KEYS = [
    'name', 'membrane_name', 'membrane_type', 'thickness_nm', 'pore_size_nm', 'flux_lmh',
    'modulus_GPa', 'tensile_strength_MPa', 'contact_angle_deg', 'rejection_percent', 'variant'
]
FORCEFIELD_KEYS = [
    'atom_types', 'bond_types', 'angle_types', 'dihedral_types', 'cross_terms',
    'id', 'k', 'r0', 'theta0', 'K1', 'K2', 'K3', 'K4', 'sigma', 'epsilon', 'charge', 'mass'
]

CODE_EXTS = ['.py']
JSON_EXTS = ['.json']


def scan_file_for_keys(filepath, keys):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    found = set()
    for key in keys:
        if re.search(rf'\b{re.escape(key)}\b', text):
            found.add(key)
    return found

def scan_codebase(root_dir, keys, exts):
    found = set()
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if any(fname.endswith(ext) for ext in exts):
                fpath = os.path.join(dirpath, fname)
                found |= scan_file_for_keys(fpath, keys)
    return found

def check_json_keys(json_path, keys):
    with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)
    found = set()
    def recursive_check(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in keys:
                    found.add(k)
                recursive_check(v)
        elif isinstance(obj, list):
            for item in obj:
                recursive_check(item)
    recursive_check(data)
    return found

if __name__ == "__main__":
    root = os.path.dirname(__file__)
    print("\n[CHECK] Scanning codebase for membrane/forcefield property names...")
    code_keys = scan_codebase(root, MEMBRANE_KEYS + FORCEFIELD_KEYS, CODE_EXTS)
    print(f"Found in code: {sorted(code_keys)}")
    # Check main forcefield JSON
    ff_json = os.path.join(root, 'forcefield_and_simulation_data.json')
    if os.path.exists(ff_json):
        json_keys = check_json_keys(ff_json, MEMBRANE_KEYS + FORCEFIELD_KEYS)
        print(f"Found in forcefield JSON: {sorted(json_keys)}")
    # Report missing/extra
    missing_in_code = set(MEMBRANE_KEYS + FORCEFIELD_KEYS) - code_keys
    missing_in_json = set(MEMBRANE_KEYS + FORCEFIELD_KEYS) - json_keys if 'json_keys' in locals() else set()
    if missing_in_code:
        print(f"[WARNING] Not found in code: {sorted(missing_in_code)}")
    if missing_in_json:
        print(f"[WARNING] Not found in forcefield JSON: {sorted(missing_in_json)}")
    print("\n[CHECK COMPLETE]")
