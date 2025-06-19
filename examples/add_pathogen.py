"""
Example: Add a new pathogen to the forcefield_params.json data file.
"""
import json
import os

DATA_PATH = os.path.join('data', 'forcefield_params.json')

new_pathogen = {
    "E_coli": {
        "diameter_nm": 0.5,
        "length_nm": 2.0,
        "charge": -1,
        "interaction_strength": 1.2
    }
}

def add_pathogen():
    with open(DATA_PATH, 'r') as f:
        params = json.load(f)
    if 'pathogen_parameters' not in params:
        params['pathogen_parameters'] = {}
    params['pathogen_parameters'].update(new_pathogen)
    with open(DATA_PATH, 'w') as f:
        json.dump(params, f, indent=2)
    print("Added E. coli to pathogen_parameters.")

if __name__ == "__main__":
    add_pathogen()
