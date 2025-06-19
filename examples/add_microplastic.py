# Example: Add a New Microplastic Material to the Force Field
import json

# Load forcefield
with open('data/forcefield_params.json', 'r') as f:
    ff = json.load(f)

# Add a new microplastic (e.g., Polypropylene, PP)
ff.setdefault('microplastic_hybrid', {})['PP'] = {
    "sigma": 4.10,
    "epsilon": 0.09,
    "hydration_energy": -13.5
}

# Save back
with open('data/forcefield_params.json', 'w') as f:
    json.dump(ff, f, indent=2)

print("Added Polypropylene (PP) to microplastic_hybrid section.")
