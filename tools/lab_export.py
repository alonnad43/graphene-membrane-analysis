"""
Exports simulation specs/results to a lab-ready CSV format with column descriptions for technicians.
"""
import csv
import json
import sys

COLUMN_DESCRIPTIONS = {
    "membrane": "Membrane type or variant",
    "flux_lmh": "Simulated water flux (L/m^2/h)",
    "rejection_percent": "Simulated oil/contaminant rejection (%)",
    "CO_ratio": "Simulated C/O ratio (XPS)",
    "ID_IG_ratio": "Simulated Raman I_D/I_G ratio",
    "modulus_GPa": "Young's modulus (GPa)",
    "tensile_strength_MPa": "Tensile strength (MPa)"
}

def export_to_csv(results_json, output_csv):
    with open(results_json, 'r') as f:
        results = json.load(f)
    if isinstance(results, dict) and 'results' in results:
        data = results['results']
    else:
        data = results
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMN_DESCRIPTIONS.keys())
        writer.writeheader()
        for row in data:
            writer.writerow({k: row.get(k, '') for k in COLUMN_DESCRIPTIONS.keys()})
    # Write column descriptions as a separate file
    with open(output_csv.replace('.csv', '_columns.txt'), 'w') as f:
        for k, v in COLUMN_DESCRIPTIONS.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python lab_export.py <results.json> <output.csv>")
        sys.exit(1)
    export_to_csv(sys.argv[1], sys.argv[2])
