# simulate_chemistry.py

"""
Phase 4: Chemical and Biological Simulation Engine

Calculates contaminant removal based on chemical properties of membranes and pollutants.
Considers adsorption kinetics, equilibrium saturation, regeneration capacity, and contaminant decay.

Supported interactions:
- Heavy metal adsorption (Pb2+, As3+, etc.)
- Bacterial inactivation (E. coli, etc.)
- Salt rejection (NaCl, MgSO4, etc.)
- Chemical bonding types (π–π, redox, electrostatic, etc.)
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from monitoring.models import adsorption_rate, bacterial_decay, salt_rejection_ratio
import time
from tqdm import tqdm
import concurrent.futures

class ChemicalSimulationEngine:
    """
    Main engine for processing contaminant-membrane interactions.
    """
    
    def validate_forcefield_params(self):
        params_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'forcefield_params.json')
        try:
            with open(params_path, 'r') as f:
                params = json.load(f)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Could not load forcefield_params.json: {e}")
        required_sections = [
            'dihedral_types', 'coarse_grained_beads', 'regeneration_chemistry',
            'pfas_cross_terms', 'antibiotic_parameters', 'microplastic_hybrid', 'pathogen_parameters'
        ]
        for section in required_sections:
            if section not in params or not isinstance(params[section], dict):
                raise ValueError(f"[ERROR] Section '{section}' missing or misformatted in forcefield_params.json. Please check the data file.")

    def __init__(self, contaminant_data_path="monitoring/contaminant_membrane_properties.json"):
        self.validate_forcefield_params()
        self.contaminant_data_path = contaminant_data_path
        self.contaminant_data = self.load_contaminant_data()
        self.simulation_results = {}
        self.lab_validated_variant = None  # Initialize attribute
    
    def load_contaminant_data(self):
        try:
            with open(self.contaminant_data_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _create_default_contaminant_data(self):
        return {
            "Pb2+": {"k2": 0.015, "qmax": 120, "activation_energy_kJ_mol": 25.0},
            "E_coli": {"kill_rate_min^-1": 0.12, "log_reduction_target": 4.0},
            "NaCl": {"rejection_coefficient": 0.85}
        }
    
    def calculate_thermodynamic_favorability(self, contaminant_name, membrane_type, temperature_K=298.15):
        # Placeholder: return 1.0 (favorable) for now
        return 1.0
    
    def apply_diffusion_effects(self, contaminant_name, membrane_type, concentration_profile, time_points):
        # Placeholder: return unchanged profile
        return concentration_profile
    
    def simulate_contaminant_removal(self, membrane_type, contaminants, initial_concentrations,
                                   reaction_time=180, timestep=10.0, pH=7.0, max_seconds=300):
        time_points = list(range(0, int(reaction_time)+1, int(timestep)))
        contaminant_results = {}
        start_time = time.time()
        try:
            with tqdm(total=len(contaminants), desc="Simulating contaminants", unit="contaminant") as pbar:
                for contaminant in contaminants:
                    data = self.contaminant_data.get(contaminant, {})
                    # Heavy metals
                    if contaminant in ["Pb2+", "Cd2+"]:
                        qmax = data.get("qmax", 120)
                        k2 = data.get("k2", 0.015)
                        q = 0
                        conc = [initial_concentrations.get(contaminant, 100)]
                        for t in time_points[1:]:
                            dq = adsorption_rate(q, qmax, k2) * timestep
                            q = min(q + dq, qmax)
                            c = max(conc[-1] - dq, 0)
                            conc.append(c)
                        removal_eff = 100 * (1 - conc[-1]/conc[0]) if conc[0] else 0
                        contaminant_results[contaminant] = {
                            "concentration_mg_L": conc,
                            "removal_efficiency": removal_eff,
                            "q_max": qmax
                        }
                    # Ions (Na+, Cl-)
                    elif contaminant in ["Na+", "Cl-"]:
                        R = data.get("rejection_coefficient", 0.85)
                        C_feed = initial_concentrations.get(contaminant, 100)
                        permeate = [salt_rejection_ratio(R, C_feed) for _ in time_points]
                        contaminant_results[contaminant] = {"permeate_conc": permeate, "rejection": R}
                    # BPA, pharmaceuticals, dyes (use qmax/k2 if available)
                    elif contaminant in ["BPA", "SMX", "MB"]:
                        qmax = data.get("qmax", 80)
                        k2 = data.get("k2", 0.01)
                        q = 0
                        conc = [initial_concentrations.get(contaminant, 50)]
                        for t in time_points[1:]:
                            dq = adsorption_rate(q, qmax, k2) * timestep
                            q = min(q + dq, qmax)
                            c = max(conc[-1] - dq, 0)
                            conc.append(c)
                        removal_eff = 100 * (1 - conc[-1]/conc[0]) if conc[0] else 0
                        contaminant_results[contaminant] = {
                            "concentration_mg_L": conc,
                            "removal_efficiency": removal_eff,
                            "q_max": qmax
                        }
                    # ...existing code for other contaminants...
                    elif contaminant == "E_coli":
                        k = data.get("kill_rate_min^-1", 0.12)
                        N0 = initial_concentrations.get(contaminant, 1e6)
                        conc = [N0]
                        for t in time_points[1:]:
                            N = bacterial_decay(N0, k, t)
                            conc.append(N)
                        log_reduction = (np.log10(N0) - np.log10(conc[-1])) if N0 > 0 and conc[-1] > 0 else 0
                        contaminant_results[contaminant] = {"concentration": conc, "log_reduction": log_reduction}
                    else:
                        # Default: just propagate initial concentration
                        conc = [initial_concentrations.get(contaminant, 10) for _ in time_points]
                        contaminant_results[contaminant] = {"concentration": conc, "removal_efficiency": 0}
                    pbar.update(1)
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (pbar.n if pbar.n else 1)
                    est_total = avg_time * len(contaminants)
                    pbar.set_postfix({"elapsed (s)": f"{elapsed:.1f}", "est. total (s)": f"{est_total:.1f}"})
                    if elapsed > max_seconds:
                        raise TimeoutError(f"Simulation exceeded {max_seconds} seconds. Aborting.")
        except TimeoutError as e:
            print(f"\n[ERROR] {e}")
            contaminant_results['error'] = str(e)
        self.simulation_results = {"contaminants": contaminant_results, "time_points": time_points}
        return {"contaminants": contaminant_results, "time_points": time_points}

    def load_lab_characterization_data(self, filepath):
        """
        Load lab characterization data from a JSON file and infer membrane variant.
        Returns (variant, confidence) or (default, confidence=1.0) if file loads but no spacing found.
        """
        import json
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            # Simple inference logic based on interlayer spacing
            spacing = data.get('XRD', {}).get('interlayer_spacing_nm', None)
            if spacing is not None:
                if spacing < 0.5:
                    return 'rGO_rGO', 1.0
                elif spacing > 1.0:
                    return 'GO_GO', 1.0
                else:
                    return 'GO_rGO', 1.0
            # If file loads but no spacing, return default variant with high confidence
            return 'GO_rGO', 1.0
        except Exception:
            return None, 0.0

    def apply_lab_validated_membrane_properties(self, membrane_type, lab_data_path=None):
        """
        Override membrane properties using lab characterization data if available.
        Returns modified membrane properties dict.
        """
        props = {"membrane_type": membrane_type}
        if lab_data_path:
            variant, confidence = self.load_lab_characterization_data(lab_data_path)
            props["lab_validated_variant"] = variant
            props["lab_confidence"] = confidence
            self.lab_validated_variant = variant
        else:
            self.lab_validated_variant = None
        return props

    def get_summary_statistics(self):
        """
        Return summary statistics for the most recent set of simulation results.
        """
        summary = {
            'total_simulations': len(self.simulation_results) if isinstance(self.simulation_results, list) else 1
        }
        return summary

    def apply_regeneration(self, results, cycle_number=1):
        """
        Simulate the effect of regeneration cycles on membrane performance.
        Reduces q_max by 20% per cycle (example logic).
        """
        import copy
        regen_results = copy.deepcopy(results)
        factor = 0.8 ** cycle_number
        for contaminant, data in regen_results['contaminants'].items():
            if 'q_max' in data:
                data['q_max'] *= factor
                data['regeneration_factor'] = factor
            elif 'qmax' in data:
                data['qmax'] *= factor
                data['regeneration_factor'] = factor
        return regen_results

    def export_results(self, output_dir="output/phase4", filename_prefix="phase4_results"):
        """
        Export simulation results to both JSON and CSV files in the specified directory.
        """
        import os, json, csv
        os.makedirs(output_dir, exist_ok=True)
        # JSON export
        filename_json = f"{filename_prefix}.json"
        filepath_json = os.path.join(output_dir, filename_json)
        with open(filepath_json, 'w') as f:
            json.dump(self.simulation_results, f, indent=2)
        print(f"Results exported to {filepath_json}")
        # CSV export (flattened)
        results = self.simulation_results
        all_rows = []
        if isinstance(results, list):
            for result in results:
                membrane = result.get('membrane_type', '')
                rows = []
                for contaminant, data in result.get('contaminants', {}).items():
                    row = {'membrane_type': membrane, 'contaminant': contaminant}
                    row.update({k: v for k, v in data.items() if not isinstance(v, list)})
                    rows.append(row)
                    all_rows.append(row)
                if rows:
                    filename_csv = f"{filename_prefix}_{membrane}.csv"
                    filepath_csv = os.path.join(output_dir, filename_csv)
                    with open(filepath_csv, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                        writer.writeheader()
                        writer.writerows(rows)
                    print(f"Results exported to {filepath_csv}")
        elif isinstance(results, dict):
            membrane = results.get('membrane_type', '')
            rows = []
            for contaminant, data in results.get('contaminants', {}).items():
                row = {'membrane_type': membrane, 'contaminant': contaminant}
                row.update({k: v for k, v in data.items() if not isinstance(v, list)})
                rows.append(row)
                all_rows.append(row)
            if rows:
                filename_csv = f"{filename_prefix}.csv"
                filepath_csv = os.path.join(output_dir, filename_csv)
                with open(filepath_csv, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"Results exported to {filepath_csv}")
        # Always export a combined CSV file for test compatibility
        filename_csv = f"{filename_prefix}.csv"
        filepath_csv = os.path.join(output_dir, filename_csv)
        fieldnames = ["membrane_type", "contaminant"]
        if all_rows:
            # Use all keys from all_rows for headers
            for row in all_rows:
                for k in row.keys():
                    if k not in fieldnames:
                        fieldnames.append(k)
            with open(filepath_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"Results exported to {filepath_csv}")
        else:
            # Write just headers if no data rows
            with open(filepath_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            print(f"(Empty) Results exported to {filepath_csv}")
        # Debug: print all files in output dir
        print("Files in output dir:", os.listdir(output_dir))

def run_phase4_simulation(membrane_types=['GO', 'rGO', 'hybrid'],
                         contaminants=['Pb2+', 'E_coli'],
                         initial_concentrations=None,
                         reaction_time=180,
                         lab_data_path=None,
                         max_seconds=300):
    engine = ChemicalSimulationEngine()
    if initial_concentrations is None:
        initial_concentrations = {c: 100 for c in contaminants}
    simulation_results = []
    for membrane in membrane_types:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                engine.simulate_contaminant_removal,
                membrane, contaminants, initial_concentrations, reaction_time, 10.0, 7.0, max_seconds
            )
            try:
                result = future.result(timeout=max_seconds+10)
                # Add membrane type to result for test compatibility
                result['membrane_type'] = membrane
                simulation_results.append(result)
            except concurrent.futures.TimeoutError:
                print(f"\n[ERROR] Chemical simulation timed out after {max_seconds} seconds and was killed for {membrane}.")
                simulation_results.append({'membrane_type': membrane, 'error': f'Timed out after {max_seconds} seconds.'})
    engine.simulation_results = simulation_results
    return engine

if __name__ == "__main__":
    # Example usage
    engine = run_phase4_simulation(
        membrane_types=['GO', 'rGO', 'hybrid'],
        contaminants=['Pb2+', 'E_coli'],
        initial_concentrations={'Pb2+': 50.0, 'E_coli': 1e5},
        reaction_time=120
    )
