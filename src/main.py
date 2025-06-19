# main.py
"""
Main entry point for the GO/rGO membrane simulation project – all phases combined.

This script orchestrates the complete simulation workflow:
1. Phase 1: Runs macroscale simulations of GO, rGO, and hybrid membranes.
2. Phase 2: Constructs realistic hybrid GO/rGO membrane structures.
3. Phase 3: Runs atomistic simulations using LAMMPS.
4. Phase 4: Simulates chemical and biological interactions.

# TODO: Connect each membrane object across all phases
# TODO: Output membrane_summary.csv with macro + nano data
"""

# Phase Execution Order:
# 1 → macroscale physical simulation
# 2 → hybrid structure optimization
# 3 → LAMMPS atomistic simulation
# 4 → contaminant removal & regeneration modeling

from membrane_model import Membrane, generate_membrane_variants_with_variability
from flux_simulator import simulate_flux_with_variability
from oil_rejection import simulate_oil_rejection_stochastic
from plot_utils import (
    plot_rejection_summary,
    plot_flux_vs_thickness_at_pressure,
    plot_flux_vs_pore_size_at_pressure
)
from properties import MEMBRANE_TYPES, PRESSURE_RANGE, WATER_PROPERTIES
from hybrid_structure import (
    run_phase2_analysis,
    create_alternating_structure, 
    create_sandwich_structure, 
    optimize_structure_for_flux
)
import numpy as np
import os
import pandas as pd
import sys
import json
import argparse

# Add src directory to sys.path so 'monitoring' can be imported from anywhere
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def validate_forcefield_params():
    params_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'forcefield_params.json')
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
    except Exception as e:
        print(f"\n[ERROR] Could not load forcefield_params.json: {e}")
        exit(1)
    required_sections = [
        'dihedral_types', 'coarse_grained_beads', 'regeneration_chemistry',
        'pfas_cross_terms', 'antibiotic_parameters', 'microplastic_hybrid', 'pathogen_parameters'
    ]
    for section in required_sections:
        if section not in params or not isinstance(params[section], dict):
            print(f"\n[ERROR] Section '{section}' missing or misformatted in forcefield_params.json. Please check the data file.")
            exit(1)

def generate_membrane_variants():
    """
    Generate all membrane variants (GO, rGO, hybrid) with realistic variability.
    Returns a list of Membrane objects.
    """
    membranes = []
    np.random.seed(42)  # For reproducibility; remove for production
    for mem_type in ['GO', 'rGO', 'hybrid']:
        props = MEMBRANE_TYPES[mem_type]
        for thickness in props['thicknesses']:
            for pore_size in props['pore_sizes']:
                # Generate 3 variants per config for variability
                noisy_variants = generate_membrane_variants_with_variability(
                    design_pore_size=pore_size,
                    design_thickness=thickness,
                    literature_CA=props['contact_angle_deg'],
                    n=3
                )
                for noisy in noisy_variants:
                    membranes.append(Membrane(
                        name=mem_type,
                        pore_size_nm=noisy['pore_size_nm'],
                        thickness_nm=noisy['thickness_nm'],
                        flux_lmh=None,  # Will be set after flux calculation
                        modulus_GPa=props['youngs_modulus_GPa'],
                        tensile_strength_MPa=None,
                        contact_angle_deg=noisy['contact_angle_deg'],
                        variant=None
                    ))
    return membranes

def main():
    parser = argparse.ArgumentParser(description="GO/rGO membrane simulation workflow")
    parser.add_argument('--single', action='store_true', help='Run a single simulation for testing')
    parser.add_argument('--membrane', type=str, default=None, help='Membrane type to simulate (GO, rGO, hybrid)')
    parser.add_argument('--thickness', type=float, default=None, help='Membrane thickness (nm)')
    parser.add_argument('--pore', type=float, default=None, help='Membrane pore size (nm)')
    args = parser.parse_args()

    validate_forcefield_params()

    # --- Output directory structure ---
    output_base = os.path.join(os.getcwd(), "output")
    phase1_dir = os.path.join(output_base, "phase1")
    phase2_dir = os.path.join(output_base, "phase2")
    phase3_dir = os.path.join(output_base, "phase3")
    phase4_dir = os.path.join(output_base, "phase4")
    os.makedirs(phase1_dir, exist_ok=True)
    os.makedirs(phase2_dir, exist_ok=True)
    os.makedirs(phase3_dir, exist_ok=True)
    os.makedirs(phase4_dir, exist_ok=True)

    if args.single:
        # Single simulation mode
        mem_type = args.membrane or 'GO'
        thickness = args.thickness or MEMBRANE_TYPES[mem_type]['thicknesses'][0]
        pore_size = args.pore or MEMBRANE_TYPES[mem_type]['pore_sizes'][0]
        ca = MEMBRANE_TYPES[mem_type]['contact_angle_deg']
        mem = Membrane(
            name=mem_type,
            pore_size_nm=pore_size,
            thickness_nm=thickness,
            flux_lmh=None,
            modulus_GPa=MEMBRANE_TYPES[mem_type]['youngs_modulus_GPa'],
            tensile_strength_MPa=None,
            contact_angle_deg=ca,
            variant=None
        )
        membranes = [mem]
    else:
        membranes = generate_membrane_variants()

    results = []
    graphs_base = os.path.join(os.getcwd(), 'graphs')
    summary_dir = os.path.join(graphs_base, 'oil_rejection_summary')
    thickness_dir = os.path.join(graphs_base, 'flux_vs_thickness_per_pressure')
    pore_dir = os.path.join(graphs_base, 'flux_vs_pore_size_per_pressure')

    # Simulate flux and rejection at different pressures
    for mem in membranes:
        mat_folder = mem.name.split()[0] if mem.name != "Hybrid" else "Hybrid"
        mat_dir = os.path.join(output_base, mat_folder)
        flux_dir = os.path.join(mat_dir, "flux_vs_pressure")
        fluxes = []
        for p in PRESSURE_RANGE:
            flux = simulate_flux_with_variability(
                {'pore_size_nm': mem.pore_size_nm, 'thickness_nm': mem.thickness_nm, 'contact_angle_deg': mem.contact_angle_deg},
                pressure_bar=p,
                viscosity_pas=WATER_PROPERTIES["viscosity_25C"],
                porosity=WATER_PROPERTIES["porosity"],
                tortuosity=WATER_PROPERTIES["tortuosity"]
            )
            fluxes.append(flux)
            if p == PRESSURE_RANGE[0]:
                print(f"  Flux validation for {mem.name} at {p} bar: {flux:.1f} L·m⁻²·h⁻¹")
        mem.flux_lmh = np.mean(fluxes)  # Store average flux for summary
        for p, flux in zip(PRESSURE_RANGE, fluxes):
            results.append({
                "material": mat_folder,
                "membrane_name": mem.name,
                "pressure_bar": p,
                "thickness_nm": mem.thickness_nm,
                "pore_size_nm": mem.pore_size_nm,
                "flux_lmh": flux,
                "modulus_GPa": mem.modulus_GPa,
                "tensile_strength_MPa": mem.tensile_strength_MPa,
                "contact_angle_deg": mem.contact_angle_deg,
                "rejection_percent": mem.rejection_percent
            })

    # Summarize rejection using new stochastic model
    rejection_rates = []
    for m in membranes:
        rejection = simulate_oil_rejection_stochastic(
            pore_size_nm=m.pore_size_nm,
            droplet_size_um=5.0,  # Use literature mean or sample from distribution
            contact_angle_deg=m.contact_angle_deg,
            membrane_type=m.name  # Pass membrane type for literature-based range
        )
        rejection_rates.append(rejection)
        m.rejection_percent = rejection

    # Print Summary
    print("\nSimulation Summary:")
    for m, r in zip(membranes, rejection_rates):
        print(f"{m.name}: Flux ≈ {m.flux_lmh:.1f} L·m⁻²·h⁻¹, Rejection ≈ {r:.1f} %")

    # Export results to Excel
    try:
        df = pd.DataFrame(results)
        excel_path = os.path.join(phase1_dir, "simulation_results_phase1.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"\nResults table saved to: {excel_path}")
    except PermissionError:
        print(f"\n[ERROR] Could not write to {excel_path}. Please close the file if it is open and re-run the script.")

    # Prepare membrane types
    membrane_types = ['GO', 'rGO', 'hybrid']
    # Oil rejection summary
    rejections = [(MEMBRANE_TYPES[m].get('rejection', 0) or 0) for m in membrane_types]
    plot_rejection_summary(membrane_types, rejections, summary_dir)    # Flux vs thickness per pressure
    for pressure in PRESSURE_RANGE:
        thicknesses = MEMBRANE_TYPES['GO']['thicknesses']
        fluxes_dict = {}
        for m in ['GO', 'rGO']:
            fluxes_dict[m] = [simulate_flux_with_variability(
                {'pore_size_nm': MEMBRANE_TYPES[m]['pore_sizes'][0],
                 'thickness_nm': t,
                 'contact_angle_deg': MEMBRANE_TYPES[m]['contact_angle_deg']},
                pressure_bar=pressure,
                viscosity_pas=WATER_PROPERTIES["viscosity_25C"],
                porosity=WATER_PROPERTIES["porosity"],
                tortuosity=WATER_PROPERTIES["tortuosity"]
            ) for t in thicknesses]
        fluxes_dict['hybrid'] = [simulate_flux_with_variability(
            {'pore_size_nm': MEMBRANE_TYPES['hybrid']['pore_sizes'][0],
             'thickness_nm': t,
             'contact_angle_deg': MEMBRANE_TYPES['hybrid']['contact_angle_deg']},
            pressure_bar=pressure,
            viscosity_pas=WATER_PROPERTIES["viscosity_25C"],
            porosity=WATER_PROPERTIES["porosity"],
            tortuosity=WATER_PROPERTIES["tortuosity"]
        ) for t in thicknesses]
        plot_flux_vs_thickness_at_pressure(thicknesses, fluxes_dict, pressure, thickness_dir)    # Flux vs pore size per pressure
    for pressure in PRESSURE_RANGE:
        pore_sizes = MEMBRANE_TYPES['GO']['pore_sizes']
        fluxes_dict = {}
        for m in ['GO', 'rGO']:
            fluxes_dict[m] = [simulate_flux_with_variability(
                {'pore_size_nm': MEMBRANE_TYPES[m]['pore_sizes'][0],
                 'thickness_nm': t,
                 'contact_angle_deg': MEMBRANE_TYPES[m]['contact_angle_deg']},
                pressure_bar=pressure,
                viscosity_pas=WATER_PROPERTIES["viscosity_25C"],
                porosity=WATER_PROPERTIES["porosity"],
                tortuosity=WATER_PROPERTIES["tortuosity"]
            ) for t in thicknesses]
        fluxes_dict['hybrid'] = [simulate_flux_with_variability(
            {'pore_size_nm': MEMBRANE_TYPES['hybrid']['pore_sizes'][0],
             'thickness_nm': t,
             'contact_angle_deg': MEMBRANE_TYPES['hybrid']['contact_angle_deg']},
            pressure_bar=pressure,
            viscosity_pas=WATER_PROPERTIES["viscosity_25C"],
            porosity=WATER_PROPERTIES["porosity"],
            tortuosity=WATER_PROPERTIES["tortuosity"]
        ) for t in thicknesses]
        plot_flux_vs_pore_size_at_pressure(pore_sizes, fluxes_dict, pressure, pore_dir)
    
    # Phase 2: Hybrid Structure Design (now always runs)
    print("\nRunning Phase 2 (Hybrid Structure Design) automatically.")
    run_phase2 = True
    target_flux = None
    target_rejection = None
    # Run Phase 2 analysis
    phase2_results = run_phase2_analysis(target_flux, target_rejection)
    # Add Phase 2 results to main results for potential Phase 3 use
    results.extend([{
        "material": "Hybrid_Phase2",
        "membrane_name": r['structure_name'],
        "pressure_bar": 1.0,  # Standard pressure
        "thickness_nm": r['thickness_nm'],
        "pore_size_nm": r['avg_pore_size'],
        "flux_lmh": r['predicted_flux'],
        "modulus_GPa": r['weighted_modulus'],
        "tensile_strength_MPa": r['weighted_strength'],
        "contact_angle_deg": None,
        "rejection_percent": r['predicted_rejection'],
        "performance_score": r['performance_score'],
        "go_fraction": r['go_fraction'],
        "total_layers": r['total_layers']        } for r in phase2_results['top_structures']])
    print(f"\nPhase 2 structures added to results. Total entries: {len(results)}")
    # Save Phase 2 results
    try:
        df2 = pd.DataFrame(phase2_results['top_structures'])
        excel_path2 = os.path.join(phase2_dir, "simulation_results_phase2.xlsx")
        df2.to_excel(excel_path2, index=False)
        print(f"\nPhase 2 results table saved to: {excel_path2}")
    except Exception:
        pass
    # Phase 3: Atomistic LAMMPS Simulations (now always runs for top hybrid structure)
    print("\nRunning Phase 3 (LAMMPS Atomistic Simulations) automatically for top hybrid structure.")
    from lammps_runner import LAMMPSRunner
    from unify_results import unify_all_results
    phase3_membrane = None
    if run_phase2 and phase2_results and phase2_results['top_structures']:
        top_structure = phase2_results['top_structures'][0]
        print(f"Selected for LAMMPS: {top_structure['structure_name']} (Phase 2)")
        phase3_membrane = Membrane(
            name=top_structure['structure_name'],
            pore_size_nm=top_structure['avg_pore_size'],
            thickness_nm=top_structure['thickness_nm'],
            flux_lmh=top_structure['predicted_flux'],
            modulus_GPa=top_structure['weighted_modulus'],
            tensile_strength_MPa=top_structure['weighted_strength'],
            contact_angle_deg=90.0,
            rejection_percent=top_structure['predicted_rejection']
        )
    if phase3_membrane:
        lammps_runner = LAMMPSRunner(working_dir=phase3_dir)
        lammps_results = lammps_runner.run_membrane_simulation(phase3_membrane)
        if lammps_results and lammps_results.get('success', False):
            print("LAMMPS Simulation Results:")
            print(f"  Water flux: {lammps_results.get('water_flux', 'N/A')} L·m⁻²·h⁻¹")
            print(f"  Young's modulus: {lammps_results.get('youngs_modulus', 'N/A')} GPa")
            print(f"  Ultimate strength: {lammps_results.get('ultimate_strength', 'N/A')} MPa")
        else:
            print("❌ LAMMPS simulation failed or returned no results. Stopping workflow.")
            print(f"Reason: {lammps_results.get('error', 'Unknown error') if lammps_results else 'Unknown error'}")
            sys.exit(1)
        # Unify and export all results
        print("Unifying all simulation results...")
        phase1_path = os.path.join(phase1_dir, "simulation_results_phase1.xlsx")
        unify_all_results(phase1_path=phase1_path, output_dir=phase3_dir)
    else:
        print("No valid hybrid structure found for LAMMPS simulation. Skipping Phase 3.")
        sys.exit(1)

    # Phase 4: Chemical and Biological Simulation (now always runs with default settings)
    print("\nRunning Phase 4 (Chemical & Biological Simulation) automatically.")
    from simulate_chemistry import run_phase4_simulation
    from plot_chemistry import plot_phase4_results
    print("\n🧪 PHASE 4: CHEMICAL AND BIOLOGICAL SIMULATION")
    print("==================================================")
    # Use default: all contaminants, default concentrations, 180 min
    contaminants = ['Pb2+', 'E_coli', 'NaCl', 'BPA', 'NO3', 'Microplastics']
    concentrations = {
        'Pb2+': 50.0, 'E_coli': 1e5, 'NaCl': 1000.0, 
        'BPA': 15.0, 'NO3': 45.0, 'Microplastics': 100.0
    }
    reaction_time = 180
    print(f"\nRunning Phase 4 simulation...")
    print(f"Contaminants: {contaminants}")
    print(f"Concentrations: {concentrations}")
    print(f"Reaction time: {reaction_time} minutes")
    try:
        membrane_types_phase4 = ['GO', 'rGO', 'hybrid']
        if run_phase2 and phase2_results:
            print("Including Phase 2 hybrid structures in chemical simulation...")
            top_structure = phase2_results['top_structures'][0]
            membrane_types_phase4.append(f"Phase2_{top_structure['structure_name']}")
        chemistry_engine = run_phase4_simulation(
            membrane_types=membrane_types_phase4[:3],
            contaminants=contaminants,
            initial_concentrations=concentrations,
            reaction_time=reaction_time
        )
        if chemistry_engine and chemistry_engine.simulation_results:
            print(f"\n✅ Phase 4 simulation completed successfully!")
            print("Generating Phase 4 visualization report...")
            plot_figures = plot_phase4_results(chemistry_engine, save_plots=True, output_dir=phase4_dir)
            print(f"Generated {len(plot_figures)} chemical simulation plots")
            for sim_result in chemistry_engine.simulation_results:
                membrane_type = sim_result['membrane_type']
                for contaminant, data in sim_result['contaminants'].items():
                    efficiency = 0
                    if 'removal_efficiency' in data:
                        efficiency = data['removal_efficiency']
                    elif 'kill_efficiency' in data:
                        efficiency = data['kill_efficiency']
                    elif 'rejection_percent' in data:
                        efficiency = max(0, data['rejection_percent'])
                    results.append({
                        "material": f"{membrane_type}_Phase4",
                        "membrane_name": f"{membrane_type}_{contaminant}",
                        "pressure_bar": 1.0,
                        "thickness_nm": MEMBRANE_TYPES.get(membrane_type, {}).get('thickness_nm', 100),
                        "pore_size_nm": MEMBRANE_TYPES.get(membrane_type, {}).get('pore_size_nm', 2.0),
                        "flux_lmh": MEMBRANE_TYPES.get(membrane_type, {}).get('flux', 100),
                        "modulus_GPa": MEMBRANE_TYPES.get(membrane_type, {}).get('modulus', 200),
                        "tensile_strength_MPa": MEMBRANE_TYPES.get(membrane_type, {}).get('strength', 30),
                        "contact_angle_deg": MEMBRANE_TYPES.get(membrane_type, {}).get('contact_angle_deg', 90),
                        "rejection_percent": efficiency,
                        "contaminant_type": contaminant,
                        "removal_mechanism": data.get('interaction_mechanisms', data.get('mechanisms', [])),
                        "simulation_type": "Chemical_Biological",
                        "phase4_data": data
                    })
            print(f"Phase 4 chemical data integrated into main results.")
        else:
            print("❌ Phase 4 simulation failed or returned no results.")
    except Exception as e:
        print(f"❌ Error in Phase 4 simulation: {e}")
        print("Phase 4 simulation failed, but previous phase results are still available.")
    
    # Final export with all phases
    try:
        df_final = pd.DataFrame(results)
        excel_path = os.path.join(output_base, "simulation_results_all_phases.xlsx")
        df_final.to_excel(excel_path, index=False)
        print(f"\nFinal results (all phases) saved to: {excel_path}")
        print(f"Total simulation entries: {len(results)}")
    except PermissionError:
        print(f"\n[ERROR] Could not write to {excel_path}. Please close the file if it is open.")

if __name__ == "__main__":
    main()
