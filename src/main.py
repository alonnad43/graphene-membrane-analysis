# main.py
"""
Main entry point for the GO/rGO membrane simulation project – all phases combined.

This script orchestrates the complete simulation workflow:
1. Phase 1: Runs macroscale simulations of GO, rGO, and hybrid membranes.
2. Phase 2: Constructs realistic hybrid GO/rGO membrane structures.
3. Phase 3: Runs atomistic simulations using LAMMPS.

# TODO: Connect each membrane object across all phases
# TODO: Output membrane_summary.csv with macro + nano data
"""

from membrane_model import Membrane
from flux_simulator import simulate_flux
from oil_rejection import simulate_oil_rejection
from plot_utils import (
    plot_rejection_summary,
    plot_flux_vs_thickness_at_pressure,
    plot_flux_vs_pore_size_at_pressure
)
from properties import MEMBRANE_TYPES, PRESSURE_RANGE
from hybrid_structure import (
    run_phase2_analysis,
    create_alternating_structure, 
    create_sandwich_structure, 
    optimize_structure_for_flux
)
import numpy as np
import os
import pandas as pd

def generate_membrane_variants():
    """
    Generate all membrane variants (GO and rGO) across defined thickness and pore size combinations.
    Returns a list of Membrane objects.
    """
    membranes = []
    
    for mem_type in ['GO', 'rGO']:
        props = MEMBRANE_TYPES[mem_type]
        for thickness in props['thicknesses']:
            for pore_size in props['pore_sizes']:
                flux = props['flux_map'][thickness]
                rejection = props['rejection_map'][pore_size]
                membranes.append(Membrane(
                    name=f"{mem_type} T{thickness} P{pore_size}",
                    pore_size_nm=pore_size,
                    thickness_nm=thickness,
                    flux_lmh=flux,
                    modulus_GPa=props['modulus'],
                    tensile_strength_MPa=props['strength'],
                    contact_angle_deg=props['contact_angle_deg'],
                    rejection_percent=rejection
                ))
    
    # Add Hybrid
    hybrid_props = MEMBRANE_TYPES["Hybrid"]
    membranes.append(Membrane(
        name="Hybrid",
        pore_size_nm=hybrid_props['pore_size'],
        thickness_nm=hybrid_props['thickness'],
        flux_lmh=hybrid_props['flux'],        modulus_GPa=hybrid_props['modulus'],
        tensile_strength_MPa=hybrid_props['strength'],
        contact_angle_deg=hybrid_props['contact_angle_deg'],
        rejection_percent=hybrid_props['rejection']
    ))

    return membranes

def main():
    membranes = generate_membrane_variants()
    output_base = r"C:\Users\ramaa\Documents\graphene_mebraine\output"
    results = []  # For Excel export    # Prepare output directories
    graphs_base = os.path.join(os.getcwd(), 'graphs')
    summary_dir = os.path.join(graphs_base, 'oil_rejection_summary')
    thickness_dir = os.path.join(graphs_base, 'flux_vs_thickness_per_pressure')
    pore_dir = os.path.join(graphs_base, 'flux_vs_pore_size_per_pressure')

    # Simulate flux and rejection at different pressures
    for mem in membranes:
        # Determine material folder (GO, rGO, Hybrid)
        mat_folder = mem.name.split()[0] if mem.name != "Hybrid" else "Hybrid"
        mat_dir = os.path.join(output_base, mat_folder)
        # Save flux vs. pressure plot in the appropriate folder
        flux_dir = os.path.join(mat_dir, "flux_vs_pressure")
        
        # Use new physics-based flux simulation
        fluxes = [simulate_flux(mem.pore_size_nm, mem.thickness_nm, p, mem.water_viscosity) for p in PRESSURE_RANGE]
        
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
                "rejection_percent": mem.rejection_percent            })
        # Removed plot_flux_vs_pressure call as per the new workflow

    # Summarize rejection using new physics-based model
    rejection_rates = []
    for m in membranes:
        # Use new physics-based oil rejection calculation
        rejection = simulate_oil_rejection(
            pore_size_nm=m.pore_size_nm,
            droplet_size_um=m.oil_droplet_size,
            contact_angle_deg=m.contact_angle_deg
        )
        rejection_rates.append(rejection)
    # Removed plot_oil_rejection call as per the new workflow

    # Print Summary
    print("\nSimulation Summary:")
    for m, r in zip(membranes, rejection_rates):
        print(f"{m.name}: Flux ≈ {m.flux_lmh} L·m⁻²·h⁻¹, Rejection ≈ {r} %")

    # Export results to Excel
    # Ensure the file is not open before writing
    try:
        df = pd.DataFrame(results)
        excel_path = os.path.join(output_base, "simulation_results.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"\nResults table saved to: {excel_path}")
    except PermissionError:
        print(f"\n[ERROR] Could not write to {excel_path}. Please close the file if it is open and re-run the script.")

    # Prepare membrane types
    membrane_types = ['GO', 'rGO', 'Hybrid']
    # Oil rejection summary
    rejections = [(MEMBRANE_TYPES[m].get('rejection', 0) or 0) for m in membrane_types]
    plot_rejection_summary(membrane_types, rejections, summary_dir)    # Flux vs thickness per pressure
    for pressure in PRESSURE_RANGE:
        thicknesses = MEMBRANE_TYPES['GO']['thicknesses']
        fluxes_dict = {}
        for m in ['GO', 'rGO']:
            # Use new parameter order: pore_size_nm, thickness_nm, pressure_bar, viscosity_mpas
            fluxes_dict[m] = [simulate_flux(MEMBRANE_TYPES[m]['pore_sizes'][0], t, pressure) for t in thicknesses]
        # Hybrid: use average pore size
        fluxes_dict['Hybrid'] = [simulate_flux(MEMBRANE_TYPES['Hybrid']['pore_size'], t, pressure) for t in thicknesses]
        plot_flux_vs_thickness_at_pressure(thicknesses, fluxes_dict, pressure, thickness_dir)    # Flux vs pore size per pressure
    for pressure in PRESSURE_RANGE:
        pore_sizes = MEMBRANE_TYPES['GO']['pore_sizes']
        fluxes_dict = {}
        for m in ['GO', 'rGO']:
            # Use new parameter order: pore_size_nm, thickness_nm, pressure_bar, viscosity_mpas
            fluxes_dict[m] = [simulate_flux(p, MEMBRANE_TYPES[m]['thicknesses'][0], pressure) for p in pore_sizes]
        # Hybrid: use average thickness
        fluxes_dict['Hybrid'] = [simulate_flux(p, MEMBRANE_TYPES['Hybrid']['thickness'], pressure) for p in pore_sizes]
        plot_flux_vs_pore_size_at_pressure(pore_sizes, fluxes_dict, pressure, pore_dir)
    
    # Phase 2: Hybrid Structure Design (optional)
    run_phase2 = input("\nRun Phase 2 (Hybrid Structure Design)? [yes/no]: ").lower().strip() in ['yes', 'y']
    
    if run_phase2:
        # Get user targets (optional)
        try:
            target_flux = float(input("Target flux (L·m⁻²·h⁻¹) [Enter for auto]: ") or "0")
            target_rejection = float(input("Target rejection (%) [Enter for auto]: ") or "0")
            
            target_flux = target_flux if target_flux > 0 else None
            target_rejection = target_rejection if target_rejection > 0 else None
            
        except ValueError:
            target_flux, target_rejection = None, None
        
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
    
    else:
        print("\nSkipping Phase 2 - Running Phase 1 only")
    
    # Phase 3: Atomistic LAMMPS Simulations (optional)
    run_phase3 = input("\nRun Phase 3 (LAMMPS Atomistic Simulations)? [yes/no]: ").lower().strip() in ['yes', 'y']
    
    if run_phase3:
        from lammps_runner import run_membrane_simulation
        from unify_results import unify_all_results
        
        print("\nAvailable membranes for atomistic simulation:")
        available_membranes = []
        
        # Show GO/rGO variants
        for i, mem in enumerate(membranes[:6]):  # First 6 are representative GO/rGO
            print(f"{i+1}. {mem.name}")
            available_membranes.append(mem)
        
        # Show hybrid options if Phase 2 was run
        if run_phase2 and phase2_results:
            for i, structure in enumerate(phase2_results['top_structures'][:3]):  # Top 3 Phase 2 structures
                idx = len(available_membranes) + 1
                print(f"{idx}. {structure['structure_name']} (Phase 2)")
                
                # Create membrane object from Phase 2 data
                hybrid_mem = Membrane(
                    name=structure['structure_name'],
                    pore_size_nm=structure['avg_pore_size'],
                    thickness_nm=structure['thickness_nm'],
                    flux_lmh=structure['predicted_flux'],
                    modulus_GPa=structure['weighted_modulus'],
                    tensile_strength_MPa=structure['weighted_strength'],
                    contact_angle_deg=90.0,  # Default
                    rejection_percent=structure['predicted_rejection']
                )
                available_membranes.append(hybrid_mem)
        
        # Get user selection
        try:
            selection = int(input(f"\nSelect membrane (1-{len(available_membranes)}): ")) - 1
            if 0 <= selection < len(available_membranes):
                selected_membrane = available_membranes[selection]
                print(f"\nRunning LAMMPS simulation for: {selected_membrane.name}")
                
                # Determine membrane type for LAMMPS
                if "GO" in selected_membrane.name and "rGO" not in selected_membrane.name:
                    lammps_type = "GO"
                elif "rGO" in selected_membrane.name:
                    lammps_type = "rGO"
                else:
                    lammps_type = "Hybrid"
                
                # Run LAMMPS simulation
                try:
                    lammps_results = run_membrane_simulation(
                        membrane_type=lammps_type,
                        membrane_object=selected_membrane,
                        simulation_steps=50000,  # Reasonable for testing
                        output_dir="lammps_sims"
                    )
                    
                    if lammps_results:
                        print(f"\nLAMMPS Simulation Results:")
                        print(f"Water flux: {lammps_results.get('water_flux', 'N/A')} L·m⁻²·h⁻¹")
                        print(f"Young's modulus: {lammps_results.get('youngs_modulus', 'N/A')} GPa")
                        print(f"Ultimate strength: {lammps_results.get('ultimate_strength', 'N/A')} MPa")
                        
                        # Add LAMMPS results to main dataset
                        results.append({
                            "material": f"{lammps_type}_LAMMPS",
                            "membrane_name": f"{selected_membrane.name}_LAMMPS",
                            "pressure_bar": 1.0,
                            "thickness_nm": selected_membrane.thickness_nm,
                            "pore_size_nm": selected_membrane.pore_size_nm,
                            "flux_lmh": lammps_results.get('water_flux', selected_membrane.flux_lmh),
                            "modulus_GPa": lammps_results.get('youngs_modulus', selected_membrane.modulus_GPa),
                            "tensile_strength_MPa": lammps_results.get('ultimate_strength', selected_membrane.tensile_strength_MPa),
                            "contact_angle_deg": lammps_results.get('contact_angle', selected_membrane.contact_angle_deg),
                            "rejection_percent": selected_membrane.rejection_percent,
                            "simulation_type": "LAMMPS_Atomistic"
                        })
                        
                        # Unify and export all results
                        print("\nUnifying all simulation results...")
                        unify_all_results()
                        
                    else:
                        print("\nLAMMPS simulation failed or returned no results.")
                        
                except Exception as e:
                    print(f"\nError running LAMMPS simulation: {e}")
                    print("Phase 3 simulation failed, but Phases 1-2 results are still available.")
            else:
                print("Invalid selection. Skipping Phase 3.")
                
        except (ValueError, KeyError):
            print("Invalid input. Skipping Phase 3.")
    
    else:
        print("\nSkipping Phase 3 - LAMMPS simulations not run")
    
    # Final export with all phases
    try:
        df_final = pd.DataFrame(results)
        excel_path = os.path.join(output_base, "simulation_results.xlsx")
        df_final.to_excel(excel_path, index=False)
        print(f"\nFinal results (all phases) saved to: {excel_path}")
        print(f"Total simulation entries: {len(results)}")
    except PermissionError:
        print(f"\n[ERROR] Could not write to {excel_path}. Please close the file if it is open.")

if __name__ == "__main__":
    main()
