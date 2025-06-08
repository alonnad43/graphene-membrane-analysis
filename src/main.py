# main.py
"""
Entry point for GO/rGO membrane simulation.

This script loads membrane data, runs water flux and oil rejection simulations,
and generates comparative plots for GO, rGO, and hybrid membranes.
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
                    contact_angle_deg=None,
                    rejection_percent=rejection
                ))
    
    # Add Hybrid
    hybrid_props = MEMBRANE_TYPES["Hybrid"]
    membranes.append(Membrane(
        name="Hybrid",
        pore_size_nm=hybrid_props['pore_size'],
        thickness_nm=hybrid_props['thickness'],
        flux_lmh=hybrid_props['flux'],
        modulus_GPa=None,
        tensile_strength_MPa=None,
        contact_angle_deg=None,
        rejection_percent=hybrid_props['rejection']
    ))

    return membranes

def main():
    membranes = generate_membrane_variants()
    output_base = r"C:\Users\ramaa\Documents\graphene_mebraine\output"
    results = []  # For Excel export

    # Prepare output directories
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
        fluxes = [simulate_flux(mem.thickness_nm, mem.pore_size_nm, p) for p in PRESSURE_RANGE]
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
        # Removed plot_flux_vs_pressure call as per the new workflow

    # Summarize rejection
    # Extract base type and pore size for rejection calculation
    rejection_rates = []
    for m in membranes:
        base_type = m.name.split()[0] if m.name != "Hybrid" else "Hybrid"
        pore_size = m.pore_size_nm if base_type in ["GO", "rGO"] else None
        rejection_rates.append(simulate_oil_rejection(base_type, pore_size))
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
    plot_rejection_summary(membrane_types, rejections, summary_dir)

    # Flux vs thickness per pressure
    for pressure in PRESSURE_RANGE:
        thicknesses = MEMBRANE_TYPES['GO']['thicknesses']
        fluxes_dict = {}
        for m in ['GO', 'rGO']:
            fluxes_dict[m] = [simulate_flux(t, MEMBRANE_TYPES[m]['pore_sizes'][0], pressure) for t in thicknesses]
        # Hybrid: use average pore size
        fluxes_dict['Hybrid'] = [simulate_flux(t, MEMBRANE_TYPES['Hybrid']['pore_size'], pressure) for t in thicknesses]
        plot_flux_vs_thickness_at_pressure(thicknesses, fluxes_dict, pressure, thickness_dir)

    # Flux vs pore size per pressure
    for pressure in PRESSURE_RANGE:
        pore_sizes = MEMBRANE_TYPES['GO']['pore_sizes']
        fluxes_dict = {}
        for m in ['GO', 'rGO']:
            fluxes_dict[m] = [simulate_flux(MEMBRANE_TYPES[m]['thicknesses'][0], p, pressure) for p in pore_sizes]
        # Hybrid: use average thickness
        fluxes_dict['Hybrid'] = [simulate_flux(MEMBRANE_TYPES['Hybrid']['thickness'], p, pressure) for p in pore_sizes]
        plot_flux_vs_pore_size_at_pressure(pore_sizes, fluxes_dict, pressure, pore_dir)

if __name__ == "__main__":
    main()
