# flux_simulator.py

"""
Phase 1: Calculates water flux across GO, rGO, or hybrid membranes using physical models.

Scientific approach: Uses Hagen-Poiseuille equation for nanopore flow.
"""

import numpy as np
import matplotlib.pyplot as plt
from properties import MEMBRANE_TYPES, PRESSURE_RANGE, WATER_VISCOSITY

def simulate_flux(pore_size_nm, thickness_nm, pressure_bar, viscosity_mpas=None):
    """
    Calculate water flux using physical model (Hagen–Poiseuille).
    
    Args:
        pore_size_nm (float): Pore size in nanometers
        thickness_nm (float): Membrane thickness in nanometers
        pressure_bar (float): Applied pressure in bar
        viscosity_mpas (float, optional): Dynamic viscosity in mPa·s (defaults to water)
    
    Returns:
        float: Water flux in L·m⁻²·h⁻¹
    
    Scientific basis:
        - Hagen–Poiseuille equation for flow through nanopores
        - Permeability = (pore_radius²) / (8 * viscosity)
        - Physical units properly converted throughout
    """
    # Use default water viscosity if not provided
    if viscosity_mpas is None:
        viscosity_mpas = WATER_VISCOSITY
    
    # Unit conversions
    pore_radius_m = (pore_size_nm * 1e-9) / 2  # Convert nm to m, diameter to radius
    thickness_m = thickness_nm * 1e-9          # Convert nm to m
    pressure_pa = pressure_bar * 1e5           # Convert bar to Pa
    viscosity_pa_s = viscosity_mpas * 1e-3     # Convert mPa·s to Pa·s

    # Simplified Hagen–Poiseuille equation for flow through nanopores
    permeability = (pore_radius_m**2) / (8 * viscosity_pa_s)
    flux_m3_per_m2_s = permeability * pressure_pa / thickness_m

    # Convert to L/m²/h
    flux_lmh = flux_m3_per_m2_s * 3600 * 1000
    return flux_lmh

# Backward compatibility function
def simulate_flux_old(thickness_nm, pore_size_nm, pressure_bar=1.0):
    """
    Legacy function for backward compatibility.
    Calls the new physics-based simulate_flux function.
    """
    return simulate_flux(pore_size_nm, thickness_nm, pressure_bar)

def plot_flux_vs_thickness():
    """
    Plot water flux vs. membrane thickness for GO and rGO membranes.
    """
    thicknesses = np.linspace(60, 150, 10)
    for membrane_name in ["GO", "rGO"]:
        pore = MEMBRANE_TYPES[membrane_name]["pore_sizes"][-1]
        fluxes = [simulate_flux(pore, t, 1.0) for t in thicknesses]  # Updated parameter order
        plt.plot(thicknesses, fluxes, label=f"{membrane_name} (pore={pore}nm)")
    plt.xlabel("Thickness (nm)")
    plt.ylabel("Flux (L·m⁻²·h⁻¹)")
    plt.title("Water Flux vs. Thickness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
