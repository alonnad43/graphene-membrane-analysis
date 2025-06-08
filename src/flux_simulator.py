# flux_simulator.py

import numpy as np
import matplotlib.pyplot as plt
from properties import MEMBRANE_TYPES, PRESSURE_RANGE

def simulate_flux(thickness_nm, pore_size_nm, pressure_bar=1.0):
    """
    Simulate water flux using a Darcy's Law variant.
    Args:
        thickness_nm (float): Membrane thickness in nanometers
        pore_size_nm (float): Pore size in nanometers
        pressure_bar (float): Applied pressure in bar
    Returns:
        float: Simulated water flux in L·m⁻²·h⁻¹
    """
    permeability = 10000 * (pore_size_nm / 1.0)  # arbitrary scaling
    flux = (permeability * pressure_bar) / thickness_nm
    return flux

def plot_flux_vs_thickness():
    """
    Plot water flux vs. membrane thickness for GO and rGO membranes.
    """
    thicknesses = np.linspace(60, 150, 10)
    for membrane_name in ["GO", "rGO"]:
        pore = MEMBRANE_TYPES[membrane_name]["pore_sizes"][-1]
        fluxes = [simulate_flux(t, pore) for t in thicknesses]
        plt.plot(thicknesses, fluxes, label=f"{membrane_name} (pore={pore}nm)")
    plt.xlabel("Thickness (nm)")
    plt.ylabel("Flux (L·m⁻²·h⁻¹)")
    plt.title("Water Flux vs. Thickness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
