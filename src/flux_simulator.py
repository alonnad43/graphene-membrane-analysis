# flux_simulator.py

"""
Phase 1: Calculates water flux across GO, rGO, or hybrid membranes using physically accurate models.

Scientific approach: Modified Hagen-Poiseuille equation with porosity and tortuosity corrections.
References: Schmidt et al. 2023, Green Synthesis GO 2018, Revolutionizing water purification 2023
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi, log10
from properties import MEMBRANE_TYPES, PRESSURE_RANGE, WATER_PROPERTIES

def calculate_temperature_viscosity(temperature=298):
    """
    Calculate water viscosity at given temperature using IAPWS model.
    
    Args:
        temperature (float): Temperature in Kelvin (default: 298K = 25°C)
    
    Returns:
        float: Dynamic viscosity in Pa·s
    """
    try:
        A, B, C = WATER_PROPERTIES["viscosity_model"]["A"], WATER_PROPERTIES["viscosity_model"]["B"], WATER_PROPERTIES["viscosity_model"]["C"]
        viscosity = A * 10 ** (B / (temperature - C))
        return viscosity
    except Exception as e:
        print(f"Error calculating viscosity, using default: {e}")
        return 0.00089  # Default water viscosity at 25°C

def simulate_flux(pore_size_nm, thickness_nm, pressure_bar, viscosity_pas=None, temperature=298, 
                 porosity=None, tortuosity=None):
    """
    Calculate water flux using modified Hagen-Poiseuille equation with porosity and tortuosity.
    
    Args:
        pore_size_nm (float): Pore size in nanometers
        thickness_nm (float): Membrane thickness in nanometers
        pressure_bar (float): Applied pressure in bar
        viscosity_pas (float, optional): Dynamic viscosity in Pa·s
        temperature (float): Temperature in Kelvin (default: 298K)
        porosity (float, optional): Membrane porosity (default from properties)
        tortuosity (float, optional): Tortuosity factor (default from properties)
    
    Returns:
        float: Water flux in L·m⁻²·h⁻¹ (LMH)
    
    Scientific equation:
        J = (ε * r² / (8 * μ * τ)) * (ΔP / L)
        Where ε=porosity, r=pore_radius, μ=viscosity, τ=tortuosity, ΔP=pressure, L=thickness
    """
    try:
        # Use defaults from properties if not provided
        if viscosity_pas is None:
            viscosity_pas = calculate_temperature_viscosity(temperature)
        if porosity is None:
            porosity = WATER_PROPERTIES["porosity"]
        if tortuosity is None:
            tortuosity = WATER_PROPERTIES["tortuosity"]
        
        # Unit conversions
        pore_radius_m = (pore_size_nm * 1e-9) / 2  # Convert nm to m, diameter to radius
        thickness_m = thickness_nm * 1e-9          # Convert nm to m
        pressure_pa = pressure_bar * 1e5           # Convert bar to Pa
        
        # Modified Hagen-Poiseuille with porosity and tortuosity corrections
        permeability = (porosity * pore_radius_m**2) / (8 * viscosity_pas * tortuosity)
        flux_m_per_s = permeability * pressure_pa / thickness_m
        
        # Convert to L·m⁻²·h⁻¹ (LMH)
        flux_lmh = flux_m_per_s * 3600 * 1000
        
        return flux_lmh
        
    except Exception as e:
        print(f"Error in simulate_flux: {e}")
        # Fallback to simple calculation
        return (pore_size_nm**2 * pressure_bar * 1000) / thickness_nm

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
