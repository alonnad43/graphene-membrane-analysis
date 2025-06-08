# plot_utils.py

"""
Utility functions for plotting flux, rejection, and other membrane simulation results.

Includes functions for plotting water flux vs. thickness and oil rejection efficiency.
"""

import matplotlib.pyplot as plt
import os

def plot_rejection_summary(membrane_types, rejections, output_dir):
    """
    Plot a single summary bar chart of oil rejection for all membrane types.
    Saves to graphs/oil_rejection_summary/.

    Args:
        membrane_types (list): List of membrane type names
        rejections (list): List of rejection efficiencies (%)
        output_dir (str): Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.bar(membrane_types, rejections, color=['#1f77b4', '#2ca02c', '#ff7f0e'])
    plt.ylabel('Oil Rejection (%)')
    plt.title('Oil Rejection Efficiency by Membrane Type')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'oil_rejection_summary.png'))
    plt.close()

def plot_flux_vs_thickness(thicknesses, fluxes_dict):
    """
    Plot water flux vs. membrane thickness for multiple membrane types.

    Args:
        thicknesses (array-like): List of thickness values (nm)
        fluxes_dict (dict): Dictionary with membrane names as keys and flux lists as values
    """
    for membrane_name, flux_values in fluxes_dict.items():
        plt.plot(thicknesses, flux_values, label=membrane_name)
    plt.xlabel('Thickness (nm)')
    plt.ylabel('Water Flux (L·m⁻²·h⁻¹)')
    plt.title('Water Flux vs. Membrane Thickness')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_flux_vs_thickness_at_pressure(thicknesses, fluxes_dict, pressure, output_dir):
    """
    Plot flux vs. thickness for all membrane types at a given pressure.
    Saves to graphs/flux_vs_thickness_per_pressure/.

    Args:
        thicknesses (array-like): List of thickness values (nm)
        fluxes_dict (dict): Dictionary with membrane names as keys and flux lists as values
        pressure (float): The pressure at which flux is measured (bar)
        output_dir (str): Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    for membrane_name, flux_values in fluxes_dict.items():
        plt.plot(thicknesses, flux_values, marker='o', label=membrane_name)
    plt.xlabel('Thickness (nm)')
    plt.ylabel('Water Flux (L·m⁻²·h⁻¹)')
    plt.title(f'Flux vs. Thickness at {pressure} bar')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname = f'flux_vs_thickness_P{pressure}.png'.replace('.', '_')
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

def plot_oil_rejection(membrane_types, rejections, output_dir=None):
    """
    Plot oil rejection efficiency for different membrane types and save the plot if output_dir is provided.

    Args:
        membrane_types (list): List of membrane type names
        rejections (list): List of rejection efficiencies (%)
        output_dir (str, optional): Directory to save the plot. If None, just show the plot.
    """
    plt.figure()
    plt.bar(membrane_types, rejections, color=['#1f77b4', '#2ca02c', '#ff7f0e'])
    plt.ylabel('Oil Rejection (%)')
    plt.title('Oil Rejection Efficiency by Membrane Type')
    plt.ylim(0, 100)
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_name = '_'.join(membrane_types).replace(' ', '_').replace('.', '_').replace('-', '_')
        filename = f"{safe_name}_oil_rejection.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()

def plot_flux_vs_pressure(pressures, fluxes, membrane_name, output_dir=None):
    """
    Plot water flux vs. applied pressure for a single membrane variant and save the plot if output_dir is provided.

    Args:
        pressures (array-like): List of pressure values (bar)
        fluxes (array-like): List of flux values (L·m⁻²·h⁻¹)
        membrane_name (str): Name of the membrane variant
        output_dir (str, optional): Directory to save the plot. If None, just show the plot.
    """
    plt.figure()
    plt.plot(pressures, fluxes, marker='o', label=membrane_name)
    plt.xlabel('Pressure (bar)')
    plt.ylabel('Water Flux (L·m⁻²·h⁻¹)')
    plt.title(f'Water Flux vs. Pressure: {membrane_name}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Clean filename: replace spaces and special chars
        safe_name = membrane_name.replace(' ', '_').replace('.', '_').replace('-', '_')
        filename = f"{safe_name}_flux_vs_pressure.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()

def plot_flux_vs_pore_size_at_pressure(pore_sizes, fluxes_dict, pressure, output_dir):
    """
    Plot flux vs. pore size for all membrane types at a given pressure.
    Saves to graphs/flux_vs_pore_size_per_pressure/.

    Args:
        pore_sizes (array-like): List of pore size values (nm)
        fluxes_dict (dict): Dictionary with membrane names as keys and flux lists as values
        pressure (float): The pressure at which flux is measured (bar)
        output_dir (str): Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    for membrane_name, flux_values in fluxes_dict.items():
        plt.plot(pore_sizes, flux_values, marker='o', label=membrane_name)
    plt.xlabel('Pore Size (nm)')
    plt.ylabel('Water Flux (L·m⁻²·h⁻¹)')
    plt.title(f'Flux vs. Pore Size at {pressure} bar')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname = f'flux_vs_pore_size_P{pressure}.png'.replace('.', '_')
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()
