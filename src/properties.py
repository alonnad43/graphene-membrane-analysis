"""
Material constants and simulation parameters for GO/rGO membranes.

This module stores all physical constants, empirical values, and simulation ranges used throughout the project.
"""

# Pore Sizes (nm)
PORE_SIZE_GO = 1.2
PORE_SIZE_RGO = 0.99
PORE_SIZE_RGOTH = 0.85

# Thicknesses (nm)
THICKNESS_GO = 100
THICKNESS_RGOTA = 150
THICKNESS_RGOTH = 60

# Water Flux (L·m⁻²·h⁻¹·bar⁻¹)
FLUX_GO = 35
FLUX_RGOTA = 10191
FLUX_RGOTH = 10720

# Oil Rejection Efficiency (%)
REJECTION_GO = 90
REJECTION_RGO = 90
REJECTION_HYBRID = 99

# Mechanical Properties
YOUNG_MODULUS_GO = 207.6  # GPa
TENSILE_STRENGTH_GO = 84.5  # MPa
YOUNG_MODULUS_RGO = 1.84  # GPa
TENSILE_STRENGTH_RGO = 150  # MPa
FRACTURE_STRAIN_GO = 0.02  # %
FRACTURE_STRAIN_RGO = 0.03  # %

# Contact Angle (Wettability)
CONTACT_ANGLE_GO = 29.5  # degrees
CONTACT_ANGLE_RGO = 73.9  # degrees
CONTACT_ANGLE_HYBRID = (CONTACT_ANGLE_GO + CONTACT_ANGLE_RGO) / 2  # Estimated


# Surface Energy (mJ/m²)
SURFACE_ENERGY_GO = 62.1
SURFACE_ENERGY_RGO = 110

# Densities (g/cm³)
DENSITY_GO = 1.8
DENSITY_RGO = 0.30

# Fluid Properties
WATER_VISCOSITY = 0.89      # mPa·s (at 25°C)
OIL_VISCOSITY = 25.0        # mPa·s
WATER_OIL_SURFACE_TENSION = 25.0  # mN/m

# Oil Properties
OIL_DROPLET_SIZE = 5.0  # µm (typical oil droplet size in water)
OIL_DENSITY = 0.85  # g/cm³

# Simulation Parameters
PRESSURE_MIN = 0.1  # bar
PRESSURE_MAX = 1.0  # bar
PRESSURE_RANGE = [round(0.1 * i, 2) for i in range(1, 11)]
FLUX_DECAY_EXPONENT = 1.5
PRESSURE_FLUX_LINEARITY = True

# Simulation Ranges
THICKNESS_RANGE = [60, 100, 150]  # nm
PORE_SIZE_RANGE = [0.3, 0.7, 1.2]  # nm

# Hybrid Membrane Note
HYBRID_STRUCTURE_NOTE = "Layered GO/rGO alternating structure with average thickness ~110 nm"

# Membrane Types Dictionary
MEMBRANE_TYPES = {
    "GO": {
        "pore_sizes": [0.3, 0.7, 1.2],
        "thicknesses": [60, 100, 150],
        "flux_map": {
            60: 70,
            100: 35,
            150: 20,
        },
        "rejection_map": {
            0.3: 99,
            0.7: 95,
            1.2: 90,
        },
        "modulus": YOUNG_MODULUS_GO,
        "strength": TENSILE_STRENGTH_GO,
        "contact_angle_deg": CONTACT_ANGLE_GO
    },
    "rGO": {
        "pore_sizes": [0.6, 0.85, 1.0],
        "thicknesses": [60, 100, 150],
        "flux_map": {
            60: 10720,
            100: 7000,
            150: 5000,
        },
        "rejection_map": {
            0.6: 98,
            0.85: 92,
            1.0: 85,
        },
        "modulus": YOUNG_MODULUS_RGO,
        "strength": TENSILE_STRENGTH_RGO,
        "contact_angle_deg": CONTACT_ANGLE_RGO
    },
    "Hybrid": {
        "pore_size": (PORE_SIZE_GO + PORE_SIZE_RGO) / 2,
        "thickness": 110,
        "flux": (FLUX_GO + FLUX_RGOTA) / 2,
        "rejection": REJECTION_HYBRID,
        "modulus": (YOUNG_MODULUS_GO + YOUNG_MODULUS_RGO) / 2,
        "strength": (TENSILE_STRENGTH_GO + TENSILE_STRENGTH_RGO) / 2,
        "contact_angle": (CONTACT_ANGLE_GO + CONTACT_ANGLE_RGO) / 2,
        "contact_angle_deg": CONTACT_ANGLE_HYBRID
    }
}

# LAMMPS Simulation Parameters for Phase 3
LAMMPS_PARAMETERS = {
    # Atomic masses (amu)
    "carbon_mass": 12.011,
    "oxygen_mass": 15.999,
    "hydrogen_mass": 1.008,
    
    # Bond lengths (Angstrom)
    "cc_bond_length": 1.42,     # C-C bond in graphene
    "co_bond_length": 1.43,     # C-O bond in GO
    "oh_bond_length": 0.96,     # O-H bond in GO
    
    # LJ parameters for carbon-carbon interactions
    "lj_epsilon_cc": 0.0703,    # kcal/mol
    "lj_sigma_cc": 3.4,         # Angstrom
    
    # LJ parameters for oxygen-oxygen interactions  
    "lj_epsilon_oo": 0.2104,    # kcal/mol
    "lj_sigma_oo": 3.12,        # Angstrom
    
    # Simulation settings
    "simulation_temperature": 300.0,  # Kelvin
    "timestep": 1.0,                  # femtoseconds
    "equilibration_steps": 50000,
    "production_steps": 100000,
    "dump_frequency": 1000,
    "thermo_frequency": 1000,
    
    # Water model parameters (TIP3P)
    "water_ow_mass": 15.9994,
    "water_hw_mass": 1.008,
    "water_ow_charge": -0.834,
    "water_hw_charge": 0.417,
    "water_density": 1.0           # g/cm³
}
