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

# Oil Properties
OIL_DROPLET_SIZE = 5.0  # µm
OIL_DENSITY = 0.85  # g/cm³
OIL_VISCOSITY = 25.0  # mPa·s
WATER_OIL_SURFACE_TENSION = 25.0  # mN/m

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
