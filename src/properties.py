"""
Material constants and simulation parameters for GO/rGO membranes.

This module stores all physical constants, empirical values, and simulation ranges used throughout the project.
"""

# Thickness (nm)
THICKNESS_GO = 100
THICKNESS_RGO = 80
THICKNESS_HYBRID = 90

# Average Pore Size (nm)
PORE_SIZE_GO = 2.0
PORE_SIZE_RGO = 1.5
PORE_SIZE_HYBRID = 1.75

# Contact Angle (degrees)
CONTACT_ANGLE_GO = 29.5
CONTACT_ANGLE_RGO = 73.9
CONTACT_ANGLE_HYBRID = 51.7

# Water Flux at 1 bar (LMH)
FLUX_GO = 120
FLUX_RGO = 80
FLUX_HYBRID = 100

# Oil Rejection (%)
REJECTION_GO = 85
REJECTION_RGO = 93
REJECTION_HYBRID = 89

# Young’s Modulus (GPa)
YOUNG_MODULUS_GO = 207
YOUNG_MODULUS_RGO = 280
YOUNG_MODULUS_HYBRID = 243.5

# Mechanical Properties
TENSILE_STRENGTH_GO = 84.5  # MPa
TENSILE_STRENGTH_RGO = 150  # MPa
FRACTURE_STRAIN_GO = 0.02  # %
FRACTURE_STRAIN_RGO = 0.03  # %

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

# Advanced Water Properties (IAPWS model)
WATER_PROPERTIES = {
    "viscosity_25C": 0.00089,  # Pa·s at 25°C
    "viscosity_model": {"A": 2.414e-5, "B": 247.8, "C": 140},  # IAPWS viscosity coefficients
    "porosity": 0.35,          # Default membrane porosity
    "tortuosity": 2.0          # Default tortuosity factor
}

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

# Membrane Properties (Literature-derived)
MEMBRANE_PROPERTIES = {
    'GO': {
        'thickness_nm': 100,
        'pore_size_nm': 2.0,
        'contact_angle_deg': 29.5,
        'water_flux_LMH': 120,
        'oil_rejection_pct': 85,
        'youngs_modulus_GPa': 207
    },
    'rGO': {
        'thickness_nm': 80,
        'pore_size_nm': 1.5,
        'contact_angle_deg': 73.9,
        'water_flux_LMH': 80,
        'oil_rejection_pct': 93,
        'youngs_modulus_GPa': 280
    },
    'hybrid': {
        'thickness_nm': 90,
        'pore_size_nm': 1.75,
        'contact_angle_deg': 51.7,
        'water_flux_LMH': 100,
        'oil_rejection_pct': 89,
        'youngs_modulus_GPa': 243.5
    }
}

# Unified membrane type dictionary for all simulation modules
MEMBRANE_TYPES = {
    'GO': {
        'thickness_nm': THICKNESS_GO,
        'pore_size_nm': PORE_SIZE_GO,
        'contact_angle_deg': CONTACT_ANGLE_GO,
        'water_flux_LMH': FLUX_GO,
        'oil_rejection_pct': REJECTION_GO,
        'youngs_modulus_GPa': YOUNG_MODULUS_GO,
        'tensile_strength_MPa': TENSILE_STRENGTH_GO,  # Standardized for all modules
        'thicknesses': [60, 100, 150],
        'pore_sizes': [0.3, 0.7, 1.2, 2.0],
        'flux_map': {60: FLUX_GO, 100: FLUX_GO, 150: FLUX_GO}
    },
    'rGO': {
        'thickness_nm': THICKNESS_RGO,
        'pore_size_nm': PORE_SIZE_RGO,
        'contact_angle_deg': CONTACT_ANGLE_RGO,
        'water_flux_LMH': FLUX_RGO,
        'oil_rejection_pct': REJECTION_RGO,
        'youngs_modulus_GPa': YOUNG_MODULUS_RGO,
        'tensile_strength_MPa': TENSILE_STRENGTH_RGO,  # Standardized for all modules
        'thicknesses': [60, 80, 100, 150],
        'pore_sizes': [0.3, 0.7, 1.2, 1.5],
        'flux_map': {60: FLUX_RGO, 80: FLUX_RGO, 100: FLUX_RGO, 150: FLUX_RGO}
    },
    'hybrid': {
        'thickness_nm': THICKNESS_HYBRID,
        'pore_size_nm': PORE_SIZE_HYBRID,
        'contact_angle_deg': CONTACT_ANGLE_HYBRID,
        'water_flux_LMH': FLUX_HYBRID,
        'oil_rejection_pct': REJECTION_HYBRID,
        'youngs_modulus_GPa': YOUNG_MODULUS_HYBRID,
        'tensile_strength_MPa': (TENSILE_STRENGTH_GO + TENSILE_STRENGTH_RGO) / 2,  # Standardized for all modules
        'thicknesses': [60, 90, 110, 150],
        'pore_sizes': [0.3, 0.7, 1.2, 1.75],
        'flux_map': {60: FLUX_HYBRID, 90: FLUX_HYBRID, 110: FLUX_HYBRID, 150: FLUX_HYBRID}
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
    "water_hw_charge": 0.417,    "water_density": 1.0           # g/cm³
}

# Phase 4: Chemical and Biological Simulation Constants
PHASE4_CONSTANTS = {
    # Default simulation parameters
    "default_pH": 7.0,
    "default_temperature_C": 25.0,
    "default_reaction_time_min": 180.0,
    "default_initial_concentration_mg_L": 100.0,
    
    # Adsorption model parameters
    "pseudo_second_order_model": True,
    "competitive_adsorption": True,
    
    # Regeneration parameters
    "default_regeneration_efficiency": 0.95,
    "default_max_cycles": 10,
    
    # Biological inactivation parameters
    "default_initial_cfu_ml": 1e6,
    "log_reduction_threshold": 4.0,
    
    # Database file paths
    "contaminant_database": "data/contaminant_data.json",
    "chemical_config": "data/chemical_config.json",
    
    # Supported contaminant categories
    "contaminant_categories": [
        "heavy_metal",
        "PFAS", 
        "organic_pollutant",
        "pesticide",
        "microplastic",
        "bacteria",
        "virus", 
        "salt",
        "inorganic_anion"
    ],
      # Supported membrane types for Phase 4
    "phase4_membranes": ["GO", "rGO", "hybrid"]
}

# Advanced Thermodynamic and Kinetic Constants (Literature-Based)
THERMODYNAMIC_CONSTANTS = {
    # Heavy Metal Thermodynamics (Pb2+ on GO)
    "Pb2+_GO": {
        "delta_G_kJ_mol": -26.3,     # Spontaneous adsorption
        "delta_H_kJ_mol": -19.2,     # Exothermic process
        "delta_S_J_mol_K": 25.1,     # Increased randomness
        "adsorption_mechanism": "physisorption"
    },
    
    # Effective Diffusion Coefficients (m²/s)
    "diffusion_coefficients": {
        "Cd2+_GO": 2.35e-12,         # Time-dependent penetration
        "Pb2+_GO": 1.14e-12          # Slower diffusion than Cd2+
    },
    
    # Zeta Potential Data (mV)
    "zeta_potentials": {
        "GO_pH4": -45.7,
        "GO_pH7": -52.9,
        "rGO_pH4": -25.0,
        "rGO_pH7": -30.0,
        "hybrid_pH4": -35.4,         # Interpolated
        "hybrid_pH7": -41.5          # Interpolated
    },
    
    # Competitive Adsorption Hierarchy
    "selectivity_order": {
        "heavy_metals": ["Pb2+", "Cd2+", "Cr6+"],  # Decreasing affinity
        "affinity_notes": "Pb2+ >> Cd2+ > Cr6+ in binary/ternary systems"
    },
    
    # Adsorption Model Types
    "supported_models": {
        "kinetic": ["pseudo_first_order", "pseudo_second_order", "intraparticle_diffusion"],
        "equilibrium": ["langmuir", "freundlich", "langmuir_freundlich_hybrid"],
        "thermodynamic": ["van_hoff_analysis", "gibbs_free_energy"]
    }
}

# Lake Victoria Field Study Constants
LAKE_VICTORIA_PARAMETERS = {
    # Water Chemistry Ranges (field measurements)
    "pH_range": [6.3, 8.9],
    "conductivity_uS_cm_range": [94.2, 110.5], 
    "turbidity_NTU_range": [21.4, 77.1],
    "temperature_C_range": [24.4, 25.8],
    "chloride_mg_L_range": [1.46, 21.9],
    "nitrate_ug_L_range": [19.3, 313.5],
    "iron_mg_L_range": [0.02, 0.14],
    
    # Biological Filtration Performance
    "pathogen_removal": {
        "E_coli": {
            "log_reduction_range": [2.5, 6.0],
            "exposure_time_min_range": [15, 110],
            "optimal_temp_C": 37,
            "membrane_types": ["GO", "3D-GO-CNT"]
        },
        "Giardia": {
            "log_reduction": 2.1,
            "exposure_time_min": 60,
            "cyst_size_um": [8, 15],
            "mechanism": "physical_exclusion"
        },
        "Salmonella": {
            "log_reduction": 2.2,
            "exposure_time_min": 30,
            "membrane_type": "rGO"
        },
        "Rotavirus": {
            "log_reduction": 7.0,
            "exposure_time_min": 240,
            "membrane_type": "CTAB-rGO-Fe3O4"
        },
        "Adenovirus": {
            "log_reduction": 7.0,
            "exposure_time_min": 240,
            "membrane_type": "CTAB-rGO-Fe3O4"
        }
    },
    
    # Salt Ion Transport Properties  
    "ion_transport": {
        "Na+": {
            "hydrated_radius_nm": 0.358,
            "donnan_exclusion_percent": 48,
            "diffusion_rate_cm2_per_s": 1.33e-5,
            "competitive_effect": "moderate"
        },
        "Cl-": {
            "hydrated_radius_nm": 0.332,
            "donnan_exclusion_percent": 42,
            "diffusion_rate_cm2_per_s": 2.03e-5,
            "competitive_effect": "minimal"
        },
        "Ca2+": {
            "hydrated_radius_nm": 0.412,
            "donnan_exclusion_percent": 74,
            "diffusion_rate_cm2_per_s": 0.79e-5,
            "competitive_effect": "strong_25_35_percent_reduction"
        },
        "Mg2+": {
            "hydrated_radius_nm": 0.428,
            "donnan_exclusion_percent": 78,
            "diffusion_rate_cm2_per_s": 0.71e-5,
            "competitive_effect": "strong_ionic_gating"
        },
        "SO4^2-": {
            "hydrated_radius_nm": 0.379,
            "donnan_exclusion_percent": 88,
            "diffusion_rate_cm2_per_s": 1.07e-5,
            "competitive_effect": "minimal_electrostatic_repulsion"
        }
    },
    
    # PFAS Removal Performance
    "pfas_removal": {
        "PFOA": {
            "removal_efficiency_percent": 100,
            "membrane_type": "GO-CTAC",
            "pH_effective_range": [3, 11],
            "ionic_competition": "none",
            "molecular_weight": 414.07,
            "chain_length": 8
        },
        "PFOS": {
            "removal_efficiency_percent": 100,
            "membrane_type": "GO-CTAC",
            "pH_effective_range": [3, 11],
            "ionic_competition": "maintains_95_percent_high_salinity",
            "molecular_weight": 500.13,
            "chain_length": 8
        },
        "GenX_HFPO-DA": {
            "removal_efficiency_percent": 90,
            "membrane_type": "GO-CTAC",
            "pH_sensitivity": "reduced_above_10",
            "ionic_competition": "competes_with_humic_acids_100_mg_L",
            "molecular_weight": 330.05,
            "replacement_for": "PFOA"
        },
        "PFBS": {
            "removal_efficiency_percent": 85,
            "membrane_type": "MAGO_Magnetic_Amine-GO",
            "pH_optimal_range": [4, 7],
            "ionic_competition": "decreases_with_ionic_strength",
            "molecular_weight": 300.10,
            "chain_length": 4
        }
    },
    
    # Implementation Parameters
    "field_implementation": {
        "regeneration_methods": {
            "hypochlorite_100ppm": {
                "flux_recovery_percent": [85, 95],
                "cycles_before_failure": [50, 100],
                "tropical_suitability": "excellent"
            },
            "solar_UV": {
                "flux_recovery_percent": [80, 90],
                "cycles_before_failure": [40, 80],
                "tropical_suitability": "enhanced"
            },
            "backwash_chemical_combo": {
                "flux_recovery_percent": [90, 98],
                "cycles_before_failure": [75, 150],
                "tropical_suitability": "excellent"
            }
        },
        "composite_membranes": {
            "GO_Zeolite_4A": {
                "water_flux_L_m2_h_bar": 28.9,
                "NaCl_rejection_percent": 53.8,
                "E_coli_removal_percent": "99+",
                "regeneration_cycles": [50, 100]
            },
            "GO_Biochar_Rice": {
                "water_flux_L_m2_h_bar": [25, 40],
                "NaCl_rejection_percent": [35, 50],
                "E_coli_removal_percent": [95, 99],
                "regeneration_cycles": [20, 50]
            },
            "GO_TiO2": {
                "water_flux_L_m2_h_bar": "5_10x_ceramic_supports",
                "NaCl_rejection_percent": 28.7,
                "E_coli_removal_percent": [97, 99],
                "regeneration_cycles": "100+"
            }
        },
        "environmental_conditions": [
            "high_biofouling_tropical",
            "variable_water_quality",
            "temperature_24_40_C",
            "pH_6_3_8_9",
            "elevated_turbidity"
        ]
    }
}

# Field Study Literature References
LAKE_VICTORIA_REFERENCES = [
    "PMC3514835 - GO antimicrobial mechanisms",
    "PMC4690451 - 3D-GO-CNT bacterial inactivation", 
    "PMC9224660 - rGO antibacterial properties",
    "CiteSeerX f4b4ab6398ee4afd - Giardia membrane filtration",
    "MDPI 2304-8158/13/18/2967 - Composite viral removal",
    "PMC11109394 - Na+ membrane transport",
    "ACS Omega 2c00766 - Ion selectivity studies",
    "RSC c6ra21432k - Divalent cation effects",
    "JChemRev 106910 - Ionic gating mechanisms",
    "PMC11249979 - GO-CTAC PFAS removal",
    "ACS EstWater 4c00187 - PFOS removal optimization",
    "NewMOA JiangAdsoprtionApril2024 - GenX studies",
    "RSC d4va00171k - MAGO PFBS removal",
    "KIWASCO-GIZ Field Studies 2024-2025"
]

# Microstructure Variants for GO/rGO Membranes
# Based on BGU lab characterization and field trials in Kenya
MICROSTRUCTURE_VARIANTS = {
    'GO_GO': {
        'interlayer_spacing_nm': 1.05,
        'description': 'Hydrated GO layers with hydrogen bonding',
        'porosity': 0.45,
        'c_o_ratio': 2.1,
        'id_ig_ratio': 0.95,  # D-band/G-band intensity ratio from Raman
        'surface_energy_mJ_m2': 62.1,
        'contact_angle_deg': 29.5,
        'flux_multiplier': 1.2,
        'rejection_multiplier': 0.9,
        'synthesis_conditions': {
            'pH': 2.0,
            'voltage_V': 0,  # Chemical reduction
            'temperature_C': 25,
            'humidity_percent': 65
        }
    },
    'rGO_rGO': {
        'interlayer_spacing_nm': 0.34,
        'description': 'π-π stacked rGO layers, dry conditions',
        'porosity': 0.25,
        'c_o_ratio': 8.5,
        'id_ig_ratio': 2.8,
        'surface_energy_mJ_m2': 110.0,
        'contact_angle_deg': 122.0,
        'flux_multiplier': 0.7,
        'rejection_multiplier': 1.15,
        'synthesis_conditions': {
            'pH': 10.0,
            'voltage_V': 3.5,  # Electrochemical reduction
            'temperature_C': 80,
            'humidity_percent': 15
        }
    },
    'GO_rGO': {
        'interlayer_spacing_nm': 0.80,
        'description': 'GO-rGO hybrid interface with mixed bonding',
        'porosity': 0.35,
        'c_o_ratio': 4.8,
        'id_ig_ratio': 1.6,
        'surface_energy_mJ_m2': 86.0,
        'contact_angle_deg': 75.8,
        'flux_multiplier': 1.0,
        'rejection_multiplier': 1.05,
        'synthesis_conditions': {
            'pH': 6.5,
            'voltage_V': 1.8,  # Partial reduction
            'temperature_C': 50,
            'humidity_percent': 40
        }
    },
    'dry_hybrid': {
        'interlayer_spacing_nm': 0.60,
        'description': 'Compressed hybrid structure, low humidity',
        'porosity': 0.28,
        'c_o_ratio': 5.2,
        'id_ig_ratio': 1.8,
        'surface_energy_mJ_m2': 95.0,
        'contact_angle_deg': 85.0,
        'flux_multiplier': 0.8,
        'rejection_multiplier': 1.1,
        'synthesis_conditions': {
            'pH': 7.0,
            'voltage_V': 2.2,
            'temperature_C': 60,
            'humidity_percent': 20
        }
    },
    'wet_hybrid': {
        'interlayer_spacing_nm': 0.85,
        'description': 'Hydrated hybrid structure, high humidity',
        'porosity': 0.40,
        'c_o_ratio': 4.2,
        'id_ig_ratio': 1.4,
        'surface_energy_mJ_m2': 78.0,
        'contact_angle_deg': 65.0,
        'flux_multiplier': 1.1,
        'rejection_multiplier': 0.95,
        'synthesis_conditions': {
            'pH': 6.0,
            'voltage_V': 1.5,
            'temperature_C': 35,
            'humidity_percent': 75
        }
    }
}

# XRD and Raman Analysis Parameters
CHARACTERIZATION_METHODS = {
    'XRD': {
        'interlayer_spacing_calculation': 'd = λ / (2 * sin(θ))',  # Bragg's law
        'wavelength_nm': 0.154,  # Cu Kα radiation
        'typical_2theta_degrees': {
            'GO': 10.5,    # ~1.05 nm spacing
            'rGO': 26.2,   # ~0.34 nm spacing
            'hybrid': 14.0  # ~0.8 nm spacing
        },
        'uncertainty_nm': 0.05  # ±0.05 nm measurement uncertainty
    },
    'Raman': {
        'D_band_cm_minus1': 1350,  # Disorder-induced band
        'G_band_cm_minus1': 1580,  # Graphitic band
        'ID_IG_interpretation': {
            'GO': 0.8-1.0,     # High disorder
            'rGO': 2.5-3.0,    # Reduced but still defective
            'hybrid': 1.4-1.8  # Intermediate disorder
        },
        'laser_wavelength_nm': 532  # Green laser
    },
    'FTIR': {
        'functional_groups_cm_minus1': {
            'COOH': 1720,     # Carboxyl stretching
            'OH': 3200-3600,  # Hydroxyl stretching
            'epoxy': 1050,    # C-O stretching
            'C=C': 1620       # Aromatic C=C
        }
    }
}

# Variant inference tolerances for lab data matching
VARIANT_INFERENCE = {
    'spacing_tolerance_nm': 0.1,
    'id_ig_tolerance': 0.3,
    'c_o_tolerance': 1.0,
    'confidence_thresholds': {
        'high': 0.9,      # < 0.05 nm difference
        'medium': 0.7,    # < 0.08 nm difference  
        'low': 0.5        # < 0.1 nm difference
    }
}
