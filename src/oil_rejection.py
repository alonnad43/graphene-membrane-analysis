# oil_rejection.py

"""
Phase 1: Estimates oil rejection efficiency using physically accurate sigmoid models.

Scientific approach: Size exclusion combined with wettability effects using sigmoid function.
References: Schmidt et al. 2023, Activated Carbon Blended with rGO 2021
"""

from math import log, exp
import numpy as np
from properties import MEMBRANE_TYPES, OIL_DROPLET_SIZE

def simulate_oil_rejection(pore_size_nm, droplet_size_um, contact_angle_deg, alpha=2.5, beta=0.1):
    """
    Sigmoid-based empirical model for oil rejection by GO/rGO membranes.
    
    Args:
        pore_size_nm (float): Pore size in nanometers
        droplet_size_um (float): Oil droplet size in micrometers
        contact_angle_deg (float): Water contact angle in degrees
        alpha (float): Size exclusion parameter (default: 2.5)
        beta (float): Wettability parameter (default: 0.1)
    
    Returns:
        float: Oil rejection efficiency (0-100%)
    
    Scientific equation:
        R = 1 / (1 + exp(-(α * log(size_ratio) + β * (90 - θ))))
        Where size_ratio = droplet_diameter / pore_diameter, θ = contact_angle
    """
    # Input validation
    if droplet_size_um <= 0 or pore_size_nm <= 0:
        return 0.0
    
    # Unit conversions to same scale (both in nm)
    pore_size = pore_size_nm                    # Already in nm
    droplet_size = droplet_size_um * 1000       # Convert μm to nm
    
    # Calculate size ratio
    size_ratio = droplet_size / pore_size
    
    # Handle edge cases
    if size_ratio <= 1.0:
        # Droplet smaller than pore - minimal rejection
        return 5.0  # Small baseline rejection due to surface effects
    
    # Sigmoid model with size exclusion and wettability
    theta = contact_angle_deg
    
    try:
        # Sigmoid function: R = 1 / (1 + exp(-(α * log(size_ratio) + β * (90 - θ))))
        exponent = -(alpha * log(size_ratio) + beta * (90 - theta))
        rejection = 1 / (1 + exp(exponent))
        
        # Ensure result is in valid range [0, 1] and convert to percentage
        rejection = min(max(rejection, 0), 1) * 100
        
    except (ValueError, OverflowError):
        # Fallback for numerical issues
        if size_ratio > 10:
            rejection = 95.0  # High rejection for large size ratios
        else:
            rejection = 50.0  # Moderate rejection as fallback
    
    return round(rejection, 2)

def simulate_oil_rejection_stochastic(pore_size_nm, droplet_size_um, contact_angle_deg, membrane_type=None, alpha=2.5, beta=0.1):
    """
    Enhanced oil rejection model with stochasticity, physical upper/lower limits, and literature-based ranges.
    
    Args:
        pore_size_nm (float): Pore size in nanometers
        droplet_size_um (float): Oil droplet size in micrometers
        contact_angle_deg (float): Water contact angle in degrees
        membrane_type (str): 'GO', 'rGO', or 'hybrid' (for literature-based range)
        alpha (float): Size exclusion parameter
        beta (float): Wettability parameter
    
    Returns:
        float: Oil rejection efficiency (literature-based, 0–100%)
    """
    # Literature-based ranges
    LIT_RANGES = {
        'GO': (85, 95),
        'rGO': (92, 97),
        'hybrid': (89, 99)
    }
    # Add log-normal variability to pore size (10% std dev)
    effective_pore = pore_size_nm * np.random.lognormal(0, 0.1)
    # Add Gaussian noise to contact angle (±3°)
    effective_contact_angle = contact_angle_deg + np.random.normal(0, 3)
    # Add variability to droplet size (±20%)
    effective_droplet = droplet_size_um * np.random.normal(1, 0.2)
    # Unit conversions
    pore_size = max(0.1, effective_pore)
    droplet_size = max(0.1, effective_droplet * 1000)  # μm to nm
    # Calculate size ratio
    size_ratio = droplet_size / pore_size
    if size_ratio <= 1.0:
        base = 5.0
    else:
        theta = effective_contact_angle
        try:
            exponent = -(alpha * np.log(size_ratio) + beta * (90 - theta))
            rejection = 1 / (1 + np.exp(exponent))
            rejection = min(max(rejection, 0), 1) * 100
        except (ValueError, OverflowError):
            rejection = 95.0 if size_ratio > 10 else 50.0
        base = rejection
    # Clamp to literature-based range for membrane type
    if membrane_type in LIT_RANGES:
        lo, hi = LIT_RANGES[membrane_type]
        # Add ±2–5% random spread within the range
        spread = np.random.uniform(-2, 2)
        base = np.clip(base + spread, lo, hi)
    else:
        base = min(max(base, 0), 99.5)
    return round(base, 2)

# Backward compatibility function
def simulate_oil_rejection_old(membrane_type, pore_size_nm=None):
    """
    Legacy function for backward compatibility.
    Uses empirical rejection mappings from properties.
    """
    if membrane_type not in MEMBRANE_TYPES:
        raise ValueError(f"Membrane type '{membrane_type}' not recognized.")

    membrane_data = MEMBRANE_TYPES[membrane_type]

    if "rejection_map" in membrane_data and pore_size_nm is not None:
        # Choose the closest available pore size in the rejection map
        closest_pore = min(
            membrane_data["rejection_map"].keys(),
            key=lambda known: abs(known - pore_size_nm)
        )
        return membrane_data["rejection_map"][closest_pore]
    elif "rejection" in membrane_data:
        return membrane_data["rejection"]
    else:
        return 0.0  # Default for undefined cases
