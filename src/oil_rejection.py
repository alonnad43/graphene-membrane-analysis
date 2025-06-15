# oil_rejection.py

"""
Phase 1: Estimates oil rejection efficiency based on physical models.

Scientific approach: Combines size exclusion and wettability effects.
References: Hu et al. 2014, Tee et al. 2024
"""

from properties import MEMBRANE_TYPES, OIL_DROPLET_SIZE

def simulate_oil_rejection(pore_size_nm, droplet_size_um, contact_angle_deg):
    """
    Estimate oil rejection based on droplet size, pore size, and contact angle.
    
    Args:
        pore_size_nm (float): Pore size in nanometers
        droplet_size_um (float): Oil droplet size in micrometers
        contact_angle_deg (float): Water contact angle in degrees
    
    Returns:
        float: Oil rejection efficiency (%)
    
    Scientific basis:
        - Size exclusion: Large droplets cannot pass through small pores
        - Wettability: Hydrophilic surfaces reject oil more effectively
        - Combined empirical model based on experimental data
    """
    # Unit conversions
    pore_d = pore_size_nm * 1e-9     # Convert nm to m
    drop_d = droplet_size_um * 1e-6  # Convert μm to m

    # Size exclusion: if droplet is smaller than pore, it can pass through
    if drop_d <= pore_d:
        return 0.0  # Full passage, no rejection

    # Size ratio factor
    size_ratio = drop_d / pore_d
    
    # Wettability factor: hydrophilic surfaces (low contact angle) reject oil better
    # Contact angle: 0° = fully hydrophilic, 90° = neutral, >90° = hydrophobic
    wettability_factor = max(0, (90 - contact_angle_deg) / 90)  # 1 = fully hydrophilic, 0 = hydrophobic
    
    # Combined rejection model
    rejection = min(100, 100 * (1 - 1 / size_ratio) * wettability_factor)
    
    return round(rejection, 2)

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
