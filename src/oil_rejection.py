# oil_rejection.py

"""
Simulates oil rejection efficiency for GO, rGO, and Hybrid membranes.

Provides a function to estimate oil rejection efficiency (%) based on membrane type
and optionally pore size, using data from properties.py.
"""

from properties import MEMBRANE_TYPES

def simulate_oil_rejection(membrane_type, pore_size_nm=None):
    """
    Returns oil rejection efficiency (%) based on membrane type and optional pore size.
    
    Args:
        membrane_type (str): 'GO', 'rGO', or 'Hybrid'
        pore_size_nm (float, optional): Used to interpolate rejection if known for that membrane type.
        
    Returns:
        float: Estimated oil rejection efficiency (%)
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
