"""
Mathematical Models for Contaminant Removal
-------------------------------------------
Contains core equations for:
- Adsorption (pseudo-second-order)
- Bacterial inactivation (first-order decay)
- Salt rejection ratio
"""

def adsorption_rate(q, qmax, k2):
    # dq/dt = k2 * (qmax - q)^2
    return k2 * (qmax - q)**2

def bacterial_decay(N0, k, t):
    # N(t) = N0 * exp(-k * t)
    from math import exp
    return N0 * exp(-k * t)

def salt_rejection_ratio(R, C_feed):
    # C_permeate = (1 - R) * C_feed
    return (1 - R) * C_feed
