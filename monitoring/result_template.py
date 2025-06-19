"""
Simulation Output Data Format
-----------------------------
This structure holds time-series results and computed performance metrics.
Use this to export results to CSV, JSON, or for plotting.
"""

results = {
    "time_points": [0, 10, 20, 30],
    "Pb2+": {
        "concentration": [],       # simulated values over time
        "removal_efficiency": None  # computed final % removal
    },
    "E_coli": {
        "concentration": [],
        "log_reduction": None
    },
    "NaCl": {
        "permeate_conc": [],
        "rejection": 0.85
    }
}
