"""
General Simulation Parameters
-----------------------------
Used across all batch simulations. Defines default run time, feed concentrations,
temperature, and pH. You can override per run using a config file or CLI flags.
"""

simulation_config = {
    "reaction_time": 180,         # total simulation time in minutes [6]
    "timestep": 10,               # timestep in minutes [6]
    "pH": 7.0,                    # lab-controlled neutral pH [9]
    "temperature_C": 25,          # ambient lab temperature in Celsius [9]
    "initial_concentrations": {
        "Pb2+": 100,              # starting concentration (mg/L) [7]
        "E_coli": 1e6             # initial CFU/mL [8]
    }
}
