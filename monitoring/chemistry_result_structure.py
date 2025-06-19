# Standardized result data structure for simulation outputs
chemistry_result_example = {
    "time_points": [0, 10, 20, 30, 40, 50, 60],
    "Pb2+": {
        "concentration_mg_L": [100, 80, 60, 45, 30, 20, 12],
        "removal_efficiency_pct": 88.0
    },
    "E_coli": {
        "concentration_CFU_mL": [1e6, 5e5, 1e4, 1e3, 100, 10, 1],
        "log_reduction": 6.0
    }
}
