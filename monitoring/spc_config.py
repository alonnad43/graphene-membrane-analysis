# SPC Alert & Logging Configuration
SPC_CONFIG = {
    "log_file": "logs/spc_alerts.log",
    "alert_method": "print"  # or "email"
}

# SPC Chart Parameters (subgroup size n and constants)
SPC_CONSTANTS = {
    4: {"A2": 0.729, "D3": 0.000, "D4": 2.282},
    5: {"A2": 0.577, "D3": 0.000, "D4": 2.114}
}
