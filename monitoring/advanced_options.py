"""
Advanced Optimization Flags
----------------------------
Use these if running long simulations or deploying to limited hardware.
"""

optimization_settings = {
    "memory_limit_gb": 8,         # limit RAM usage [local or cloud]
    "chunk_size": 1000,           # number of points per processing batch
    "precision": "float32"        # float64 for accuracy, float32 for speed
}
