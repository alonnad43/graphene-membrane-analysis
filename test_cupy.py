import cupy as cp

# Show CuPy version
print("CuPy version:", cp.__version__)

# Check GPU device info
device_id = cp.cuda.runtime.getDevice()
device_props = cp.cuda.runtime.getDeviceProperties(device_id)

print(f"Device ID: {device_id}")
print(f"Device Name: {device_props['name'].decode()}")
print(f"Memory (MB): {device_props['totalGlobalMem'] // 1024**2}")

# Run small test computation
a = cp.arange(1_000_000, dtype=cp.float32)
b = cp.sin(a)
print("Test sum of sin(a):", cp.sum(b).item())
