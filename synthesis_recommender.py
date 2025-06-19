"""
Synthesis Recommender: Suggests the optimal synthesis method (electrochemical, solar, or biological)
based on field parameters and logs reduction quality (C/O ratio, ID/IG ratio) for comparison to target specs.

Required Input Data:
- Temperature (°C): Ambient or process temperature at the synthesis site.
- Sunlight (W/m²): Available solar irradiance (for solar method).
- Biomass (kg): Amount of biological material available (for biological method).

Optional Input Data:
- Actual C/O Ratio: Measured carbon-to-oxygen ratio of the product (from XPS or similar).
- Actual ID/IG Ratio: Measured Raman D/G band intensity ratio of the product.

Usage:
    python synthesis_recommender.py <temp_C> <sunlight_Wm2> <biomass_kg> [actual_CO actual_IDIG]
"""
import json
import sys

SYNTHESIS_METHODS = {
    "electrochemical": {
        "min_temp": 20,
        "max_temp": 60,
        "min_sunlight": 0,
        "max_sunlight": 1000,
        "min_biomass": 0,
        "max_biomass": 10,
        "target_CO": 4.0,
        "target_IDIG": 1.5
    },
    "solar": {
        "min_temp": 10,
        "max_temp": 50,
        "min_sunlight": 500,
        "max_sunlight": 2000,
        "min_biomass": 0,
        "max_biomass": 10,
        "target_CO": 2.5,
        "target_IDIG": 1.0
    },
    "biological": {
        "min_temp": 15,
        "max_temp": 40,
        "min_sunlight": 0,
        "max_sunlight": 1500,
        "min_biomass": 5,
        "max_biomass": 100,
        "target_CO": 3.0,
        "target_IDIG": 1.2
    }
}

def recommend_method(temp_C, sunlight_Wm2, biomass_kg):
    candidates = []
    for method, params in SYNTHESIS_METHODS.items():
        if (params['min_temp'] <= temp_C <= params['max_temp'] and
            params['min_sunlight'] <= sunlight_Wm2 <= params['max_sunlight'] and
            params['min_biomass'] <= biomass_kg <= params['max_biomass']):
            candidates.append(method)
    if not candidates:
        return None
    # Prefer method with closest target C/O ratio to 3.0 (as a generic optimal)
    candidates.sort(key=lambda m: abs(SYNTHESIS_METHODS[m]['target_CO'] - 3.0))
    return candidates[0]

def log_reduction_quality(method, actual_CO, actual_IDIG):
    target_CO = SYNTHESIS_METHODS[method]['target_CO']
    target_IDIG = SYNTHESIS_METHODS[method]['target_IDIG']
    print(f"Method: {method}")
    print(f"  Target C/O: {target_CO}, Actual C/O: {actual_CO}")
    print(f"  Target ID/IG: {target_IDIG}, Actual ID/IG: {actual_IDIG}")
    print(f"  C/O deviation: {abs(target_CO - actual_CO):.2f}")
    print(f"  ID/IG deviation: {abs(target_IDIG - actual_IDIG):.2f}")

def main():
    if len(sys.argv) < 4:
        print("Usage: python synthesis_recommender.py <temp_C> <sunlight_Wm2> <biomass_kg> [actual_CO actual_IDIG]")
        sys.exit(1)
    temp_C = float(sys.argv[1])
    sunlight_Wm2 = float(sys.argv[2])
    biomass_kg = float(sys.argv[3])
    method = recommend_method(temp_C, sunlight_Wm2, biomass_kg)
    if not method:
        print("No suitable synthesis method found for the given field parameters.")
        sys.exit(1)
    print(f"Recommended synthesis method: {method}")
    if len(sys.argv) >= 6:
        actual_CO = float(sys.argv[4])
        actual_IDIG = float(sys.argv[5])
        log_reduction_quality(method, actual_CO, actual_IDIG)

if __name__ == "__main__":
    main()
