#!/usr/bin/env python3
"""
Debug script with timeout to identify where the code is hanging.
Windows-compatible version using threading.
"""

import sys
import time
import traceback
import threading
import os

class TimeoutException(Exception):
    pass

def test_with_timeout():
    def timeout_handler():
        time.sleep(10)  # 10 second timeout
        print(f"\n❌ TIMEOUT: Code execution exceeded 10 seconds!")
        print("   Terminating process...")
        os._exit(1)  # Force exit
    
    # Start timeout thread
    timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
    timeout_thread.start()
    
    try:
        print("=== Phase 1 Debug Test (10s timeout) ===")
        
        # Step 1: Test basic imports
        print("Step 1: Testing basic imports...")
        sys.path.append('src')
        
        # Step 2: Test properties import
        print("Step 2: Testing properties import...")
        from properties import WATER_PROPERTIES
        print(f"   ✅ WATER_PROPERTIES loaded: {type(WATER_PROPERTIES)}")
        
        # Step 3: Test flux_simulator import
        print("Step 3: Testing flux_simulator import...")
        from flux_simulator import calculate_temperature_viscosity
        print("   ✅ calculate_temperature_viscosity imported")
        
        # Step 4: Test viscosity calculation
        print("Step 4: Testing viscosity calculation...")
        viscosity = calculate_temperature_viscosity(298)
        print(f"   ✅ Viscosity: {viscosity:.6f} Pa·s")
        
        # Step 5: Test flux simulator import
        print("Step 5: Testing simulate_flux import...")
        from flux_simulator import simulate_flux
        print("   ✅ simulate_flux imported")
        
        # Step 6: Test flux calculation with explicit parameters
        print("Step 6: Testing flux calculation...")
        flux = simulate_flux(
            pore_size_nm=1.0,
            thickness_nm=100.0,
            pressure_bar=1.0,
            viscosity_pas=0.00089,
            temperature=298,
            porosity=0.35,
            tortuosity=2.0
        )
        print(f"   ✅ Flux: {flux:.2f} L·m⁻²·h⁻¹")
        
        # Step 7: Test oil rejection
        print("Step 7: Testing oil rejection...")
        from oil_rejection import simulate_oil_rejection
        rejection = simulate_oil_rejection(
            pore_size_nm=1.0,
            droplet_size_um=5.0,
            contact_angle_deg=30.0
        )
        print(f"   ✅ Oil rejection: {rejection:.1f}%")
        
        print("\n✅ ALL TESTS PASSED - No hanging issues detected!")
        
    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("Starting debug test with timeout protection...")
    success = test_with_timeout()
    if success:
        print("\n🎉 Debug test completed successfully!")
    else:
        print("\n💥 Debug test failed - check errors above")
        sys.exit(1)
