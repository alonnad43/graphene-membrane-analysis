#!/usr/bin/env python3
"""
Simple test of membrane variants without the lab data integration.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from membrane_model import Membrane, MembraneVariant
from properties import MICROSTRUCTURE_VARIANTS

def test_basic_variants():
    """Test basic membrane variant functionality."""
    print("🧪 TESTING BASIC MEMBRANE VARIANTS")
    print("=" * 50)
    
    # Test 1: List all variants
    print("\n1. Available Variants:")
    for variant_name, props in MICROSTRUCTURE_VARIANTS.items():
        print(f"   {variant_name}: {props['interlayer_spacing_nm']:.2f} nm - {props['description']}")
    
    # Test 2: Variant inference
    print("\n2. Variant Inference Tests:")
    test_cases = [
        (1.05, 'GO_GO'),
        (0.34, 'rGO_rGO'),
        (0.80, 'GO_rGO'),
        (0.60, 'dry_hybrid'),
        (0.85, 'wet_hybrid')
    ]
    
    for spacing, expected in test_cases:
        variant, confidence = MembraneVariant.infer_variant_from_spacing(spacing)
        status = "✅" if variant == expected else "❌"
        print(f"   {status} {spacing} nm → {variant} (confidence: {confidence:.2f})")
    
    # Test 3: Membrane with variant
    print("\n3. Membrane Creation with Variant:")
    membrane = Membrane(
        name="Test GO",
        pore_size_nm=2.0,
        thickness_nm=100,
        flux_lmh=120,
        modulus_GPa=207,
        tensile_strength_MPa=30,
        contact_angle_deg=65,
        variant='GO_GO'
    )
    
    print(f"   Created membrane: {membrane.name}")
    print(f"   Variant: {membrane.variant}")
    print(f"   Interlayer spacing: {membrane.get_interlayer_spacing():.3f} nm")
    print(f"   Contact angle: {membrane.contact_angle_deg:.1f}°")
    
    # Test 4: Variant effects
    print("\n4. Variant Effects on Properties:")
    base_flux = 120
    base_rejection = 85
    
    for variant_name in ['GO_GO', 'rGO_rGO', 'GO_rGO']:
        props = MembraneVariant.get_variant_properties(variant_name)
        modified_flux = base_flux * props['flux_multiplier']
        modified_rejection = base_rejection * props['rejection_multiplier']
        
        print(f"   {variant_name}:")
        print(f"     Flux: {base_flux:.0f} → {modified_flux:.0f} L/m²/h ({props['flux_multiplier']:.1f}x)")
        print(f"     Rejection: {base_rejection:.0f} → {modified_rejection:.0f}% ({props['rejection_multiplier']:.1f}x)")
    
    print("\n✅ All basic variant tests completed successfully!")

if __name__ == "__main__":
    test_basic_variants()
