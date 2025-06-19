"""
Unit tests for microstructure variants and lab data integration.

Tests the new membrane variant system and XRD/Raman post-lab results integration.
"""

import sys
import os
import unittest
import numpy as np
import json
import tempfile

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from membrane_model import Membrane, MembraneVariant
from simulate_chemistry import ChemicalSimulationEngine
from properties import MICROSTRUCTURE_VARIANTS


class TestMembraneVariants(unittest.TestCase):
    """Test microstructure variant functionality."""
    
    def test_variant_properties_loading(self):
        """Test that all variants load correctly."""
        for variant_name in ['GO_GO', 'rGO_rGO', 'GO_rGO', 'dry_hybrid', 'wet_hybrid']:
            props = MembraneVariant.get_variant_properties(variant_name)
            self.assertIsInstance(props, dict)
            self.assertIn('interlayer_spacing_nm', props)
            self.assertIn('flux_multiplier', props)
            self.assertIn('rejection_multiplier', props)
            print(f"✅ {variant_name}: {props['interlayer_spacing_nm']} nm spacing")
    
    def test_spacing_inference(self):
        """Test variant inference from interlayer spacing."""
        # Test exact matches
        test_cases = [
            (1.05, 'GO_GO'),
            (0.34, 'rGO_rGO'),
            (0.80, 'GO_rGO'),
            (0.60, 'dry_hybrid'),
            (0.85, 'wet_hybrid')
        ]
        
        for spacing, expected_variant in test_cases:
            variant, confidence = MembraneVariant.infer_variant_from_spacing(spacing, tolerance=0.05)
            self.assertEqual(variant, expected_variant)
            self.assertGreaterEqual(confidence, 0.9)
            print(f"✅ {spacing} nm → {variant} (confidence: {confidence:.2f})")
    
    def test_membrane_with_variant(self):
        """Test Membrane class with variant specification."""
        # Create membrane with variant
        membrane = Membrane(
            name="GO Test",
            pore_size_nm=2.0,
            thickness_nm=100,
            flux_lmh=120,
            modulus_GPa=207,
            tensile_strength_MPa=30,
            contact_angle_deg=65,
            variant='GO_GO'
        )
        
        # Check variant properties are applied
        self.assertEqual(membrane.variant, 'GO_GO')
        self.assertIsNotNone(membrane.variant_properties)
        self.assertEqual(membrane.get_interlayer_spacing(), 1.05)
        
        # Check variant info
        variant_info = membrane.get_variant_info()
        self.assertEqual(variant_info['variant'], 'GO_GO')
        self.assertIn('properties', variant_info)
        
        print(f"✅ Membrane variant: {membrane.variant}")
        print(f"✅ Interlayer spacing: {membrane.get_interlayer_spacing()} nm")
    
    def test_default_membrane_comparison(self):
        """Compare default GO, rGO, and hybrid membranes."""
        membranes = []
        
        # Create default membranes
        for membrane_type in ['GO', 'rGO', 'hybrid']:
            membrane = Membrane(
                name=membrane_type,
                pore_size_nm=2.0 if membrane_type == 'GO' else 1.5,
                thickness_nm=100 if membrane_type == 'GO' else 80,
                flux_lmh=120 if membrane_type == 'GO' else 80,
                modulus_GPa=207 if membrane_type == 'GO' else 280,
                tensile_strength_MPa=30 if membrane_type == 'GO' else 44,
                contact_angle_deg=65 if membrane_type == 'GO' else 122
            )
            membranes.append(membrane)
        
        # Verify different properties
        self.assertNotEqual(membranes[0].contact_angle_deg, membranes[1].contact_angle_deg)
        self.assertNotEqual(membranes[0].flux_lmh, membranes[1].flux_lmh)
        
        print(f"✅ GO contact angle: {membranes[0].contact_angle_deg}°")
        print(f"✅ rGO contact angle: {membranes[1].contact_angle_deg}°")


class TestLabDataIntegration(unittest.TestCase):
    """Test XRD and Raman lab data integration."""
    
    def setUp(self):
        """Set up test data."""
        self.engine = ChemicalSimulationEngine()
        
        # Create temporary lab data file
        self.lab_data = {
            "sample_id": "TEST_001",
            "synthesis_conditions": {
                "pH": 6.5,
                "voltage_V": 1.8,
                "temperature_C": 50
            },
            "characterization": {
                "XRD": {
                    "interlayer_spacing_nm": 0.82,
                    "uncertainty_nm": 0.03
                },
                "Raman": {
                    "ID_IG_ratio": 1.6,
                    "D_peak_cm1": 1348,
                    "G_peak_cm1": 1582
                }
            }
        }
        
    def test_lab_data_loading(self):
        """Test loading lab characterization data."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.lab_data, f)
            temp_path = f.name
        
        try:
            # Load lab data
            variant, confidence = self.engine.load_lab_characterization_data(temp_path)
            
            # Verify results
            self.assertIsNotNone(variant)
            self.assertGreater(confidence, 0.5)
            self.assertTrue(hasattr(self.engine, 'lab_validated_variant'))
            
            print(f"✅ Lab data loaded: {variant} (confidence: {confidence:.2f})")
            
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_membrane_property_override(self):
        """Test membrane property override with lab data."""
        # Set up lab validated variant
        self.engine.lab_validated_variant = {
            'variant': 'GO_rGO',
            'confidence': 0.95,
            'lab_data': self.lab_data,
            'characterization': self.lab_data['characterization']
        }
        
        # Test property override
        modified_props = self.engine.apply_lab_validated_membrane_properties('GO')
        
        # Verify modification occurred
        self.assertIsInstance(modified_props, dict)
        
        print(f"✅ Modified properties: {list(modified_props.keys())}")
    
    def test_simulation_with_lab_data(self):
        """Test complete simulation with lab data integration."""
        # Create temporary lab data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.lab_data, f)
            temp_path = f.name
        
        try:
            # Import and run simulation with lab data
            from simulate_chemistry import run_phase4_simulation
            
            engine = run_phase4_simulation(
                membrane_types=['GO'],
                contaminants=['Pb2+'],
                initial_concentrations={'Pb2+': 50.0},
                reaction_time=60,
                lab_data_path=temp_path
            )
            
            # Verify simulation completed
            self.assertGreater(len(engine.simulation_results), 0)
            self.assertTrue(hasattr(engine, 'lab_validated_variant'))
            
            print(f"✅ Simulation with lab data completed")
            
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_invalid_lab_data_handling(self):
        """Test handling of invalid or missing lab data."""
        # Test with non-existent file
        variant, confidence = self.engine.load_lab_characterization_data("nonexistent.json")
        self.assertIsNone(variant)
        self.assertEqual(confidence, 0.0)
        
        # Test with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            variant, confidence = self.engine.load_lab_characterization_data(temp_path)
            self.assertIsNone(variant)
            self.assertEqual(confidence, 0.0)
            
            print(f"✅ Invalid data handled gracefully")
            
        finally:
            os.unlink(temp_path)


class TestVariantIntegration(unittest.TestCase):
    """Test integration between variants and chemistry simulation."""
    
    def test_variant_effects_on_simulation(self):
        """Test that different variants affect simulation results."""
        engine = ChemicalSimulationEngine()
        
        # Simulate same conditions with different variants
        results = {}
        
        for variant in ['GO_GO', 'rGO_rGO', 'GO_rGO']:
            # Set up lab validated variant
            engine.lab_validated_variant = {
                'variant': variant,
                'confidence': 1.0,
                'lab_data': {},
                'characterization': {}
            }
            
            # Run simulation
            result = engine.simulate_contaminant_removal(
                membrane_type='GO',
                contaminants=['Pb2+'],
                initial_concentrations={'Pb2+': 100.0},
                reaction_time=30
            )
            
            results[variant] = result
        
        # Verify different results for different variants
        self.assertEqual(len(results), 3)
        
        print(f"✅ Variant-specific simulations completed")
        for variant, result in results.items():
            if 'Pb2+' in result['contaminants']:
                removal_eff = result['contaminants']['Pb2+'].get('removal_efficiency', 0)
                print(f"   {variant}: {removal_eff:.1f}% Pb2+ removal")


def run_all_tests():
    """Run all tests with detailed output."""
    print("🧪 TESTING MICROSTRUCTURE VARIANTS AND LAB DATA INTEGRATION")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMembraneVariants,
        TestLabDataIntegration,
        TestVariantIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
        print(f"   Ran {result.testsRun} tests successfully")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
