#!/usr/bin/env python3
"""
Membrane Variant Selector CLI

Command-line interface for selecting membrane microstructure variants
and running simulations with custom interlayer spacing parameters.
"""

import argparse
import sys
import os
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from membrane_model import MembraneVariant, Membrane
from simulate_chemistry import run_phase4_simulation
from properties import MICROSTRUCTURE_VARIANTS


def list_variants():
    """List all available membrane variants."""
    print("\n🔬 AVAILABLE MEMBRANE MICROSTRUCTURE VARIANTS")
    print("=" * 60)
    
    for variant_name, props in MICROSTRUCTURE_VARIANTS.items():
        print(f"\n{variant_name}:")
        print(f"  Interlayer spacing: {props['interlayer_spacing_nm']:.2f} nm")
        print(f"  Description: {props['description']}")
        print(f"  C/O ratio: {props['c_o_ratio']}")
        print(f"  ID/IG ratio: {props['id_ig_ratio']}")
        print(f"  Contact angle: {props['contact_angle_deg']:.1f}°")
        print(f"  Flux multiplier: {props['flux_multiplier']:.2f}x")
        print(f"  Rejection multiplier: {props['rejection_multiplier']:.2f}x")


def infer_variant_interactive():
    """Interactive variant inference from user input."""
    print("\n🔍 VARIANT INFERENCE FROM LAB DATA")
    print("=" * 40)
    
    try:
        # Get interlayer spacing
        spacing_input = input("Enter XRD interlayer spacing (nm): ").strip()
        if not spacing_input:
            print("No spacing provided. Exiting.")
            return None
            
        spacing = float(spacing_input)
        
        # Get tolerance
        tolerance_input = input("Enter measurement uncertainty (±nm) [default: 0.1]: ").strip()
        tolerance = float(tolerance_input) if tolerance_input else 0.1
        
        # Infer variant
        variant, confidence = MembraneVariant.infer_variant_from_spacing(spacing, tolerance)
        
        print(f"\n📊 INFERENCE RESULTS:")
        print(f"Measured spacing: {spacing:.3f} ± {tolerance:.3f} nm")
        print(f"Closest variant: {variant}")
        print(f"Confidence: {confidence:.2f}")
        
        if confidence >= 0.9:
            print("✅ High confidence match")
        elif confidence >= 0.7:
            print("⚠️ Medium confidence match")
        else:
            print("❌ Low confidence match - consider alternative variants")
        
        # Show variant details
        variant_props = MembraneVariant.get_variant_properties(variant)
        print(f"\nVariant details:")
        print(f"  Expected spacing: {variant_props['interlayer_spacing_nm']:.3f} nm")
        print(f"  Description: {variant_props['description']}")
        
        return variant
        
    except ValueError as e:
        print(f"Error: Invalid input - {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def create_membrane_with_variant():
    """Interactive membrane creation with variant selection."""
    print("\n🏗️ CREATE MEMBRANE WITH VARIANT")
    print("=" * 35)
    
    # Membrane basic properties
    name = input("Membrane name [default: Custom]: ").strip() or "Custom"
    
    # Get variant
    print("\nSelect variant:")
    print("1. Choose from list")
    print("2. Infer from XRD data")
    print("3. Custom spacing")
    
    choice = input("Choice (1-3): ").strip()
    
    variant = None
    custom_spacing = None
    
    if choice == '1':
        # List variants and select
        list_variants()
        variant_names = list(MICROSTRUCTURE_VARIANTS.keys())
        print(f"\nAvailable variants: {', '.join(variant_names)}")
        variant = input("Enter variant name: ").strip()
        
        if variant not in variant_names:
            print(f"Invalid variant. Using GO_rGO as default.")
            variant = 'GO_rGO'
            
    elif choice == '2':
        # Infer from XRD
        variant = infer_variant_interactive()
        
    elif choice == '3':
        # Custom spacing
        try:
            custom_spacing = float(input("Enter custom interlayer spacing (nm): ").strip())
            print(f"Using custom spacing: {custom_spacing:.3f} nm")
        except ValueError:
            print("Invalid spacing. Using default.")
            custom_spacing = 0.8
    
    # Create membrane
    membrane = Membrane(
        name=name,
        pore_size_nm=2.0,  # Default
        thickness_nm=100,  # Default
        flux_lmh=120,     # Default
        modulus_GPa=207,  # Default
        tensile_strength_MPa=30,  # Default
        contact_angle_deg=65,     # Default
        variant=variant
    )
    
    # Display membrane info
    print(f"\n✅ MEMBRANE CREATED: {membrane.name}")
    print(f"Variant: {membrane.variant or 'Default'}")
    print(f"Interlayer spacing: {membrane.get_interlayer_spacing():.3f} nm")
    print(f"Contact angle: {membrane.contact_angle_deg:.1f}°")
    
    return membrane


def run_simulation_with_variant():
    """Run Phase 4 simulation with selected variant."""
    print("\n🧪 RUN SIMULATION WITH VARIANT")
    print("=" * 32)
    
    # Get membrane types
    membrane_input = input("Membrane types (comma-separated) [default: GO,rGO]: ").strip()
    membrane_types = [m.strip() for m in membrane_input.split(',')] if membrane_input else ['GO', 'rGO']
    
    # Get contaminants
    contaminant_input = input("Contaminants (comma-separated) [default: Pb2+]: ").strip()
    contaminants = [c.strip() for c in contaminant_input.split(',')] if contaminant_input else ['Pb2+']
    
    # Check for lab data file
    lab_data_path = None
    lab_file_input = input("Lab characterization file path (optional): ").strip()
    if lab_file_input and os.path.exists(lab_file_input):
        lab_data_path = lab_file_input
        print(f"Using lab data: {lab_data_path}")
    elif lab_file_input:
        print(f"Warning: File not found - {lab_file_input}")
    
    # Run simulation
    print(f"\n🚀 Starting simulation...")
    print(f"Membrane types: {membrane_types}")
    print(f"Contaminants: {contaminants}")
    
    try:
        engine = run_phase4_simulation(
            membrane_types=membrane_types,
            contaminants=contaminants,
            initial_concentrations={c: 100.0 for c in contaminants},
            reaction_time=180,
            lab_data_path=lab_data_path
        )
        
        print(f"\n✅ Simulation completed successfully!")
        print(f"Results saved to output/ directory")
        
        return engine
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return None


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Membrane Variant Selector and Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python membrane_variant_cli.py --list
  python membrane_variant_cli.py --infer
  python membrane_variant_cli.py --create
  python membrane_variant_cli.py --simulate
  python membrane_variant_cli.py --simulate --lab-data data/lab_characterization_example.json
        """
    )
    
    # Add arguments
    parser.add_argument('--list', action='store_true',
                       help='List all available membrane variants')
    parser.add_argument('--infer', action='store_true',
                       help='Infer variant from XRD interlayer spacing')
    parser.add_argument('--create', action='store_true',
                       help='Create membrane interactively with variant')
    parser.add_argument('--simulate', action='store_true',
                       help='Run Phase 4 simulation with variant selection')
    parser.add_argument('--lab-data', type=str,
                       help='Path to lab characterization JSON file')
    parser.add_argument('--spacing', type=float,
                       help='Directly specify interlayer spacing (nm)')
    parser.add_argument('--membrane-types', type=str, default='GO,rGO',
                       help='Comma-separated membrane types (default: GO,rGO)')
    parser.add_argument('--contaminants', type=str, default='Pb2+',
                       help='Comma-separated contaminants (default: Pb2+)')
    
    args = parser.parse_args()
    
    # Handle arguments
    if args.list:
        list_variants()
        
    elif args.infer:
        if args.spacing:
            variant, confidence = MembraneVariant.infer_variant_from_spacing(args.spacing)
            print(f"Spacing: {args.spacing} nm → Variant: {variant} (confidence: {confidence:.2f})")
        else:
            infer_variant_interactive()
            
    elif args.create:
        create_membrane_with_variant()
        
    elif args.simulate:
        if args.lab_data or args.spacing:
            # Non-interactive simulation
            membrane_types = [m.strip() for m in args.membrane_types.split(',')]
            contaminants = [c.strip() for c in args.contaminants.split(',')]
            
            print(f"Running simulation with:")
            print(f"  Membrane types: {membrane_types}")
            print(f"  Contaminants: {contaminants}")
            if args.lab_data:
                print(f"  Lab data: {args.lab_data}")
            
            engine = run_phase4_simulation(
                membrane_types=membrane_types,
                contaminants=contaminants,
                initial_concentrations={c: 100.0 for c in contaminants},
                reaction_time=180,
                lab_data_path=args.lab_data
            )
            
            if engine:
                print("✅ Simulation completed successfully!")
            else:
                print("❌ Simulation failed!")
                sys.exit(1)
        else:
            # Interactive simulation
            run_simulation_with_variant()
    
    else:
        # No specific action - show help and interactive menu
        parser.print_help()
        print("\n" + "=" * 50)
        print("INTERACTIVE MODE")
        print("=" * 50)
        
        while True:
            print("\nSelect an option:")
            print("1. List variants")
            print("2. Infer variant from XRD data")
            print("3. Create membrane with variant")
            print("4. Run simulation")
            print("5. Exit")
            
            choice = input("Choice (1-5): ").strip()
            
            if choice == '1':
                list_variants()
            elif choice == '2':
                infer_variant_interactive()
            elif choice == '3':
                create_membrane_with_variant()
            elif choice == '4':
                run_simulation_with_variant()
            elif choice == '5':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
