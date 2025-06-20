# hybrid_structure.py

"""
Phase 2: Constructs hybrid GO/rGO membrane configurations with physics-based modeling.

Scientific approach: Uses dynamic interlayer spacing, physics-based flux/rejection models,
and prepares layer metadata for Phase 3 LAMMPS simulations.
"""

import numpy as np
import os
from src.properties import MEMBRANE_TYPES

def compute_interlayer_spacing(layer1, layer2):
    """
    Compute interlayer spacing between two adjacent layers based on their types.
    
    Args:
        layer1 (str): First layer type ('GO' or 'rGO')
        layer2 (str): Second layer type ('GO' or 'rGO')
    
    Returns:
        float: Interlayer spacing in nm
    
    Scientific basis:
        - GO–GO = 1.05 nm (hydrated, with intercalated water)
        - rGO–rGO = 0.34 nm (π-π stacking distance)
        - GO–rGO = 0.80 nm (hybrid interface, intermediate spacing)
    """
    spacing_map = {
        ('GO', 'GO'): 1.05,    # Hydrated GO layers
        ('rGO', 'rGO'): 0.34,  # π-π stacking
        ('GO', 'rGO'): 0.80,   # Hybrid interface
        ('rGO', 'GO'): 0.80    # Hybrid interface (symmetric)
    }
    
    return spacing_map.get((layer1, layer2), 0.70)  # Default average spacing

class HybridStructure:
    """
    Represents the structural configuration of a hybrid GO/rGO membrane with physics-based properties.
    
    Attributes:
        layers (list): List of layer types ['GO', 'rGO', 'GO', ...]
        total_thickness (float): Total membrane thickness in nm (dynamically calculated)
        stacking_sequence (str): Description of the stacking pattern
        layer_metadata (list): List of dicts with layer positioning and spacing info
    """
    
    def __init__(self, layers):
        self.layers = layers
        self.stacking_sequence = "-".join(layers)
        
        # Calculate dynamic thickness and create layer metadata
        self.total_thickness, self.layer_metadata = self._calculate_structure_properties()
    
    def _calculate_structure_properties(self):
        """
        Calculate total thickness and layer metadata using physics-based interlayer spacing.
        
        Returns:
            tuple: (total_thickness, layer_metadata)
        """
        if len(self.layers) == 0:
            return 0.0, []
        
        if len(self.layers) == 1:
            # Single layer: use default layer thickness
            layer_thickness = 0.34  # Single graphene/GO sheet thickness
            metadata = [{
                'layer_type': self.layers[0],
                'layer_index': 0,
                'z_start': 0.0,
                'z_end': layer_thickness,
                'thickness': layer_thickness,
                'spacing_above': 0.0
            }]
            return layer_thickness, metadata
        
        # Multi-layer structure
        metadata = []
        current_z = 0.0
        layer_thickness = 0.34  # Individual sheet thickness
        
        for i, layer in enumerate(self.layers):
            z_start = current_z
            z_end = z_start + layer_thickness
            
            # Calculate spacing above this layer (to next layer)
            spacing_above = 0.0
            if i < len(self.layers) - 1:
                next_layer = self.layers[i + 1]
                spacing_above = compute_interlayer_spacing(layer, next_layer)
            
            metadata.append({
                'layer_type': layer,
                'layer_index': i,
                'z_start': z_start,
                'z_end': z_end,
                'thickness': layer_thickness,
                'spacing_above': spacing_above
            })
            
            current_z = z_end + spacing_above
        
        total_thickness = current_z
        return total_thickness, metadata
    
    def to_dict(self):
        """
        Convert HybridStructure to dictionary for serialization.
        
        Returns:
            dict: Serializable structure data
        """
        return {
            'layers': self.layers,
            'stacking_sequence': self.stacking_sequence,
            'total_thickness': self.total_thickness,
            'layer_metadata': self.layer_metadata,
            'num_layers': len(self.layers),
            'go_fraction': self.layers.count('GO') / len(self.layers) if self.layers else 0,
            'rgo_fraction': self.layers.count('rGO') / len(self.layers) if self.layers else 0
        }
    
    def __repr__(self):
        return f"<HybridStructure: {self.stacking_sequence}, {self.total_thickness:.2f} nm>"
    
    def calculate_effective_flux(self, base_flux):
        """
        Calculate effective flux with interface penalty.
        
        Args:
            base_flux (float): Theoretical base flux (LMH)
        
        Returns:
            float: Penalized effective flux
        """
        num_interfaces = sum(1 for i in range(1, len(self.layers)) if self.layers[i] != self.layers[i-1])
        interface_penalty = 0.97 ** num_interfaces  # 3% loss per interface
        return base_flux * interface_penalty

def create_alternating_structure(num_layers=6, start_with='GO'):
    """
    Create an alternating GO/rGO structure.
    
    Args:
        num_layers (int): Total number of layers
        start_with (str): Starting layer type ('GO' or 'rGO')
    
    Returns:
        HybridStructure: Configured hybrid structure
    """
    layers = []
    current = start_with
    
    for i in range(num_layers):
        layers.append(current)
        current = 'rGO' if current == 'GO' else 'GO'
    
    return HybridStructure(layers)

def create_sandwich_structure(core_type='rGO', shell_type='GO', core_layers=4, shell_layers=1):
    """
    Create a sandwich structure with core surrounded by shell layers.
    
    Args:
        core_type (str): Material for core layers
        shell_type (str): Material for shell layers
        core_layers (int): Number of core layers
        shell_layers (int): Number of shell layers on each side
    
    Returns:
        HybridStructure: Configured hybrid structure
    """
    layers = [shell_type] * shell_layers + [core_type] * core_layers + [shell_type] * shell_layers
    return HybridStructure(layers)

def optimize_structure_for_flux(target_flux):
    """
    Suggest optimal hybrid structure configuration for target flux.
    
    Args:
        target_flux (float): Target water flux in L·m⁻²·h⁻¹
    
    Returns:
        HybridStructure: Optimized structure
    """
    # Simple heuristic: more rGO for higher flux
    if target_flux > 5000:
        return create_sandwich_structure(core_type='rGO', shell_type='GO', core_layers=6, shell_layers=1)
    elif target_flux > 1000:
        return create_alternating_structure(num_layers=6, start_with='rGO')
    else:
        return create_sandwich_structure(core_type='GO', shell_type='rGO', core_layers=4, shell_layers=2)

def run_phase2_analysis(target_flux=None, target_rejection=None):
    """
    Run Phase 2 structural design and optimization.
    
    Args:
        target_flux (float): Target water flux (L·m⁻²·h⁻¹)
        target_rejection (float): Target oil rejection (%)
    
    Returns:
        dict: Phase 2 results with optimized structures
    """
    print("\n" + "="*50)
    print("PHASE 2: HYBRID STRUCTURE DESIGN")
    print("="*50)
    
    # Generate various hybrid configurations
    structures = []
    
    # 1. Alternating structures
    for num_layers in [4, 6, 8, 10]:
        for start_material in ['GO', 'rGO']:
            struct = create_alternating_structure(num_layers, start_material)
            struct.name = f"Alt_{num_layers}L_{start_material}"
            structures.append(struct)
    
    # 2. Sandwich structures
    for core_type in ['GO', 'rGO']:
        shell_type = 'rGO' if core_type == 'GO' else 'GO'
        for core_layers in [2, 4, 6]:
            struct = create_sandwich_structure(core_type, shell_type, core_layers, 1)
            struct.name = f"Sand_{core_type}core_{core_layers}L"
            structures.append(struct)
    
    # 3. Target-optimized structures
    if target_flux:
        opt_struct = optimize_structure_for_flux(target_flux)
        opt_struct.name = f"Opt_Flux_{target_flux}"
        structures.append(opt_struct)
    
    # Predict properties for each structure
    structure_results = []
    for struct in structures:
        result = predict_hybrid_properties(struct)
        result['structure'] = struct
        structure_results.append(result)
    
    # Rank structures by performance
    if target_flux and target_rejection:
        ranked_results = rank_structures_by_targets(structure_results, target_flux, target_rejection)
    else:
        ranked_results = rank_structures_by_performance(structure_results)
    
    # Generate Phase 2 visualizations
    phase2_output_dir = os.path.join(os.getcwd(), 'output', 'phase2_structures')
    visualize_structures(ranked_results[:5], phase2_output_dir)  # Top 5 structures
    
    print(f"\nPhase 2 Complete: {len(structures)} structures analyzed")
    print(f"Top structure: {ranked_results[0]['structure'].name}")
    print(f"Predicted flux: {ranked_results[0]['predicted_flux']:.1f} L·m⁻²·h⁻¹")
    print(f"Predicted rejection: {ranked_results[0]['predicted_rejection']:.1f}%")
    
    return {
        'all_structures': structure_results,
        'top_structures': ranked_results[:5],
        'output_dir': phase2_output_dir
    }

def predict_hybrid_properties(structure):
    """
    Predict membrane properties based on hybrid structure using physics-based Phase 1 models.
    
    Args:
        structure (HybridStructure): The hybrid structure to analyze
    
    Returns:
        dict: Predicted properties with error estimates
    
    Scientific basis:
        - Uses Hagen-Poiseuille flux model from flux_simulator.py
        - Uses size exclusion + wettability rejection model from oil_rejection.py
        - Incorporates dynamic thickness from interlayer spacing calculations    """
    
    from src.flux_simulator import simulate_flux
    from src.properties import WATER_PROPERTIES, OIL_DROPLET_SIZE
    from src.oil_rejection import simulate_oil_rejection
    from src.membrane_model import compute_interface_penalty
      # Calculate composition fractions
    if not structure.layers:
        # Handle empty structure
        return {
            'structure_name': getattr(structure, 'name', 'Empty'),
            'go_fraction': 0,
            'rgo_fraction': 0,
            'total_layers': 0,
            'thickness_nm': 0,
            'avg_pore_size': 0,
            'contact_angle_deg': 0,
            'predicted_flux': 0,
            'predicted_rejection': 0,
            'flux_error': 0,
            'rejection_error': 0,
            'weighted_modulus': 0,
            'tensile_strength_MPa': 0,
            'performance_score': 0,
            'layer_metadata': []
        }
    
    go_fraction = structure.layers.count('GO') / len(structure.layers)
    rgo_fraction = structure.layers.count('rGO') / len(structure.layers)
    
    # Physics-based property calculations
    # 1. Average pore size (weighted by layer fraction)
    go_pore_size = 1.2   # nm (from properties.py)
    rgo_pore_size = 0.99 # nm (from properties.py)
    avg_pore_size = go_fraction * go_pore_size + rgo_fraction * rgo_pore_size
    
    # Handle edge case where avg_pore_size is 0
    if avg_pore_size <= 0:
        avg_pore_size = 1.0  # Default pore size
    
    # 2. Use dynamic thickness from structure
    thickness = structure.total_thickness
    
    # Handle edge case where thickness is 0
    if thickness <= 0:
        thickness = 1.0  # Default thickness
      # 3. Physics-based flux prediction with advanced parameters
    predicted_flux = simulate_flux(
        pore_size_nm=avg_pore_size,
        thickness_nm=thickness,
        pressure_bar=1.0,
        viscosity_pas=WATER_PROPERTIES["viscosity_25C"],
        porosity=WATER_PROPERTIES["porosity"],
        tortuosity=WATER_PROPERTIES["tortuosity"]
    )
    
    # 4. Weighted contact angle
    go_contact_angle = 29.5  # degrees (from properties.py)
    rgo_contact_angle = 73.9 # degrees (from properties.py)
    contact_angle = go_fraction * go_contact_angle + rgo_fraction * rgo_contact_angle
    
    # 5. Physics-based oil rejection prediction
    predicted_rejection = simulate_oil_rejection(
        pore_size_nm=avg_pore_size,
        droplet_size_um=OIL_DROPLET_SIZE,
        contact_angle_deg=contact_angle
    )
    
    # 6. Weighted mechanical properties
    go_props = MEMBRANE_TYPES['GO']
    rgo_props = MEMBRANE_TYPES['rGO']
    
    weighted_modulus = (go_fraction * go_props['youngs_modulus_GPa'] + 
                       rgo_fraction * rgo_props['youngs_modulus_GPa'])
    weighted_tensile_strength = (go_fraction * go_props['tensile_strength_MPa'] + 
                                rgo_fraction * rgo_props['tensile_strength_MPa'])
      # 7. Interface penalty for hybrid structures
    interface_penalty = compute_interface_penalty(structure.layers)
    predicted_flux *= (1 - interface_penalty)  # Apply penalty to flux
    
    # 8. Error estimates (±5% flux, ±3% rejection)
    flux_error = predicted_flux * 0.05
    rejection_error = predicted_rejection * 0.03
    
    # 9. Performance score
    performance_score = predicted_flux * (predicted_rejection / 100)
    
    return {
        'structure_name': getattr(structure, 'name', structure.stacking_sequence),
        'go_fraction': go_fraction,
        'rgo_fraction': rgo_fraction,
        'total_layers': len(structure.layers),
        'thickness_nm': thickness,
        'avg_pore_size': avg_pore_size,
        'contact_angle_deg': contact_angle,
        'predicted_flux': predicted_flux,
        'predicted_rejection': predicted_rejection,
        'flux_error': flux_error,
        'rejection_error': rejection_error,        'weighted_modulus': weighted_modulus,
        'tensile_strength_MPa': weighted_tensile_strength,
        'interface_penalty': interface_penalty,
        'performance_score': performance_score,
        'layer_metadata': structure.layer_metadata
    }

def rank_structures_by_performance(structure_results):
    """Rank structures by overall performance score."""
    return sorted(structure_results, key=lambda x: x['performance_score'], reverse=True)

def rank_structures_by_targets(structure_results, target_flux, target_rejection):
    """Rank structures by how close they are to targets."""
    def target_score(result):
        flux_error = abs(result['predicted_flux'] - target_flux) / target_flux
        rejection_error = abs(result['predicted_rejection'] - target_rejection) / target_rejection
        return 1 / (1 + flux_error + rejection_error)  # Higher is better
    
    for result in structure_results:
        result['target_score'] = target_score(result)
    
    return sorted(structure_results, key=lambda x: x['target_score'], reverse=True)

def visualize_structures(top_structures, output_dir):
    """
    Create visualizations of the top hybrid structures.
    """
    import matplotlib.pyplot as plt
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Structure comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Top 5 Hybrid Membrane Structures - Phase 2 Analysis', fontsize=16, fontweight='bold')
    
    for i, result in enumerate(top_structures[:5]):
        if i >= 5:
            break
        
        row = i // 3
        col = i % 3
        ax = axes[row, col] if len(top_structures) > 3 else axes[col]
        
        structure = result['structure']
        
        # Create a simple layer visualization
        layer_colors = {'GO': '#1f77b4', 'rGO': '#2ca02c'}
        y_positions = np.arange(len(structure.layers))
        colors = [layer_colors[layer] for layer in structure.layers]
        
        bars = ax.barh(y_positions, [1] * len(structure.layers), color=colors, alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(structure.layers)
        ax.set_xlabel('Layer Thickness (relative)')
        ax.set_title(f"{result['structure_name']}\n"
                    f"Thickness: {result['thickness_nm']:.2f} nm\n"
                    f"Flux: {result['predicted_flux']:.0f} ± {result['flux_error']:.0f} L·m⁻²·h⁻¹\n"
                    f"Rejection: {result['predicted_rejection']:.1f} ± {result['rejection_error']:.1f}%")
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(top_structures), 6):
        row = i // 3
        col = i % 3
        if row < 2 and col < 3:
            fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_structures_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance comparison chart
    plt.figure(figsize=(12, 8))
    
    names = [r['structure_name'] for r in top_structures]
    fluxes = [r['predicted_flux'] for r in top_structures]
    rejections = [r['predicted_rejection'] for r in top_structures]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Flux bars
    bars1 = ax1.bar(x - width/2, fluxes, width, label='Flux (L·m⁻²·h⁻¹)', 
                    color='#1f77b4', alpha=0.7)
    ax1.set_xlabel('Structure')
    ax1.set_ylabel('Water Flux (L·m⁻²·h⁻¹)', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    
    # Rejection bars on secondary y-axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, rejections, width, label='Rejection (%)', 
                    color='#ff7f0e', alpha=0.7)
    ax2.set_ylabel('Oil Rejection (%)', color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_title('Performance Comparison - Top Hybrid Structures')
    
    # Add value labels on bars
    for bar, flux in zip(bars1, fluxes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(fluxes)*0.01,
                f'{flux:.0f}', ha='center', va='bottom', fontsize=9)
    
    for bar, rejection in zip(bars2, rejections):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rejection:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Phase 2 visualizations saved to: {output_dir}")
