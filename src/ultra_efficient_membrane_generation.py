"""
Ultra-Efficient Membrane Generation with Advanced Scientific Computing

Implements batch vectorized membrane property calculations with:
- NumPy vectorization for parameter combinations
- Pre-allocated arrays for memory efficiency
- Tensor operations for property matrices
- Batch interpolation for flux/rejection lookup
- Optimized data structures
"""

import numpy as np
import pandas as pd
from itertools import product
from scipy.interpolate import RegularGridInterpolator
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Import the ultra-efficient simulators
try:
    from src.ultra_efficient_flux import ULTRA_FLUX_SIMULATOR
    from src.ultra_efficient_oil_rejection import ULTRA_REJECTION_SIMULATOR
    from src.flux_simulator import simulate_flux
    from src.oil_rejection import simulate_oil_rejection
    ADVANCED_SIMULATORS_AVAILABLE = True
except ImportError:
    # Fallback to standard simulators
    from flux_simulator import simulate_flux
    from oil_rejection import simulate_oil_rejection
    ADVANCED_SIMULATORS_AVAILABLE = False

from src.properties import MEMBRANE_TYPES, PRESSURE_RANGE
from src.membrane_model import Membrane

class UltraEfficientMembraneGenerator:
    """
    Ultra-efficient membrane generation using advanced scientific computing methods.
    """
    
    def __init__(self):
        self.membrane_types = MEMBRANE_TYPES
        self.pressure_range = PRESSURE_RANGE
        
        # Pre-compile parameter tensors for all combinations
        self._setup_parameter_tensors()
        
        # Setup fast lookup interpolators
        self._setup_property_interpolators()
    
    def _setup_parameter_tensors(self):
        """Pre-compile parameter tensors for all membrane combinations."""
        self.parameter_combinations = {}
        
        for mem_type, props in self.membrane_types.items():
            # Create all combinations of thickness and pore size
            thicknesses = np.array(props['thicknesses'])
            pore_sizes = np.array(props['pore_sizes'])
            
            # Create meshgrid for all combinations
            T_grid, P_grid = np.meshgrid(thicknesses, pore_sizes, indexing='ij')
            
            self.parameter_combinations[mem_type] = {
                'thickness_tensor': T_grid,
                'pore_size_tensor': P_grid,
                'combinations_shape': T_grid.shape,
                'total_combinations': T_grid.size
            }
    
    def _setup_property_interpolators(self):
        """Setup interpolators for fast property lookup."""
        self.flux_interpolators = {}
        self.rejection_interpolators = {}
        
        for mem_type, props in self.membrane_types.items():
            if 'flux_map' in props:
                # Create flux interpolator
                thicknesses = sorted(props['flux_map'].keys())
                flux_values = [props['flux_map'][t] for t in thicknesses]
                
                if len(thicknesses) > 1:
                    self.flux_interpolators[mem_type] = RegularGridInterpolator(
                        (thicknesses,), flux_values,
                        method='linear', bounds_error=False, fill_value=None
                    )
            
            if 'rejection_map' in props:
                # Create rejection interpolator
                pore_sizes = sorted(props['rejection_map'].keys())
                rejection_values = [props['rejection_map'][p] for p in pore_sizes]
                
                if len(pore_sizes) > 1:
                    self.rejection_interpolators[mem_type] = RegularGridInterpolator(
                        (pore_sizes,), rejection_values,
                        method='linear', bounds_error=False, fill_value=None
                    )
    
    def generate_membrane_variants_batch(self, membrane_types=None):
        """
        Generate all membrane variants using batch operations.
        
        Args:
            membrane_types (list, optional): Membrane types to generate
            
        Returns:
            dict: Batch generation results with pre-computed properties
        """
        if membrane_types is None:
            membrane_types = list(self.membrane_types.keys())
        
        batch_results = {}
        
        for mem_type in membrane_types:
            if mem_type not in self.parameter_combinations:
                continue
            
            param_data = self.parameter_combinations[mem_type]
            thickness_tensor = param_data['thickness_tensor']
            pore_size_tensor = param_data['pore_size_tensor']
            
            # Batch flux calculation if advanced simulators available
            if ADVANCED_SIMULATORS_AVAILABLE:
                flux_tensor = self._calculate_flux_batch_advanced(
                    thickness_tensor, pore_size_tensor, mem_type
                )
                rejection_tensor = self._calculate_rejection_batch_advanced(
                    pore_size_tensor, mem_type
                )
            else:
                flux_tensor = self._calculate_flux_batch_fallback(
                    thickness_tensor, pore_size_tensor, mem_type
                )
                rejection_tensor = self._calculate_rejection_batch_fallback(
                    pore_size_tensor, mem_type
                )
            
            # Create membrane name tensor
            name_tensor = np.array([
                [f"{mem_type}_T{t:.0f}_P{p:.1f}" 
                 for p in pore_size_tensor[i, :]]
                for i, t in enumerate(thickness_tensor[:, 0])
            ])
            
            batch_results[mem_type] = {
                'names': name_tensor,
                'thicknesses': thickness_tensor,
                'pore_sizes': pore_size_tensor,
                'flux_values': flux_tensor,
                'rejection_values': rejection_tensor,
                'combinations_shape': thickness_tensor.shape,
                'total_variants': thickness_tensor.size
            }
        
        return batch_results
    
    def _calculate_flux_batch_advanced(self, thickness_tensor, pore_size_tensor, mem_type):
        """Calculate flux using advanced ultra-efficient simulator."""
        # Use batch simulation for entire parameter space
        pressure = 1.0  # Standard pressure for variant generation
        # Pass unique 1D arrays, not flattened meshgrids
        thicknesses = np.unique(thickness_tensor)
        pore_sizes = np.unique(pore_size_tensor)
        flux_results = ULTRA_FLUX_SIMULATOR.simulate_flux_batch(
            pore_sizes,
            thicknesses,
            [pressure]
        )
        
        # Reshape to match parameter tensor shape
        # Transpose to (thickness, pore_size) before reshape
        return flux_results[:, :, 0].T.reshape(thickness_tensor.shape)
    
    def _calculate_rejection_batch_advanced(self, pore_size_tensor, mem_type):
        """Calculate rejection using advanced ultra-efficient simulator."""
        # Standard droplet size for rejection calculation
        droplet_size = 5.0  # μm
        contact_angle = 65.0  # degrees, typical for GO
        
        rejection_results = ULTRA_REJECTION_SIMULATOR.simulate_rejection_batch(
            pore_size_tensor.flatten(),
            [droplet_size],
            [contact_angle]
        )
        
        # Reshape to match parameter tensor shape
        return rejection_results[:, 0].reshape(pore_size_tensor.shape)
    
    def _calculate_flux_batch_fallback(self, thickness_tensor, pore_size_tensor, mem_type):
        """Fallback flux calculation using vectorized operations."""
        # Use interpolator if available, otherwise use properties
        flux_tensor = np.zeros_like(thickness_tensor)
        
        if mem_type in self.flux_interpolators:
            interpolator = self.flux_interpolators[mem_type]
            for i in range(thickness_tensor.shape[0]):
                flux_tensor[i, :] = interpolator(thickness_tensor[i, 0])
        else:
            # Use property mapping
            props = self.membrane_types[mem_type]
            if 'flux_map' in props:
                for i in range(thickness_tensor.shape[0]):
                    thickness = thickness_tensor[i, 0]
                    # Find closest thickness in flux_map
                    closest_t = min(props['flux_map'].keys(), 
                                  key=lambda x: abs(x - thickness))
                    flux_tensor[i, :] = props['flux_map'][closest_t]
        
        return flux_tensor
    
    def _calculate_rejection_batch_fallback(self, pore_size_tensor, mem_type):
        """Fallback rejection calculation using vectorized operations."""
        rejection_tensor = np.zeros_like(pore_size_tensor)
        
        if mem_type in self.rejection_interpolators:
            interpolator = self.rejection_interpolators[mem_type]
            for j in range(pore_size_tensor.shape[1]):
                rejection_tensor[:, j] = interpolator(pore_size_tensor[0, j])
        else:
            # Use property mapping
            props = self.membrane_types[mem_type]
            if 'rejection_map' in props:
                for j in range(pore_size_tensor.shape[1]):
                    pore_size = pore_size_tensor[0, j]
                    # Find closest pore size in rejection_map
                    closest_p = min(props['rejection_map'].keys(),
                                  key=lambda x: abs(x - pore_size))
                    rejection_tensor[:, j] = props['rejection_map'][closest_p]
        
        return rejection_tensor
    
    def create_membrane_objects_batch(self, batch_results):
        """
        Convert batch results to Membrane objects efficiently.
        
        Args:
            batch_results (dict): Results from generate_membrane_variants_batch
            
        Returns:
            list: List of Membrane objects
        """
        membranes = []
        
        for mem_type, results in batch_results.items():
            names = results['names']
            thicknesses = results['thicknesses']
            pore_sizes = results['pore_sizes']
            flux_values = results['flux_values']
            rejection_values = results['rejection_values']
            
            # Flatten arrays for iteration
            flat_names = names.flatten()
            flat_thicknesses = thicknesses.flatten()
            flat_pore_sizes = pore_sizes.flatten()
            flat_flux = flux_values.flatten()
            flat_rejection = rejection_values.flatten()
            
            # Create Membrane objects
            for i in range(len(flat_names)):
                membrane = Membrane(
                    name=flat_names[i],
                    pore_size_nm=flat_pore_sizes[i],
                    thickness_nm=flat_thicknesses[i],
                    flux_lmh=flat_flux[i],
                    rejection_percent=flat_rejection[i]
                )
                membranes.append(membrane)
        
        return membranes
    
    def performance_ranking_batch(self, batch_results, weights=None):
        """
        Batch performance ranking of membrane variants.
        
        Args:
            batch_results (dict): Results from generate_membrane_variants_batch
            weights (dict, optional): Weights for ranking criteria
            
        Returns:
            pd.DataFrame: Ranked performance results
        """
        if weights is None:
            weights = {'flux': 0.6, 'rejection': 0.4}
        
        ranking_data = []
        
        for mem_type, results in batch_results.items():
            # Flatten results for processing
            flat_names = results['names'].flatten()
            flat_flux = results['flux_values'].flatten()
            flat_rejection = results['rejection_values'].flatten()
            flat_thickness = results['thicknesses'].flatten()
            flat_pore_size = results['pore_sizes'].flatten()
            
            # Normalize metrics for ranking
            flux_normalized = flat_flux / np.max(flat_flux) if np.max(flat_flux) > 0 else flat_flux
            rejection_normalized = flat_rejection / 100.0  # Convert % to 0-1
            
            # Calculate performance scores
            performance_scores = (weights['flux'] * flux_normalized + 
                                weights['rejection'] * rejection_normalized)
            
            # Create ranking records
            for i in range(len(flat_names)):
                record = {
                    'membrane_name': flat_names[i],
                    'membrane_type': mem_type,
                    'thickness_nm': flat_thickness[i],
                    'pore_size_nm': flat_pore_size[i],
                    'flux_lmh': flat_flux[i],
                    'rejection_percent': flat_rejection[i],
                    'performance_score': performance_scores[i],
                    'flux_normalized': flux_normalized[i],
                    'rejection_normalized': rejection_normalized[i]
                }
                ranking_data.append(record)
        
        # Convert to DataFrame and sort by performance
        df = pd.DataFrame(ranking_data)
        df = df.sort_values('performance_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def parameter_space_analysis(self, membrane_types=None):
        """
        Comprehensive parameter space analysis with statistics.
        
        Args:
            membrane_types (list, optional): Membrane types to analyze
            
        Returns:
            dict: Comprehensive analysis results
        """
        batch_results = self.generate_membrane_variants_batch(membrane_types)
        ranking_df = self.performance_ranking_batch(batch_results)
        
        # Calculate statistics
        analysis = {
            'total_variants': len(ranking_df),
            'membrane_types': list(batch_results.keys()),
            'parameter_ranges': {},
            'performance_statistics': {},
            'top_performers': ranking_df.head(10).to_dict('records'),
            'batch_generation_time': 'N/A'  # Would be measured in actual run
        }
        
        # Parameter range analysis
        for col in ['thickness_nm', 'pore_size_nm', 'flux_lmh', 'rejection_percent']:
            analysis['parameter_ranges'][col] = {
                'min': ranking_df[col].min(),
                'max': ranking_df[col].max(),
                'mean': ranking_df[col].mean(),
                'std': ranking_df[col].std(),
                'median': ranking_df[col].median()
            }
        
        # Performance statistics by membrane type
        for mem_type in analysis['membrane_types']:
            type_data = ranking_df[ranking_df['membrane_type'] == mem_type]
            analysis['performance_statistics'][mem_type] = {
                'count': len(type_data),
                'mean_performance': type_data['performance_score'].mean(),
                'best_performance': type_data['performance_score'].max(),
                'mean_flux': type_data['flux_lmh'].mean(),
                'mean_rejection': type_data['rejection_percent'].mean()
            }
        
        return analysis

# Create global instance for efficient reuse
ULTRA_MEMBRANE_GENERATOR = UltraEfficientMembraneGenerator()

def generate_membrane_variants_ultra_fast(membrane_types=None):
    """
    Ultra-fast membrane variant generation with global generator instance.
    """
    batch_results = ULTRA_MEMBRANE_GENERATOR.generate_membrane_variants_batch(membrane_types)
    return ULTRA_MEMBRANE_GENERATOR.create_membrane_objects_batch(batch_results)

def membrane_performance_analysis_batch(membrane_types=None):
    """
    Efficient batch performance analysis of membrane variants.
    """
    return ULTRA_MEMBRANE_GENERATOR.parameter_space_analysis(membrane_types)

if __name__ == "__main__":
    # Performance demonstration
    import time
    
    generator = UltraEfficientMembraneGenerator()
    
    print("Testing ultra-efficient membrane generation...")
    
    start_time = time.time()
    batch_results = generator.generate_membrane_variants_batch()
    generation_time = time.time() - start_time
    
    print(f"Batch generation completed in {generation_time:.4f} seconds")
    
    # Count total variants
    total_variants = sum(results['total_variants'] for results in batch_results.values())
    print(f"Generated {total_variants:,} membrane variants")
    print(f"Generation rate: {total_variants/generation_time:,.0f} variants/second")
    
    # Performance analysis
    start_time = time.time()
    analysis = generator.parameter_space_analysis()
    analysis_time = time.time() - start_time
    
    print(f"Parameter space analysis completed in {analysis_time:.4f} seconds")
    print(f"Top performer: {analysis['top_performers'][0]['membrane_name']}")
    print(f"Performance score: {analysis['top_performers'][0]['performance_score']:.3f}")
    
    # Memory efficiency test
    start_time = time.time()
    membrane_objects = generator.create_membrane_objects_batch(batch_results)
    object_creation_time = time.time() - start_time
    
    print(f"Membrane object creation: {object_creation_time:.4f} seconds")
    print(f"Object creation rate: {len(membrane_objects)/object_creation_time:,.0f} objects/second")
