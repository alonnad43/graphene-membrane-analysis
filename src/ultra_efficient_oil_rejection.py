"""
Ultra-Efficient Oil Rejection Simulator with Advanced Scientific Computing

Implements batch vectorized sigmoid modeling with:
- NumPy vectorization for parameter sweeps
- JIT compilation for sigmoid calculations  
- Pre-compiled droplet size distributions
- Batch wettability modeling
- Memory-efficient tensor operations
"""

import numpy as np
import pandas as pd
from scipy.special import expit  # Optimized sigmoid function
from scipy.interpolate import RegularGridInterpolator
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Pre-compiled constants for maximum efficiency
SIGMOID_PARAMS = {
    'alpha_default': 2.5,  # Size exclusion parameter
    'beta_default': 0.1,   # Wettability parameter
    'baseline_rejection': 5.0,  # Minimum rejection %
    'max_rejection': 99.0  # Maximum rejection %
}

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    
    @jit(nopython=True, cache=True)
    def batch_sigmoid_kernel(size_ratios, contact_angles, alpha, beta):
        """
        Ultra-optimized JIT kernel for batch sigmoid calculations.
        """
        # Avoid log of values <= 1.0 to prevent numerical issues
        log_ratios = np.where(size_ratios > 1.0, np.log(size_ratios), 0.0)
        
        # Sigmoid calculation: 1 / (1 + exp(-(α * log(ratio) + β * (90 - θ))))
        exponents = -(alpha * log_ratios + beta * (90.0 - contact_angles))
        
        # Use stable sigmoid implementation
        rejections = np.where(
            exponents > 0,
            1.0 / (1.0 + np.exp(-exponents)),
            np.exp(exponents) / (1.0 + np.exp(exponents))
        )
        
        # Handle small size ratios with baseline rejection
        rejections = np.where(
            size_ratios <= 1.0,
            SIGMOID_PARAMS['baseline_rejection'] / 100.0,
            rejections
        )
        
        return rejections * 100.0  # Convert to percentage
        
except ImportError:
    NUMBA_AVAILABLE = False
    
    def batch_sigmoid_kernel(size_ratios, contact_angles, alpha, beta):
        """
        NumPy fallback for sigmoid calculations when Numba not available.
        """
        log_ratios = np.where(size_ratios > 1.0, np.log(size_ratios), 0.0)
        exponents = -(alpha * log_ratios + beta * (90.0 - contact_angles))
        
        # Use scipy's optimized sigmoid (expit)
        rejections = expit(exponents)
        
        rejections = np.where(
            size_ratios <= 1.0,
            SIGMOID_PARAMS['baseline_rejection'] / 100.0,
            rejections
        )
        
        return np.clip(rejections * 100.0, 0.0, SIGMOID_PARAMS['max_rejection'])

class UltraEfficientOilRejectionSimulator:
    """
    Ultra-efficient oil rejection simulator using advanced scientific computing.
    """
    
    def __init__(self):
        self.default_params = {
            'alpha': SIGMOID_PARAMS['alpha_default'],
            'beta': SIGMOID_PARAMS['beta_default'],
            'contact_angle': 65.0  # Typical GO contact angle
        }
        
        # Pre-compile lookup tables for common scenarios
        self._setup_lookup_tables()
    
    def _setup_lookup_tables(self):
        """Setup pre-compiled lookup tables for fast calculations."""
        # Common ranges for interpolation
        size_ratio_range = np.logspace(-1, 3, 200)  # 0.1 to 1000
        contact_angle_range = np.linspace(10, 150, 50)  # 10° to 150°
        
        # Pre-compute rejection matrix
        SR_grid, CA_grid = np.meshgrid(size_ratio_range, contact_angle_range, indexing='ij')
        
        rejection_matrix = batch_sigmoid_kernel(
            SR_grid, CA_grid, 
            self.default_params['alpha'], 
            self.default_params['beta']
        )
        
        self.rejection_interpolator = RegularGridInterpolator(
            (size_ratio_range, contact_angle_range), rejection_matrix,
            method='linear', bounds_error=False, fill_value=None
        )
    
    def simulate_rejection_batch(self, pore_sizes_nm, droplet_sizes_um, 
                                contact_angles_deg=None, alpha=None, beta=None):
        """
        Ultra-efficient batch oil rejection simulation.
        
        Args:
            pore_sizes_nm (array-like): Pore sizes in nm
            droplet_sizes_um (array-like): Oil droplet sizes in μm
            contact_angles_deg (array-like, optional): Contact angles in degrees
            alpha (float, optional): Size exclusion parameter
            beta (float, optional): Wettability parameter
            
        Returns:
            np.ndarray: Rejection efficiencies in %
        """
        # Convert to numpy arrays
        pore_sizes = np.atleast_1d(np.array(pore_sizes_nm))
        droplet_sizes = np.atleast_1d(np.array(droplet_sizes_um))
        
        # Create meshgrids for full parameter space
        P_grid, D_grid = np.meshgrid(pore_sizes, droplet_sizes, indexing='ij')
        
        # Calculate size ratios (convert droplet μm to nm)
        size_ratios = (D_grid * 1000.0) / P_grid
        
        # Handle optional parameters
        if contact_angles_deg is None:
            contact_angles = np.full_like(size_ratios, self.default_params['contact_angle'])
        else:
            contact_angles_deg = np.atleast_1d(np.array(contact_angles_deg))
            contact_angles = np.broadcast_to(contact_angles_deg, size_ratios.shape)
        
        if alpha is None:
            alpha = self.default_params['alpha']
        if beta is None:
            beta = self.default_params['beta']
        
        # Batch rejection calculation using optimized kernel
        rejection_values = batch_sigmoid_kernel(
            size_ratios.flatten(), 
            contact_angles.flatten(), 
            alpha, beta
        )
        
        return rejection_values.reshape(size_ratios.shape)
    
    def membrane_screening_batch(self, membrane_configs, oil_properties):
        """
        Batch screening of multiple membrane configurations against oil types.
        
        Args:
            membrane_configs (list): List of membrane configuration dicts
            oil_properties (list): List of oil property dicts
            
        Returns:
            pd.DataFrame: Screening results matrix
        """
        results = []
        
        for i, membrane in enumerate(membrane_configs):
            for j, oil in enumerate(oil_properties):
                # Batch calculation for this membrane-oil combination
                rejection_matrix = self.simulate_rejection_batch(
                    membrane['pore_sizes'],
                    oil['droplet_sizes'],
                    oil.get('contact_angles', None)
                )
                
                # Summary statistics
                result = {
                    'membrane_id': i,
                    'membrane_name': membrane.get('name', f'Membrane_{i}'),
                    'oil_id': j,
                    'oil_type': oil.get('type', f'Oil_{j}'),
                    'mean_rejection': np.mean(rejection_matrix),
                    'max_rejection': np.max(rejection_matrix),
                    'min_rejection': np.min(rejection_matrix),
                    'rejection_std': np.std(rejection_matrix),
                    'efficiency_score': np.mean(rejection_matrix) * np.min(rejection_matrix) / 100.0  # Combined metric
                }
                results.append(result)
        
        return pd.DataFrame(results)
    
    def optimization_study(self, target_rejection, pore_range, droplet_range, 
                          contact_angle_range=None):
        """
        Efficient optimization study to find optimal parameters for target rejection.
        
        Args:
            target_rejection (float): Target rejection efficiency %
            pore_range (tuple): (min_pore, max_pore) in nm
            droplet_range (tuple): (min_droplet, max_droplet) in μm
            contact_angle_range (tuple, optional): (min_angle, max_angle) in degrees
            
        Returns:
            dict: Optimization results with parameter combinations
        """
        # Create parameter arrays
        pore_sizes = np.linspace(pore_range[0], pore_range[1], 50)
        droplet_sizes = np.linspace(droplet_range[0], droplet_range[1], 40)
        
        if contact_angle_range is None:
            contact_angles = [self.default_params['contact_angle']]
        else:
            contact_angles = np.linspace(contact_angle_range[0], contact_angle_range[1], 20)
        
        # Batch simulation
        rejection_results = self.simulate_rejection_batch(
            pore_sizes, droplet_sizes, contact_angles
        )
        
        # Find combinations close to target
        target_mask = np.abs(rejection_results - target_rejection) < 5.0  # ±5% tolerance
        
        # Extract optimal combinations
        if np.any(target_mask):
            optimal_indices = np.where(target_mask)
            optimal_combinations = []
            
            for idx in zip(*optimal_indices):
                if len(idx) == 3:  # pore, droplet, contact_angle
                    combo = {
                        'pore_size_nm': pore_sizes[idx[0]],
                        'droplet_size_um': droplet_sizes[idx[1]], 
                        'contact_angle_deg': contact_angles[idx[2]],
                        'predicted_rejection': rejection_results[idx]
                    }
                elif len(idx) == 2:  # pore, droplet only
                    combo = {
                        'pore_size_nm': pore_sizes[idx[0]],
                        'droplet_size_um': droplet_sizes[idx[1]],
                        'contact_angle_deg': self.default_params['contact_angle'],
                        'predicted_rejection': rejection_results[idx]
                    }
                optimal_combinations.append(combo)
        else:
            optimal_combinations = []
        
        return {
            'target_rejection': target_rejection,
            'total_combinations': rejection_results.size,
            'optimal_combinations': optimal_combinations,
            'rejection_statistics': {
                'mean': np.mean(rejection_results),
                'std': np.std(rejection_results),
                'min': np.min(rejection_results),
                'max': np.max(rejection_results)
            }
        }
    
    @lru_cache(maxsize=1000)
    def fast_single_rejection(self, pore_size_nm, droplet_size_um, contact_angle_deg=None):
        """
        Fast single rejection calculation with LRU caching.
        """
        if contact_angle_deg is None:
            contact_angle_deg = self.default_params['contact_angle']
            
        size_ratio = (droplet_size_um * 1000.0) / pore_size_nm
        
        if size_ratio <= 1.0:
            return SIGMOID_PARAMS['baseline_rejection']
        
        # Use interpolator for fast lookup
        try:
            return self.rejection_interpolator([size_ratio, contact_angle_deg])[0]
        except:
            # Fallback to direct calculation
            return batch_sigmoid_kernel(
                np.array([size_ratio]), 
                np.array([contact_angle_deg]),
                self.default_params['alpha'],
                self.default_params['beta']
            )[0]

# Create global instance for efficient reuse
ULTRA_REJECTION_SIMULATOR = UltraEfficientOilRejectionSimulator()

def simulate_oil_rejection_ultra_fast(pore_size_nm, droplet_size_um, contact_angle_deg=65.0):
    """
    Ultra-fast single rejection calculation with global simulator instance.
    Maintains backward compatibility while using optimized backend. 
    """
    return ULTRA_REJECTION_SIMULATOR.fast_single_rejection(
        pore_size_nm, droplet_size_um, contact_angle_deg
    )

def membrane_oil_screening_batch(membrane_configs, oil_configs):
    """
    Efficient batch screening of membrane-oil combinations.
    """
    return ULTRA_REJECTION_SIMULATOR.membrane_screening_batch(
        membrane_configs, oil_configs
    )

if __name__ == "__main__":
    # Performance demonstration
    import time
    
    simulator = UltraEfficientOilRejectionSimulator()
    
    # Large parameter space test
    pore_sizes = np.linspace(5, 200, 100)  # 5-200 nm
    droplet_sizes = np.logspace(-1, 2, 80)  # 0.1-100 μm
    contact_angles = np.linspace(30, 120, 60)  # 30-120°
    
    print(f"Testing {len(pore_sizes)}×{len(droplet_sizes)}×{len(contact_angles)} = {len(pore_sizes)*len(droplet_sizes)*len(contact_angles):,} combinations")
    
    start_time = time.time()
    results = simulator.simulate_rejection_batch(pore_sizes, droplet_sizes, contact_angles)
    end_time = time.time()
    
    print(f"Batch calculation completed in {end_time-start_time:.4f} seconds")
    print(f"Results shape: {results.shape}")
    print(f"Performance: {results.size/(end_time-start_time):,.0f} calculations/second") 
    print(f"Mean rejection: {np.mean(results):.2f}%")
    print(f"Rejection range: {np.min(results):.2f}% - {np.max(results):.2f}%")
    
    # Test optimization study
    print("\nOptimization study for 90% rejection:")
    opt_results = simulator.optimization_study(
        target_rejection=90.0,
        pore_range=(10, 50),
        droplet_range=(1, 10)
    )
    print(f"Found {len(opt_results['optimal_combinations'])} optimal combinations")
