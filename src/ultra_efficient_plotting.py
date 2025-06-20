"""
Ultra-Efficient Plotting Module with Advanced Scientific Visualization

Implements optimized plotting with:
- Pre-computed data matrices for fast rendering
- Vectorized data aggregation and grouping
- Memory-efficient plot generation
- Batch figure creation and export
- Advanced scientific visualization techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import os
from datetime import datetime
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Pre-defined color schemes for consistency
MEMBRANE_COLORS = {
    'GO': '#FF6B6B',      # Red
    'rGO': '#4ECDC4',     # Teal  
    'hybrid': '#45B7D1'   # Blue
}

CONTAMINANT_MARKERS = {
    'heavy_metal': 'o',
    'bacteria': 's', 
    'salt': '^',
    'organic_dye': 'D',
    'organic_pollutant': 'v',
    'virus': 'P',
    'inorganic_anion': 'h'
}

class UltraEfficientPlotter:
    """
    Ultra-efficient plotting class using advanced scientific visualization methods.
    """
    
    def __init__(self, output_dir=r"C:\Users\ramaa\Documents\graphene_mebraine\output\plots\ultra_efficient"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Pre-compile plot templates and configurations
        self.plot_configs = self._setup_plot_configurations()
        
        # Setup efficient data caching
        self.data_cache = {}
    
    def _setup_plot_configurations(self):
        """Setup pre-compiled plot configurations for different plot types."""
        return {
            'time_series': {
                'figsize': (12, 8),
                'dpi': 300,
                'linewidth': 2.5,
                'markersize': 6,
                'alpha': 0.8
            },
            'heatmap': {
                'figsize': (10, 8),
                'dpi': 300,
                'cmap': 'viridis',
                'annot': True,
                'fmt': '.2f'
            },
            'comparison': {
                'figsize': (14, 10),
                'dpi': 300,
                'width': 0.35,
                'alpha': 0.8
            },
            'matrix': {
                'figsize': (16, 12),
                'dpi': 300,
                'subplot_spacing': 0.3
            }
        }
    
    @lru_cache(maxsize=100)
    def _prepare_chemistry_data_fast(self, data_hash):
        """Fast chemistry data preparation with LRU caching."""
        # This would be called with a hash of the data for caching
        pass
    
    def plot_batch_chemistry_results(self, results_list, save_plots=True):
        """
        Ultra-efficient batch plotting of chemistry results.
        
        Args:
            results_list (list): List of simulation results
            save_plots (bool): Whether to save plots
            
        Returns:
            dict: Dictionary of created figures
        """
        if not results_list:
            return {}
        
        # Pre-process data into efficient structures
        processed_data = self._preprocess_chemistry_data_batch(results_list)
        
        figures = {}
        
        # Batch time series plots
        if processed_data['time_series_data']:
            figures['time_series'] = self._create_time_series_plots_batch(
                processed_data['time_series_data']
            )
        
        # Batch comparison plots
        if processed_data['comparison_data']:
            figures['comparison'] = self._create_comparison_plots_batch(
                processed_data['comparison_data']
            )
        
        # Batch heatmap plots
        if processed_data['matrix_data']:
            figures['heatmaps'] = self._create_heatmap_plots_batch(
                processed_data['matrix_data']
            )
        
        # Save all plots if requested
        if save_plots:
            self._save_figures_batch(figures)
        
        return figures
    
    def _preprocess_chemistry_data_batch(self, results_list):
        """
        Efficiently preprocess chemistry results for batch plotting.
        
        Args:
            results_list (list): Raw simulation results
            
        Returns:
            dict: Preprocessed data structures optimized for plotting
        """
        # Initialize data structures
        time_series_data = {}
        comparison_data = {}
        matrix_data = {}
        
        # Get all unique contaminants and membranes
        all_contaminants = set()
        all_membranes = set()
        
        for result in results_list:
            all_membranes.add(result.get('membrane_type', 'Unknown'))
            if 'contaminants' in result:
                all_contaminants.update(result['contaminants'].keys())
        
        all_contaminants = sorted(list(all_contaminants))
        all_membranes = sorted(list(all_membranes))
        
        # Pre-allocate matrices for efficient data aggregation
        n_membranes = len(all_membranes)
        n_contaminants = len(all_contaminants)
        
        # Create efficient data structures
        for contaminant in all_contaminants:
            time_series_data[contaminant] = {
                'membrane_types': [],
                'time_arrays': [],
                'concentration_arrays': [],
                'colors': [],
                'labels': []
            }
            
            comparison_data[contaminant] = {
                'membrane_types': all_membranes,
                'initial_concentrations': np.zeros(n_membranes),
                'final_concentrations': np.zeros(n_membranes),
                'removal_efficiencies': np.zeros(n_membranes),
                'membrane_indices': {mem: i for i, mem in enumerate(all_membranes)}
            }
        
        # Initialize matrix for heatmap data
        efficiency_matrix = np.zeros((n_contaminants, n_membranes))
        flux_matrix = np.zeros((n_contaminants, n_membranes))
        
        # Efficiently populate data structures
        for result in results_list:
            membrane_type = result.get('membrane_type', 'Unknown')
            if membrane_type not in all_membranes:
                continue
                
            mem_idx = all_membranes.index(membrane_type)
            time_points = result.get('time_min', [])
            
            for contaminant in all_contaminants:
                if contaminant not in result.get('contaminants', {}):
                    continue
                
                cont_idx = all_contaminants.index(contaminant)
                contaminant_data = result['contaminants'][contaminant]
                
                # Time series data
                if 'concentration_mg_L' in contaminant_data:
                    concentrations = contaminant_data['concentration_mg_L']
                    
                    time_series_data[contaminant]['membrane_types'].append(membrane_type)
                    time_series_data[contaminant]['time_arrays'].append(np.array(time_points))
                    time_series_data[contaminant]['concentration_arrays'].append(np.array(concentrations))
                    time_series_data[contaminant]['colors'].append(
                        MEMBRANE_COLORS.get(membrane_type, '#666666')
                    )
                    time_series_data[contaminant]['labels'].append(
                        f"{membrane_type} (Final: {concentrations[-1]:.2f} mg/L)"
                    )
                    
                    # Comparison data
                    comp_data = comparison_data[contaminant]
                    comp_data['initial_concentrations'][mem_idx] = concentrations[0]
                    comp_data['final_concentrations'][mem_idx] = concentrations[-1]
                    comp_data['removal_efficiencies'][mem_idx] = (
                        (concentrations[0] - concentrations[-1]) / concentrations[0] * 100
                        if concentrations[0] > 0 else 0
                    )
                    
                    # Matrix data
                    efficiency_matrix[cont_idx, mem_idx] = comp_data['removal_efficiencies'][mem_idx]
        
        # Store matrix data
        matrix_data['efficiency_matrix'] = efficiency_matrix
        matrix_data['flux_matrix'] = flux_matrix  # Would be populated if available
        matrix_data['contaminant_labels'] = all_contaminants
        matrix_data['membrane_labels'] = all_membranes
        
        return {
            'time_series_data': time_series_data,
            'comparison_data': comparison_data,
            'matrix_data': matrix_data,
            'metadata': {
                'n_results': len(results_list),
                'n_contaminants': n_contaminants,
                'n_membranes': n_membranes
            }
        }
    
    def _create_time_series_plots_batch(self, time_series_data):
        """Create batch time series plots with optimized rendering."""
        figures = {}
        config = self.plot_configs['time_series']
        
        for contaminant, data in time_series_data.items():
            if not data['time_arrays']:
                continue
            
            fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
            
            # Batch plot all time series for this contaminant
            for i in range(len(data['time_arrays'])):
                ax.plot(
                    data['time_arrays'][i],
                    data['concentration_arrays'][i],
                    color=data['colors'][i],
                    linewidth=config['linewidth'],
                    label=data['labels'][i],
                    marker='o',
                    markersize=config['markersize'],
                    alpha=config['alpha']
                )
            
            # Optimize plot appearance
            ax.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Concentration (mg/L)', fontsize=12, fontweight='bold')
            ax.set_title(f'{contaminant.replace("_", " ").title()} Removal Over Time', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            
            # Set log scale if concentrations span multiple orders of magnitude
            all_concentrations = np.concatenate(data['concentration_arrays'])
            if np.max(all_concentrations) / np.min(all_concentrations[all_concentrations > 0]) > 100:
                ax.set_yscale('log')
            
            plt.tight_layout()
            figures[f'time_series_{contaminant}'] = fig
        
        return figures
    
    def _create_comparison_plots_batch(self, comparison_data):
        """Create batch comparison plots with optimized bar charts."""
        figures = {}
        config = self.plot_configs['comparison']
        
        # Create multi-panel comparison figure
        n_contaminants = len(comparison_data)
        if n_contaminants == 0:
            return figures
        
        # Calculate subplot layout
        n_cols = min(3, n_contaminants)
        n_rows = (n_contaminants + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(config['figsize'][0] * n_cols / 3, 
                                 config['figsize'][1] * n_rows / 2))
        
        for i, (contaminant, data) in enumerate(comparison_data.items()):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Prepare data for bar chart
            x_pos = np.arange(len(data['membrane_types']))
            
            # Create bars
            bars = ax.bar(x_pos, data['removal_efficiencies'],
                         width=config['width'], alpha=config['alpha'],
                         color=[MEMBRANE_COLORS.get(mem, '#666666') 
                               for mem in data['membrane_types']])
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
            
            # Customize subplot
            ax.set_xlabel('Membrane Type', fontsize=10)
            ax.set_ylabel('Removal Efficiency (%)', fontsize=10)
            ax.set_title(f'{contaminant.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(data['membrane_types'], rotation=45, ha='right')
            ax.set_ylim(0, 105)
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        figures['comparison_multi'] = fig
        
        return figures
    
    def _create_heatmap_plots_batch(self, matrix_data):
        """Create batch heatmap plots with optimized matrix visualization."""
        figures = {}
        config = self.plot_configs['heatmap']
        
        if matrix_data['efficiency_matrix'].size == 0:
            return figures
        
        # Efficiency heatmap
        fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])
        
        # Create heatmap with optimized settings
        sns.heatmap(
            matrix_data['efficiency_matrix'],
            xticklabels=matrix_data['membrane_labels'],
            yticklabels=matrix_data['contaminant_labels'],
            annot=config['annot'],
            fmt=config['fmt'],
            cmap=config['cmap'],
            cbar_kws={'label': 'Removal Efficiency (%)'},
            ax=ax
        )
        
        ax.set_title('Contaminant Removal Efficiency Matrix', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Membrane Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Contaminant Type', fontsize=12, fontweight='bold')
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        figures['efficiency_heatmap'] = fig
        
        return figures
    
    def _save_figures_batch(self, figures_dict):
        """Efficiently save all figures in batch."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for plot_type, fig_data in figures_dict.items():
            if isinstance(fig_data, dict):
                # Multiple figures in this type
                for fig_name, fig in fig_data.items():
                    filename = f"{plot_type}_{fig_name}_{timestamp}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                    plt.close(fig)  # Free memory
            else:
                # Single figure
                filename = f"{plot_type}_{timestamp}.png"
                filepath = os.path.join(self.output_dir, filename)
                fig_data.savefig(filepath, dpi=300, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                plt.close(fig_data)  # Free memory
    
    def plot_membrane_performance_matrix(self, performance_data, save_plot=True):
        """
        Create optimized membrane performance matrix visualization.
        
        Args:
            performance_data (pd.DataFrame): Performance ranking data
            save_plot (bool): Whether to save the plot
            
        Returns:
            matplotlib.figure.Figure: Performance matrix figure
        """
        config = self.plot_configs['matrix']
        
        # Create pivot table for matrix visualization
        pivot_flux = performance_data.pivot_table(
            values='flux_lmh', 
            index='membrane_type', 
            columns='pore_size_nm',
            aggfunc='mean'
        )
        
        pivot_rejection = performance_data.pivot_table(
            values='rejection_percent',
            index='membrane_type',
            columns='pore_size_nm', 
            aggfunc='mean'
        )
        
        # Create subplot matrix
        fig = plt.figure(figsize=config['figsize'], dpi=config['dpi'])
        gs = GridSpec(2, 2, figure=fig, hspace=config['subplot_spacing'], 
                     wspace=config['subplot_spacing'])
        
        # Flux heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(pivot_flux, annot=True, fmt='.1f', cmap='Blues',
                   cbar_kws={'label': 'Flux (LMH)'}, ax=ax1)
        ax1.set_title('Water Flux Matrix', fontweight='bold')
        ax1.set_xlabel('Pore Size (nm)')
        ax1.set_ylabel('Membrane Type')
        
        # Rejection heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        sns.heatmap(pivot_rejection, annot=True, fmt='.1f', cmap='Reds',
                   cbar_kws={'label': 'Rejection (%)'}, ax=ax2)
        ax2.set_title('Oil Rejection Matrix', fontweight='bold')
        ax2.set_xlabel('Pore Size (nm)')
        ax2.set_ylabel('Membrane Type')
        
        # Performance scatter plot
        ax3 = fig.add_subplot(gs[1, :])
        
        for mem_type in performance_data['membrane_type'].unique():
            type_data = performance_data[performance_data['membrane_type'] == mem_type]
            ax3.scatter(type_data['flux_lmh'], type_data['rejection_percent'],
                       c=MEMBRANE_COLORS.get(mem_type, '#666666'),
                       s=100, alpha=0.7, label=mem_type,
                       edgecolors='black', linewidth=0.5)
        
        ax3.set_xlabel('Water Flux (LMH)', fontweight='bold')
        ax3.set_ylabel('Oil Rejection (%)', fontweight='bold')
        ax3.set_title('Flux vs Rejection Performance Map', fontweight='bold')
        ax3.legend(title='Membrane Type', frameon=True, fancybox=True)
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Membrane Performance Analysis Matrix', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_matrix_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig

# Create global instance for efficient reuse
ULTRA_PLOTTER = UltraEfficientPlotter(output_dir=r"C:\Users\ramaa\Documents\graphene_mebraine\output\plots\ultra_efficient")

def plot_chemistry_results_ultra_fast(results_list, save_plots=True):
    """
    Ultra-fast chemistry results plotting with global plotter instance.
    """
    return ULTRA_PLOTTER.plot_batch_chemistry_results(results_list, save_plots)

def plot_membrane_performance_ultra_fast(performance_data, save_plot=True):
    """
    Ultra-fast membrane performance plotting.
    """
    return ULTRA_PLOTTER.plot_membrane_performance_matrix(performance_data, save_plot)

def plot_real_simulation_results_from_output(output_dir="output", save_plots=True):
    """
    Loads real simulation results from the output directory (CSV or JSON),
    aggregates them, and plots using ultra-efficient plotting functions.
    """
    import glob
    import json
    import pandas as pd
    import os

    # Collect all JSON and CSV result files
    result_files = glob.glob(os.path.join(output_dir, "*.json")) + \
                   glob.glob(os.path.join(output_dir, "*.csv"))
    results_list = []

    for file in result_files:
        if file.endswith(".json"):
            with open(file, "r") as f:
                data = json.load(f)
                # If the file contains a list, extend; else, append
                if isinstance(data, list):
                    results_list.extend(data)
                else:
                    results_list.append(data)
        elif file.endswith(".csv"):
            df = pd.read_csv(file)
            # Convert DataFrame rows to dicts (if compatible)
            results_list.extend(df.to_dict(orient="records"))

    if not results_list:
        print("No simulation results found in output directory.")
        return None

    # Use the ultra-efficient plotting function
    figures = plot_chemistry_results_ultra_fast(results_list, save_plots=save_plots)
    print(f"Plotted {len(figures)} figure groups from real simulation results.")
    return figures

if __name__ == "__main__":
    # Performance demonstration
    import time
    
    plotter = UltraEfficientPlotter()
    
    # Create mock data for testing
    mock_results = []
    for i in range(10):
        result = {
            'membrane_type': ['GO', 'rGO', 'hybrid'][i % 3],
            'time_min': list(range(0, 61, 10)),
            'contaminants': {
                'Pb2+': {
                    'concentration_mg_L': [100 * np.exp(-0.05 * t) for t in range(0, 61, 10)]
                },
                'E_coli': {
                    'concentration_mg_L': [1000 * np.exp(-0.1 * t) for t in range(0, 61, 10)]
                }
            }
        }
        mock_results.append(result)
    
    print("Testing ultra-efficient plotting...")
    
    start_time = time.time()
    figures = plotter.plot_batch_chemistry_results(mock_results, save_plots=False)
    plotting_time = time.time() - start_time
    
    print(f"Batch plotting completed in {plotting_time:.4f} seconds")
    print(f"Generated {len(figures)} figure groups")
    print(f"Total figures: {sum(len(figs) if isinstance(figs, dict) else 1 for figs in figures.values())}")
    
    # Clean up figures to free memory
    for fig_group in figures.values():
        if isinstance(fig_group, dict):
            for fig in fig_group.values():
                plt.close(fig)
        else:
            plt.close(fig_group)
