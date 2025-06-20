# plot_chemistry.py

"""
Visualization module for Phase 4 Chemical and Biological Simulation results.

Creates publication-quality plots for:
- Contaminant reduction over time
- Membrane saturation curves
- Comparative performance across membrane types
- Regeneration cycle effects
- Multi-contaminant removal efficiency
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os
from datetime import datetime

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ChemistryPlotter:
    """
    Visualization class for Phase 4 chemical simulation results.
    """
    
    def __init__(self, results=None, output_dir=r"C:\Users\ramaa\Documents\graphene_mebraine\output\plots\phase4_chemistry"):
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define colors for membrane types
        self.membrane_colors = {
            'GO': '#FF6B6B',      # Red
            'rGO': '#4ECDC4',     # Teal
            'hybrid': '#45B7D1',  # Blue
            'Hybrid': '#45B7D1'   # Alternative naming
        }
        
        # Define markers for contaminant types
        self.contaminant_markers = {
            'heavy metal': 'o',
            'bacteria': 's',
            'salt': '^',
            'organic_dye': 'D',
            'organic_pollutant': 'v',
            'virus': 'P',
            'inorganic_anion': 'h'
        }
    
    def plot_contaminant_reduction_time_series(self, results=None, save_plots=True):
        """
        Plot contaminant concentration reduction over time for all membrane types.
        
        Args:
            results (list): List of simulation results or None to use self.results
            save_plots (bool): Whether to save plots to files
            
        Returns:
            dict: Dictionary of matplotlib figure objects
        """
        if results is None:
            results = self.results
        
        if not results:
            print("No results available for plotting")
            return {}
        
        figures = {}
        
        # Get all unique contaminants
        all_contaminants = set()
        for result in results:
            all_contaminants.update(result['contaminants'].keys())
        
        for contaminant in all_contaminants:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            for result in results:
                if contaminant not in result['contaminants']:
                    continue
                
                membrane_type = result['membrane_type']
                contaminant_data = result['contaminants'][contaminant]
                time_points = result['time_points']
                
                # Plot concentration vs time
                if 'concentration_mg_L' in contaminant_data:
                    conc = contaminant_data['concentration_mg_L']
                    label = f"{membrane_type} (Final: {conc[-1]:.2f} mg/L)"
                    ax.plot(time_points, conc, 
                           color=self.membrane_colors.get(membrane_type, 'gray'),
                           linewidth=2.5, label=label,
                           marker='o', markersize=4, alpha=0.8)
                
                elif 'cfu_ml' in contaminant_data:
                    cfu = contaminant_data['cfu_ml']
                    label = f"{membrane_type} (Final: {cfu[-1]:.1e} CFU/mL)"
                    ax.semilogy(time_points, cfu,
                               color=self.membrane_colors.get(membrane_type, 'gray'),
                               linewidth=2.5, label=label,
                               marker='s', markersize=4, alpha=0.8)
                
                elif 'permeate_concentration_mg_L' in contaminant_data:
                    conc = contaminant_data['permeate_concentration_mg_L']
                    rejection = contaminant_data['rejection_percent']
                    label = f"{membrane_type} (Rejection: {rejection:.1f}%)"
                    ax.plot(time_points, conc,
                           color=self.membrane_colors.get(membrane_type, 'gray'),
                           linewidth=2.5, label=label,
                           marker='^', markersize=4, alpha=0.8)
            
            # Formatting
            ax.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
            
            if 'bacteria' in contaminant or 'virus' in contaminant:
                ax.set_ylabel('Viable Count (CFU/mL)', fontsize=12, fontweight='bold')
                ax.set_yscale('log')
            else:
                ax.set_ylabel('Concentration (mg/L)', fontsize=12, fontweight='bold')
            
            ax.set_title(f'{contaminant} Removal Over Time', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                filename = f"contaminant_reduction_{contaminant.replace('+', 'plus').replace('_', '-')}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Saved: {filepath}")
            
            figures[contaminant] = fig
        
        return figures
    
    def plot_membrane_saturation_curves(self, results=None, save_plots=True):
        """
        Plot membrane saturation curves for adsorption processes.
        
        Args:
            results (list): List of simulation results
            save_plots (bool): Whether to save plots to files
            
        Returns:
            dict: Dictionary of matplotlib figure objects
        """
        if results is None:
            results = self.results
        
        if not results:
            return {}
        
        figures = {}
        
        # Find contaminants with adsorption data
        adsorption_contaminants = set()
        for result in results:
            for contaminant, data in result['contaminants'].items():
                if 'saturation_percent' in data:
                    adsorption_contaminants.add(contaminant)
        
        for contaminant in adsorption_contaminants:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            for result in results:
                if contaminant not in result['contaminants']:
                    continue
                
                membrane_type = result['membrane_type']
                contaminant_data = result['contaminants'][contaminant]
                time_points = result['time_points']
                
                if 'saturation_percent' not in contaminant_data:
                    continue
                
                saturation = contaminant_data['saturation_percent']
                adsorbed = contaminant_data.get('adsorbed_mg_g', [])
                q_max = contaminant_data.get('q_max', 100)
                
                color = self.membrane_colors.get(membrane_type, 'gray')
                
                # Plot saturation percentage
                ax1.plot(time_points, saturation, 
                        color=color, linewidth=2.5,
                        label=f"{membrane_type} (q_max: {q_max:.1f} mg/g)",
                        marker='o', markersize=4, alpha=0.8)
                
                # Plot adsorbed amount
                if len(adsorbed) > 0:
                    ax2.plot(time_points, adsorbed,
                            color=color, linewidth=2.5,
                            label=f"{membrane_type}",
                            marker='s', markersize=4, alpha=0.8)
                    
                    # Add q_max line
                    ax2.axhline(y=q_max, color=color, linestyle='--', alpha=0.5)
            
            # Format saturation plot
            ax1.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Saturation (%)', fontsize=12, fontweight='bold')
            ax1.set_title(f'{contaminant} Membrane Saturation', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 105)
            
            # Format adsorbed amount plot
            ax2.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Adsorbed Amount (mg/g)', fontsize=12, fontweight='bold')
            ax2.set_title(f'{contaminant} Adsorption Kinetics', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                filename = f"membrane_saturation_{contaminant.replace('+', 'plus').replace('_', '-')}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Saved: {filepath}")
            
            figures[f"{contaminant}_saturation"] = fig
        
        return figures
    
    def plot_comparative_performance(self, results=None, save_plots=True):
        """
        Create comparative performance plot across all membrane types and contaminants.
        
        Args:
            results (list): List of simulation results
            save_plots (bool): Whether to save plots to files
            
        Returns:
            matplotlib.figure.Figure: Comparative performance heatmap
        """
        if results is None:
            results = self.results
        
        if not results:
            return None
        
        # Prepare data for heatmap
        performance_data = {}
        membrane_types = set()
        contaminants = set()
        
        for result in results:
            membrane_type = result['membrane_type']
            membrane_types.add(membrane_type)
            
            for contaminant, data in result['contaminants'].items():
                contaminants.add(contaminant)
                
                # Get removal efficiency
                efficiency = 0
                if 'removal_efficiency' in data:
                    efficiency = data['removal_efficiency']
                elif 'kill_efficiency' in data:
                    efficiency = data['kill_efficiency']
                elif 'rejection_percent' in data:
                    efficiency = max(0, data['rejection_percent'])
                
                performance_data[(membrane_type, contaminant)] = efficiency
        
        # Create performance matrix
        membrane_list = sorted(list(membrane_types))
        contaminant_list = sorted(list(contaminants))
        
        performance_matrix = np.zeros((len(contaminant_list), len(membrane_list)))
        
        for i, contaminant in enumerate(contaminant_list):
            for j, membrane in enumerate(membrane_list):
                performance_matrix[i, j] = performance_data.get((membrane, contaminant), 0)
        
        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        im = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks and labels
        ax.set_xticks(range(len(membrane_list)))
        ax.set_yticks(range(len(contaminant_list)))
        ax.set_xticklabels(membrane_list, fontsize=11, fontweight='bold')
        ax.set_yticklabels(contaminant_list, fontsize=11)
        
        # Add text annotations
        for i in range(len(contaminant_list)):
            for j in range(len(membrane_list)):
                text = f"{performance_matrix[i, j]:.1f}%"
                ax.text(j, i, text, ha="center", va="center", 
                       color="white" if performance_matrix[i, j] < 50 else "black",
                       fontweight='bold', fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Removal Efficiency (%)', fontsize=12, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Membrane Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Contaminant', fontsize=12, fontweight='bold')
        ax.set_title('Phase 4: Comparative Contaminant Removal Performance', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            filepath = os.path.join(self.output_dir, "comparative_performance_heatmap.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_regeneration_effects(self, original_results, regenerated_results, save_plots=True):
        """
        Plot the effects of regeneration on membrane performance.
        
        Args:
            original_results (list): Results before regeneration
            regenerated_results (list): Results after regeneration cycles
            save_plots (bool): Whether to save plots to files
            
        Returns:
            matplotlib.figure.Figure: Regeneration effects plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Compare key metrics before and after regeneration
        metrics = ['q_max', 'removal_efficiency', 'k2', 'saturation_percent']
        metric_labels = ['Max Capacity (mg/g)', 'Removal Efficiency (%)', 
                        'Rate Constant (g/mg·min)', 'Final Saturation (%)']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]
            
            # Extract data for comparison
            membranes = []
            original_values = []
            regenerated_values = []
            
            for orig_result, regen_result in zip(original_results, regenerated_results):
                membrane = orig_result['membrane_type']
                
                for contaminant in orig_result['contaminants']:
                    if contaminant not in regen_result['contaminants']:
                        continue
                    
                    orig_data = orig_result['contaminants'][contaminant]
                    regen_data = regen_result['contaminants'][contaminant]
                    
                    if metric in orig_data and metric in regen_data:
                        membranes.append(f"{membrane}\n{contaminant}")
                        
                        if metric == 'saturation_percent':
                            original_values.append(orig_data[metric][-1])
                            regenerated_values.append(regen_data[metric][-1])
                        else:
                            original_values.append(orig_data[metric])
                            regenerated_values.append(regen_data[metric])
            
            if membranes:
                x = np.arange(len(membranes))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, original_values, width, 
                              label='Original', alpha=0.8, color='#3498db')
                bars2 = ax.bar(x + width/2, regenerated_values, width,
                              label='After Regeneration', alpha=0.8, color='#e74c3c')
                
                ax.set_xlabel('Membrane-Contaminant', fontsize=11, fontweight='bold')
                ax.set_ylabel(label, fontsize=11, fontweight='bold')
                ax.set_title(f'Regeneration Effect on {label}', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(membranes, rotation=45, ha='right', fontsize=9)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=8)
                
                for bar in bars2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_plots:
            filepath = os.path.join(self.output_dir, "regeneration_effects.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_multi_contaminant_summary(self, results=None, save_plots=True):
        """
        Create summary plot for multi-contaminant removal scenarios.
        
        Args:
            results (list): List of simulation results
            save_plots (bool): Whether to save plots to files
            
        Returns:
            matplotlib.figure.Figure: Multi-contaminant summary plot
        """
        if results is None:
            results = self.results
        
        if not results:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Overall removal efficiency by membrane type
        membrane_performance = {}
        for result in results:
            membrane = result['membrane_type']
            if membrane not in membrane_performance:
                membrane_performance[membrane] = []
            
            for contaminant, data in result['contaminants'].items():
                # Defensive: skip if data is not a dict (corrupt or unexpected input)
                if not isinstance(data, dict):
                    print(f"[WARN] Skipping contaminant '{contaminant}' for membrane '{membrane}' due to unexpected data type: {type(data)}")
                    continue
                efficiency = 0
                if 'removal_efficiency' in data:
                    efficiency = data['removal_efficiency']
                elif 'kill_efficiency' in data:
                    efficiency = data['kill_efficiency']
                elif 'rejection_percent' in data:
                    efficiency = max(0, data['rejection_percent'])
                
                membrane_performance[membrane].append(efficiency)

        # Box plot of performance
        membrane_names = list(membrane_performance.keys())
        performance_data = [membrane_performance[m] for m in membrane_names]
        
        box_plot = ax1.boxplot(performance_data, labels=membrane_names, patch_artist=True)
        for patch, membrane in zip(box_plot['boxes'], membrane_names):
            patch.set_facecolor(self.membrane_colors.get(membrane, 'gray'))
            patch.set_alpha(0.7)
        
        ax1.set_xlabel('Membrane Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Removal Efficiency (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Overall Performance Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Contaminant type performance
        contaminant_types = {}
        for result in results:
            for contaminant, data in result['contaminants'].items():
                if not isinstance(data, dict):
                    print(f"[WARN] Skipping contaminant '{contaminant}' for type analysis due to unexpected data type: {type(data)}")
                    continue
                # Determine contaminant type (simplified)
                if 'bacteria' in contaminant.lower() or 'coli' in contaminant.lower():
                    cont_type = 'Bacteria'
                elif any(metal in contaminant for metal in ['Pb', 'As', 'Cd', 'Cr']):
                    cont_type = 'Heavy Metals'
                elif 'nacl' in contaminant.lower() or 'mgso4' in contaminant.lower():
                    cont_type = 'Salts'
                else:
                    cont_type = 'Other'
                
                if cont_type not in contaminant_types:
                    contaminant_types[cont_type] = []
                
                efficiency = 0
                if 'removal_efficiency' in data:
                    efficiency = data['removal_efficiency']
                elif 'kill_efficiency' in data:
                    efficiency = data['kill_efficiency']
                elif 'rejection_percent' in data:
                    efficiency = max(0, data['rejection_percent'])
                
                contaminant_types[cont_type].append(efficiency)
        
        # Bar plot of contaminant types
        type_names = list(contaminant_types.keys())
        avg_performance = [np.mean(contaminant_types[t]) for t in type_names]
        std_performance = [np.std(contaminant_types[t]) for t in type_names]
        
        bars = ax2.bar(type_names, avg_performance, yerr=std_performance, 
                      capsize=5, alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        
        ax2.set_xlabel('Contaminant Type', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Removal Efficiency (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Performance by Contaminant Type', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, avg, std in zip(bars, avg_performance, std_performance):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{avg:.1f}±{std:.1f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        # 3. Time to equilibrium comparison
        equilibrium_times = {}
        for result in results:
            membrane = result['membrane_type']
            if membrane not in equilibrium_times:
                equilibrium_times[membrane] = []
            
            for contaminant, data in result['contaminants'].items():
                if 'saturation_percent' in data:
                    saturation = data['saturation_percent']
                    time_points = result['time_points']
                    
                    # Find time to reach 95% saturation
                    eq_time = None
                    for i, sat in enumerate(saturation):
                        if sat >= 95:
                            eq_time = time_points[i]
                            break
                    
                    if eq_time is not None:
                        equilibrium_times[membrane].append(eq_time)
        
        # Plot equilibrium times
        if equilibrium_times:
            eq_membranes = list(equilibrium_times.keys())
            eq_times = [np.mean(equilibrium_times[m]) if equilibrium_times[m] 
                       else 0 for m in eq_membranes]
            
            bars = ax3.bar(eq_membranes, eq_times, alpha=0.8,
                          color=[self.membrane_colors.get(m, 'gray') for m in eq_membranes])
            
            ax3.set_xlabel('Membrane Type', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Time to 95% Saturation (min)', fontsize=12, fontweight='bold')
            ax3.set_title('Equilibration Time Comparison', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. Mechanism frequency analysis
        mechanisms = {}
        for result in results:
            for contaminant, data in result['contaminants'].items():
                if 'interaction_mechanisms' in data:
                    for mechanism in data['interaction_mechanisms']:
                        mechanisms[mechanism] = mechanisms.get(mechanism, 0) + 1
                elif 'mechanisms' in data:
                    for mechanism in data['mechanisms']:
                        mechanisms[mechanism] = mechanisms.get(mechanism, 0) + 1
        
        if mechanisms:
            mech_names = list(mechanisms.keys())
            mech_counts = list(mechanisms.values())
            
            wedges, texts, autotexts = ax4.pie(mech_counts, labels=mech_names, autopct='%1.1f%%',
                                              startangle=90, colors=plt.cm.Set3.colors)
            
            ax4.set_title('Removal Mechanism Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            filepath = os.path.join(self.output_dir, "multi_contaminant_summary.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def create_chemistry_report(self, results=None, save_plots=True):
        """
        Generate a comprehensive visual report for Phase 4 results.
        
        Args:
            results (list): List of simulation results
            save_plots (bool): Whether to save all plots
            
        Returns:
            dict: Dictionary of all generated figures
        """
        if results is not None:
            self.results = results
        
        print(f"\n📊 Generating Phase 4 Chemistry Visualization Report")
        print(f"Output directory: {self.output_dir}")
        
        all_figures = {}
        
        # Generate all plots
        print("Creating contaminant reduction time series...")
        all_figures.update(self.plot_contaminant_reduction_time_series(save_plots=save_plots))
        
        print("Creating membrane saturation curves...")
        all_figures.update(self.plot_membrane_saturation_curves(save_plots=save_plots))
        
        print("Creating comparative performance heatmap...")
        comp_fig = self.plot_comparative_performance(save_plots=save_plots)
        if comp_fig:
            all_figures['comparative_performance'] = comp_fig
        
        print("Creating multi-contaminant summary...")
        summary_fig = self.plot_multi_contaminant_summary(save_plots=save_plots)
        if summary_fig:
            all_figures['multi_contaminant_summary'] = summary_fig
        
        print(f"✅ Phase 4 visualization report complete!")
        print(f"Generated {len(all_figures)} plots in {self.output_dir}")
        
        return all_figures

def plot_phase4_results(simulation_engine, save_plots=True):
    """
    Convenience function to plot all Phase 4 results.
    
    Args:
        simulation_engine: ChemicalSimulationEngine with completed simulations
        save_plots (bool): Whether to save plots to files
        
    Returns:
        dict: Dictionary of generated figures
    """
    plotter = ChemistryPlotter(results=simulation_engine.simulation_results)
    return plotter.create_chemistry_report(save_plots=save_plots)

if __name__ == "__main__":
    # Example usage - requires simulation results
    print("Phase 4 Chemistry Plotter ready.")
    print("Use plot_phase4_results(simulation_engine) to generate visualizations.")
