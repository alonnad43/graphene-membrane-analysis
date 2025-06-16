# plot_all_results.py

"""
Generates comparative plots for:
- Water flux vs. pressure
- Oil rejection rate
- Stress-strain profiles

# TODO: Add plot export to PDF summary
# TODO: Add plotting styles for consistent visual identity
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from datetime import datetime

class ComprehensivePlotter:
    """
    Creates publication-quality plots for all simulation results.
    """
    
    def __init__(self, style='scientific'):
        self.setup_style(style)
        self.colors = {
            'GO': '#1f77b4',
            'rGO': '#2ca02c', 
            'Hybrid': '#ff7f0e'
        }
    
    def setup_style(self, style):
        """Set up plotting style."""
        if style == 'scientific':
            plt.style.use('seaborn-v0_8-whitegrid' if hasattr(plt.style, 'use') else 'default')
            plt.rcParams.update({
                'font.size': 12,
                'font.family': 'serif',
                'axes.linewidth': 1.2,
                'lines.linewidth': 2,
                'figure.figsize': (8, 6),
                'figure.dpi': 300
            })
    
    def plot_flux_vs_pressure_comprehensive(self, unified_data, output_dir):
        """
        Create comprehensive flux vs pressure plots with Phase 1 and Phase 3 data.
        
        Args:
            unified_data (pd.DataFrame): Unified results data
            output_dir (str): Directory for output plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Water Flux Analysis - All Phases', fontsize=16, fontweight='bold')
        
        # Plot 1: Phase 1 theoretical flux
        ax1 = axes[0, 0]
        for material in unified_data['material'].unique():
            data = unified_data[unified_data['material'] == material]
            ax1.scatter(data['pressure_bar'], data['flux_lmh'], 
                       color=self.colors.get(material, 'gray'), 
                       label=material, alpha=0.7, s=50)
        
        ax1.set_xlabel('Pressure (bar)')
        ax1.set_ylabel('Flux (L·m⁻²·h⁻¹)')
        ax1.set_title('Phase 1: Theoretical Flux')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: LAMMPS flux comparison
        ax2 = axes[0, 1]
        lammps_data = unified_data[unified_data['lammps_success'] == True]
        
        if not lammps_data.empty:
            for material in lammps_data['material'].unique():
                data = lammps_data[lammps_data['material'] == material]
                ax2.scatter(data['pressure_bar'], data['lammps_flux_rate'], 
                           color=self.colors.get(material, 'gray'), 
                           label=f'{material} (LAMMPS)', alpha=0.7, s=50)
        
        ax2.set_xlabel('Pressure (bar)')
        ax2.set_ylabel('LAMMPS Flux Rate')
        ax2.set_title('Phase 3: LAMMPS Flux')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Theoretical vs LAMMPS correlation
        ax3 = axes[1, 0]
        if not lammps_data.empty and 'lammps_theoretical_ratio' in lammps_data.columns:
            valid_data = lammps_data.dropna(subset=['lammps_theoretical_ratio'])
            if not valid_data.empty:
                ax3.scatter(valid_data['flux_lmh'], valid_data['lammps_flux_rate'],
                           c=[self.colors.get(m, 'gray') for m in valid_data['material']],
                           alpha=0.7, s=50)
                
                # Add diagonal line for perfect correlation
                min_val = min(valid_data['flux_lmh'].min(), valid_data['lammps_flux_rate'].min())
                max_val = max(valid_data['flux_lmh'].max(), valid_data['lammps_flux_rate'].max())
                ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect correlation')
        
        ax3.set_xlabel('Theoretical Flux (L·m⁻²·h⁻¹)')
        ax3.set_ylabel('LAMMPS Flux Rate')
        ax3.set_title('Phase 1 vs Phase 3 Correlation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance score distribution
        ax4 = axes[1, 1]
        if 'performance_score' in unified_data.columns:
            materials = unified_data['material'].unique()
            performance_data = [unified_data[unified_data['material'] == m]['performance_score'].dropna() 
                              for m in materials]
            
            bp = ax4.boxplot(performance_data, labels=materials, patch_artist=True)
            for patch, material in zip(bp['boxes'], materials):
                patch.set_facecolor(self.colors.get(material, 'gray'))
                patch.set_alpha(0.7)
        
        ax4.set_ylabel('Performance Score')
        ax4.set_title('Overall Performance Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'flux_comprehensive_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rejection_analysis(self, unified_data, output_dir):
        """Create oil rejection analysis plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Oil Rejection Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Rejection by material
        ax1 = axes[0]
        materials = unified_data['material'].unique()
        rejection_means = [unified_data[unified_data['material'] == m]['rejection_percent'].mean() 
                          for m in materials]
        rejection_stds = [unified_data[unified_data['material'] == m]['rejection_percent'].std() 
                         for m in materials]
        
        bars = ax1.bar(materials, rejection_means, yerr=rejection_stds, 
                      color=[self.colors.get(m, 'gray') for m in materials],
                      alpha=0.7, capsize=5)
        
        ax1.set_ylabel('Oil Rejection (%)')
        ax1.set_title('Average Rejection by Material')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean in zip(bars, rejection_means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{mean:.1f}%', ha='center', va='bottom')
        
        # Plot 2: Rejection vs flux trade-off
        ax2 = axes[1]
        for material in materials:
            data = unified_data[unified_data['material'] == material]
            ax2.scatter(data['flux_lmh'], data['rejection_percent'],
                       color=self.colors.get(material, 'gray'),
                       label=material, alpha=0.7, s=50)
        
        ax2.set_xlabel('Flux (L·m⁻²·h⁻¹)')
        ax2.set_ylabel('Oil Rejection (%)')
        ax2.set_title('Flux vs Rejection Trade-off')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance score vs rejection
        ax3 = axes[2]
        if 'performance_score' in unified_data.columns:
            for material in materials:
                data = unified_data[unified_data['material'] == material]
                valid_data = data.dropna(subset=['performance_score'])
                if not valid_data.empty:
                    ax3.scatter(valid_data['rejection_percent'], valid_data['performance_score'],
                               color=self.colors.get(material, 'gray'),
                               label=material, alpha=0.7, s=50)
        
        ax3.set_xlabel('Oil Rejection (%)')
        ax3.set_ylabel('Performance Score')
        ax3.set_title('Rejection vs Overall Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rejection_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_structural_analysis(self, unified_data, output_dir):
        """Create structural analysis plots for hybrid membranes."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter for hybrid membranes with structural data
        hybrid_data = unified_data[
            (unified_data['material'] == 'Hybrid') & 
            (unified_data.get('structure_assigned', False) == True)
        ]
        
        if hybrid_data.empty:
            print("No structural data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hybrid Membrane Structural Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Layer composition effect on flux
        ax1 = axes[0, 0]
        if 'go_fraction' in hybrid_data.columns:
            scatter = ax1.scatter(hybrid_data['go_fraction'], hybrid_data['flux_lmh'],
                                 c=hybrid_data['total_layers'], cmap='viridis',
                                 alpha=0.7, s=100)
            plt.colorbar(scatter, ax=ax1, label='Total Layers')
        
        ax1.set_xlabel('GO Fraction')
        ax1.set_ylabel('Flux (L·m⁻²·h⁻¹)')
        ax1.set_title('Layer Composition vs Flux')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Thickness effect
        ax2 = axes[0, 1]
        if 'total_thickness_nm' in hybrid_data.columns:
            ax2.scatter(hybrid_data['total_thickness_nm'], hybrid_data['flux_lmh'],
                       color=self.colors['Hybrid'], alpha=0.7, s=50)
        
        ax2.set_xlabel('Total Thickness (nm)')
        ax2.set_ylabel('Flux (L·m⁻²·h⁻¹)')
        ax2.set_title('Thickness vs Flux')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Layer sequence visualization
        ax3 = axes[1, 0]
        if 'layer_sequence' in hybrid_data.columns:
            sequences = hybrid_data['layer_sequence'].value_counts()
            sequences_truncated = sequences.head(10)  # Show top 10
            
            bars = ax3.barh(range(len(sequences_truncated)), sequences_truncated.values,
                           color=self.colors['Hybrid'], alpha=0.7)
            ax3.set_yticks(range(len(sequences_truncated)))
            ax3.set_yticklabels([seq[:20] + '...' if len(seq) > 20 else seq 
                               for seq in sequences_truncated.index])
        
        ax3.set_xlabel('Frequency')
        ax3.set_title('Most Common Layer Sequences')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Performance vs structure complexity
        ax4 = axes[1, 1]
        if all(col in hybrid_data.columns for col in ['total_layers', 'performance_score']):
            valid_data = hybrid_data.dropna(subset=['performance_score'])
            if not valid_data.empty:
                ax4.scatter(valid_data['total_layers'], valid_data['performance_score'],
                           color=self.colors['Hybrid'], alpha=0.7, s=50)
        
        ax4.set_xlabel('Total Layers')
        ax4.set_ylabel('Performance Score')
        ax4.set_title('Structure Complexity vs Performance')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'structural_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_dashboard(self, unified_data, summary_stats, output_dir):
        """Create a comprehensive summary dashboard."""
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Membrane Simulation Summary Dashboard', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # Statistics text box
        ax_stats = fig.add_subplot(gs[0, :2])
        stats_text = f"""
        Total Simulations: {summary_stats['total_simulations']}
        Materials Tested: {summary_stats['materials_tested']}
        Pressure Range: {summary_stats['pressure_range']['min']:.1f} - {summary_stats['pressure_range']['max']:.1f} bar
        
        Flux Statistics:
        • Mean: {summary_stats['flux_statistics']['mean']:.1f} L·m⁻²·h⁻¹
        • Range: {summary_stats['flux_statistics']['min']:.1f} - {summary_stats['flux_statistics']['max']:.1f}
        
        Rejection Statistics:
        • Mean: {summary_stats['rejection_statistics']['mean']:.1f}%
        • Range: {summary_stats['rejection_statistics']['min']:.1f} - {summary_stats['rejection_statistics']['max']:.1f}%
        """
        
        if 'phase3_success_rate' in summary_stats:
            stats_text += f"\nLAMMPS Success Rate: {summary_stats['phase3_success_rate']:.1%}"
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax_stats.axis('off')
        
        # Top performers table
        ax_table = fig.add_subplot(gs[0, 2:])
        top_performers = summary_stats.get('top_performers', {})
        
        if top_performers:
            table_data = []
            headers = ['Rank', 'Membrane', 'Flux', 'Rejection', 'Score']
            
            for i, perf in enumerate(top_performers.get('best_overall', [])[:3]):
                table_data.append([
                    i+1,
                    perf['membrane_name'][:15] + '...' if len(perf['membrane_name']) > 15 else perf['membrane_name'],
                    f"{unified_data[unified_data['membrane_name'] == perf['membrane_name']]['flux_lmh'].iloc[0]:.0f}",
                    f"{unified_data[unified_data['membrane_name'] == perf['membrane_name']]['rejection_percent'].iloc[0]:.1f}%",
                    f"{perf['performance_score']:.2f}"
                ])
            
            table = ax_table.table(cellText=table_data, colLabels=headers,
                                 cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
        
        ax_table.axis('off')
        ax_table.set_title('Top Performers', fontweight='bold')
        
        # Additional plots in remaining subplots
        # Flux distribution
        ax1 = fig.add_subplot(gs[1, :2])
        unified_data['flux_lmh'].hist(bins=20, alpha=0.7, ax=ax1)
        ax1.set_xlabel('Flux (L·m⁻²·h⁻¹)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Flux Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Material comparison
        ax2 = fig.add_subplot(gs[1, 2:])
        materials = unified_data['material'].unique()
        flux_by_material = [unified_data[unified_data['material'] == m]['flux_lmh'].values 
                           for m in materials]
        bp = ax2.boxplot(flux_by_material, labels=materials, patch_artist=True)
        for patch, material in zip(bp['boxes'], materials):
            patch.set_facecolor(self.colors.get(material, 'gray'))
            patch.set_alpha(0.7)
        ax2.set_ylabel('Flux (L·m⁻²·h⁻¹)')
        ax2.set_title('Flux by Material')
        ax2.grid(True, alpha=0.3)
        
        # Continue with more plots...
        
        plt.savefig(os.path.join(output_dir, 'summary_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_plots(self, unified_data, summary_stats, output_base_dir):
        """
        Generate all comprehensive plots.
        
        Args:
            unified_data (pd.DataFrame): Unified simulation results
            summary_stats (dict): Summary statistics
            output_base_dir (str): Base directory for all plots
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = os.path.join(output_base_dir, f"comprehensive_plots_{timestamp}")
        os.makedirs(plot_dir, exist_ok=True)
        
        print("Generating comprehensive plots...")
        
        # Generate all plot types
        self.plot_flux_vs_pressure_comprehensive(unified_data, plot_dir)
        self.plot_rejection_analysis(unified_data, plot_dir)
        self.plot_structural_analysis(unified_data, plot_dir)
        self.create_summary_dashboard(unified_data, summary_stats, plot_dir)
        print(f"All plots saved to: {plot_dir}")
        return plot_dir
    
    def plot_phase4_contaminant_removal(self, unified_data, output_dir):
        """
        Create Phase 4 chemical/biological removal efficiency plots.
        
        Args:
            unified_data (pd.DataFrame): Unified dataset with Phase 4 data
            output_dir (str): Directory to save plots
        """
        # Filter for Phase 4 data
        phase4_data = unified_data[unified_data['phase'] == 'Phase4_Chemical_Biological'].copy()
        
        if phase4_data.empty:
            print("No Phase 4 data available for plotting")
            return
        
        # Contaminant removal efficiency by membrane type
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Phase 4: Chemical & Biological Contaminant Removal', fontsize=16, fontweight='bold')
        
        # Plot 1: Removal efficiency by contaminant type
        if 'contaminant_type' in phase4_data.columns and 'removal_efficiency_percent' in phase4_data.columns:
            contaminant_summary = phase4_data.groupby(['contaminant_type', 'membrane_type'])['removal_efficiency_percent'].mean().reset_index()
            contaminant_pivot = contaminant_summary.pivot(index='contaminant_type', columns='membrane_type', values='removal_efficiency_percent')
            
            contaminant_pivot.plot(kind='bar', ax=axes[0,0], color=[self.colors.get(col, '#666666') for col in contaminant_pivot.columns])
            axes[0,0].set_title('Removal Efficiency by Contaminant Type')
            axes[0,0].set_ylabel('Removal Efficiency (%)')
            axes[0,0].legend(title='Membrane Type')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Individual contaminant performance
        if 'contaminant' in phase4_data.columns:
            top_contaminants = phase4_data.groupby('contaminant')['removal_efficiency_percent'].mean().sort_values(ascending=False).head(8)
            contaminant_data = phase4_data[phase4_data['contaminant'].isin(top_contaminants.index)]
            
            for membrane in ['GO', 'rGO', 'hybrid']:
                mem_data = contaminant_data[contaminant_data['membrane_type'] == membrane]
                if not mem_data.empty:
                    axes[0,1].bar(mem_data['contaminant'], mem_data['removal_efficiency_percent'], 
                                 label=membrane, alpha=0.7, color=self.colors.get(membrane, '#666666'))
            
            axes[0,1].set_title('Top Performing Contaminants')
            axes[0,1].set_ylabel('Removal Efficiency (%)')
            axes[0,1].legend()
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Adsorption capacity comparison (if available)
        if 'adsorption_capacity_mg_g' in phase4_data.columns:
            adsorption_data = phase4_data.dropna(subset=['adsorption_capacity_mg_g'])
            if not adsorption_data.empty:
                membrane_capacity = adsorption_data.groupby('membrane_type')['adsorption_capacity_mg_g'].mean()
                axes[1,0].bar(membrane_capacity.index, membrane_capacity.values, 
                             color=[self.colors.get(mem, '#666666') for mem in membrane_capacity.index])
                axes[1,0].set_title('Average Adsorption Capacity')
                axes[1,0].set_ylabel('Capacity (mg/g)')
        
        # Plot 4: pH dependency (if available)
        if 'pH' in phase4_data.columns:
            ph_performance = phase4_data.groupby(['pH', 'membrane_type'])['removal_efficiency_percent'].mean().reset_index()
            for membrane in ['GO', 'rGO', 'hybrid']:
                mem_data = ph_performance[ph_performance['membrane_type'] == membrane]
                if not mem_data.empty:
                    axes[1,1].plot(mem_data['pH'], mem_data['removal_efficiency_percent'], 
                                  marker='o', label=membrane, color=self.colors.get(membrane, '#666666'))
            
            axes[1,1].set_title('pH Dependency of Removal Efficiency')
            axes[1,1].set_xlabel('pH')
            axes[1,1].set_ylabel('Removal Efficiency (%)')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'phase4_contaminant_removal_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Phase 4 contaminant removal plot saved: {plot_path}")
        return plot_path
