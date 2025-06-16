# unify_results.py

"""
Combines all results from Phase 1, 2, 3, and 4 into a master summary.

# TODO: Add flags for filtering hybrid-only, GO-only, etc.
# TODO: Add ranking logic by performance/cost ratio
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

class ResultsUnifier:
    """
    Combines results from all simulation phases into unified datasets.
    """

    def __init__(self):
        self.phase1_data = None
        self.phase2_data = None
        self.phase3_data = None
        self.phase4_data = None
        self.unified_data = None
    
    def load_phase1_results(self, excel_path=None, dataframe=None):
        """
        Load Phase 1 (macroscale) simulation results.
        
        Args:
            excel_path (str): Path to Excel file with Phase 1 results
            dataframe (pd.DataFrame): Direct dataframe input
        """
        if dataframe is not None:
            self.phase1_data = dataframe
        elif excel_path and os.path.exists(excel_path):
            self.phase1_data = pd.read_excel(excel_path)
        else:
            raise ValueError("Must provide either excel_path or dataframe for Phase 1 data")
    
    def load_phase2_results(self, structure_data):
        """
        Load Phase 2 (structural) results.
        
        Args:
            structure_data (list): List of HybridStructure objects or dictionaries
        """
        if isinstance(structure_data[0], dict):
            self.phase2_data = pd.DataFrame(structure_data)
        else:
            # Convert HybridStructure objects to DataFrame
            phase2_records = []
            for structure in structure_data:
                record = {
                    'structure_id': getattr(structure, 'name', str(hash(str(structure.layers)))),
                    'layer_sequence': structure.stacking_sequence,
                    'total_layers': len(structure.layers),
                    'total_thickness_nm': structure.total_thickness,
                    'interlayer_spacing_nm': structure.interlayer_spacing,
                    'go_fraction': structure.layers.count('GO') / len(structure.layers),
                    'rgo_fraction': structure.layers.count('rGO') / len(structure.layers)
                }
                phase2_records.append(record)
            self.phase2_data = pd.DataFrame(phase2_records)
    
    def load_phase3_results(self, lammps_results):
        """
        Load Phase 3 (LAMMPS) simulation results.
        
        Args:
            lammps_results (dict): Dictionary of LAMMPS analysis results
        """
        phase3_records = []
        
        for sim_name, results in lammps_results.items():
            if results.get('success', False):
                record = {
                    'simulation_name': sim_name,
                    'lammps_success': True,
                    'final_temperature': results['thermodynamics'].get('final_temperature'),
                    'final_pressure': results['thermodynamics'].get('final_pressure'),
                    'avg_potential_energy': results['thermodynamics'].get('avg_potential_energy'),
                    'simulation_steps': results['thermodynamics'].get('simulation_steps'),
                    'estimated_flux_rate': results['flux_analysis'].get('estimated_flux_rate'),
                    'water_z_drift': results['flux_analysis'].get('z_drift'),
                    'box_volume_change': results['trajectory_summary'].get('box_volume_change')
                }
            else:
                record = {
                    'simulation_name': sim_name,
                    'lammps_success': False,
                    'error': results.get('error', 'Unknown error')
                }
            phase3_records.append(record)
        
        self.phase3_data = pd.DataFrame(phase3_records)
    
    def create_unified_dataset(self):
        """
        Combine all phase results into a single comprehensive dataset.
        
        Returns:
            pd.DataFrame: Unified results
        """
        if self.phase1_data is None:
            raise ValueError("Phase 1 data must be loaded first")
        
        # Start with Phase 1 data as base
        unified = self.phase1_data.copy()
        
        # Add Phase 2 structural information
        if self.phase2_data is not None:
            # Try to match by membrane name or create mapping
            unified = self._merge_phase2_data(unified)
        
        # Add Phase 3 LAMMPS results
        if self.phase3_data is not None:
            unified = self._merge_phase3_data(unified)
        
        # Calculate derived properties
        unified = self._calculate_derived_properties(unified)
        
        self.unified_data = unified
        return unified
    
    def _merge_phase2_data(self, df):
        """Merge Phase 2 structural data with main dataset."""
        # Simple merge based on material type for now
        # In practice, would need more sophisticated matching
        
        df['structure_assigned'] = False
        
        for idx, row in df.iterrows():
            material = row['material']
            if material == 'Hybrid' and not self.phase2_data.empty:                # Assign first available hybrid structure
                structure_row = self.phase2_data.iloc[0]
                df.at[idx, 'layer_sequence'] = structure_row['layer_sequence']
                df.at[idx, 'total_layers'] = structure_row['total_layers']
                df.at[idx, 'interlayer_spacing_nm'] = structure_row['interlayer_spacing_nm']
                df.at[idx, 'go_fraction'] = structure_row['go_fraction']
                df.at[idx, 'rgo_fraction'] = structure_row['rgo_fraction']
                df.at[idx, 'structure_assigned'] = True
        
        return df
    
    def _merge_phase3_data(self, df):
        """Merge Phase 3 LAMMPS data with main dataset."""
        # Match by membrane name (simplified)
        
        df['lammps_simulated'] = False
        df['lammps_success'] = False
        df['atomistic_flux_lmh'] = np.nan
        df['atomistic_modulus_gpa'] = np.nan
        df['simulation_temperature_k'] = np.nan
        df['simulation_pressure_bar'] = np.nan
        
        for idx, row in df.iterrows():
            membrane_name = row['membrane_name']
            
            # Try to find matching LAMMPS result
            lammps_match = self.phase3_data[
                self.phase3_data['simulation_name'].str.contains(membrane_name.replace(' ', '_'), case=False, na=False)
            ]
            
            if not lammps_match.empty:
                lammps_row = lammps_match.iloc[0]
                df.at[idx, 'lammps_simulated'] = True
                df.at[idx, 'lammps_success'] = lammps_row['lammps_success']
                
                if lammps_row['lammps_success']:
                    df.at[idx, 'atomistic_flux_lmh'] = lammps_row.get('estimated_flux_rate', np.nan)
                    df.at[idx, 'simulation_temperature_k'] = lammps_row.get('final_temperature', np.nan)
                    df.at[idx, 'simulation_pressure_bar'] = lammps_row.get('final_pressure', np.nan)
        
        return df
    
    def _calculate_derived_properties(self, df):
        """Calculate additional derived properties."""
        
        # Performance metrics
        df['flux_per_pressure'] = df['flux_lmh'] / df['pressure_bar']
        df['rejection_efficiency'] = df['rejection_percent'] / 100.0
        
        # Performance score (higher is better)
        df['performance_score'] = (
            df['flux_per_pressure'] * df['rejection_efficiency']
        )
        
        # Merge empirical and atomistic results as specified in instructions
        if 'atomistic_flux_lmh' in df.columns:
            # If atomistic simulation was successful, use simulated flux
            df['unified_flux_lmh'] = df.apply(lambda row: 
                row['atomistic_flux_lmh'] if pd.notna(row['atomistic_flux_lmh']) 
                else row['flux_lmh'], axis=1)
            
            # LAMMPS vs Theoretical comparison
            df['atomistic_empirical_ratio'] = df['atomistic_flux_lmh'] / df['flux_lmh']
        else:
            df['unified_flux_lmh'] = df['flux_lmh']
        
        # Material efficiency ranking
        df['material_rank'] = df.groupby('material')['performance_score'].rank(ascending=False)
        
        return df
    
    def generate_summary_statistics(self):
        """
        Generate summary statistics across all phases.
        
        Returns:
            dict: Summary statistics
        """
        if self.unified_data is None:
            self.create_unified_dataset()
        
        df = self.unified_data
        
        summary = {
            'total_simulations': len(df),
            'materials_tested': df['material'].nunique(),
            'pressure_range': {
                'min': df['pressure_bar'].min(),
                'max': df['pressure_bar'].max()
            },
            'flux_statistics': {
                'mean': df['flux_lmh'].mean(),
                'std': df['flux_lmh'].std(),
                'min': df['flux_lmh'].min(),
                'max': df['flux_lmh'].max()
            },
            'rejection_statistics': {
                'mean': df['rejection_percent'].mean(),
                'std': df['rejection_percent'].std(),
                'min': df['rejection_percent'].min(),
                'max': df['rejection_percent'].max()
            }
        }
        
        # Phase-specific statistics
        if 'structure_assigned' in df.columns:
            summary['phase2_coverage'] = df['structure_assigned'].sum() / len(df)
        
        if 'lammps_success' in df.columns:
            summary['phase3_success_rate'] = df['lammps_success'].sum() / len(df)
        
        # Top performers
        top_flux = df.nlargest(3, 'flux_lmh')[['membrane_name', 'flux_lmh']].to_dict('records')
        top_rejection = df.nlargest(3, 'rejection_percent')[['membrane_name', 'rejection_percent']].to_dict('records')
        top_performance = df.nlargest(3, 'performance_score')[['membrane_name', 'performance_score']].to_dict('records')
        
        summary['top_performers'] = {
            'highest_flux': top_flux,
            'highest_rejection': top_rejection,
            'best_overall': top_performance
        }
        
        return summary
    
    def export_unified_results(self, output_dir):
        """
        Export all unified results to files.
        
        Args:
            output_dir (str): Directory for output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.unified_data is None:
            self.create_unified_dataset()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export main dataset
        excel_path = os.path.join(output_dir, f"unified_results_{timestamp}.xlsx")
        self.unified_data.to_excel(excel_path, index=False)
        
        csv_path = os.path.join(output_dir, f"unified_results_{timestamp}.csv")
        self.unified_data.to_csv(csv_path, index=False)
        
        # Export summary statistics
        summary = self.generate_summary_statistics()
        summary_path = os.path.join(output_dir, f"summary_statistics_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return {
            'excel_file': excel_path,
            'csv_file': csv_path,
            'summary_file': summary_path
        }
    
    def filter_results(self, material=None, min_flux=None, max_pressure=None, 
                      lammps_success_only=False):
        """
        Filter unified results based on criteria.
        
        Args:
            material (str): Filter by material type
            min_flux (float): Minimum flux threshold
            max_pressure (float): Maximum pressure threshold
            lammps_success_only (bool): Only include successful LAMMPS runs
        
        Returns:
            pd.DataFrame: Filtered results
        """
        if self.unified_data is None:
            self.create_unified_dataset()
        
        df = self.unified_data.copy()
        
        if material:
            df = df[df['material'] == material]
        
        if min_flux:
            df = df[df['flux_lmh'] >= min_flux]
        
        if max_pressure:
            df = df[df['pressure_bar'] <= max_pressure]
        
        if lammps_success_only and 'lammps_success' in df.columns:
            df = df[df['lammps_success'] == True]
        
        return df
    
    def load_phase4_results(self, chemistry_engine):
        """
        Load Phase 4 (chemical/biological) simulation results.
        
        Args:
            chemistry_engine: ChemistrySimulationEngine with results
        """
        if chemistry_engine and hasattr(chemistry_engine, 'simulation_results'):
            phase4_records = []
            
            for sim_result in chemistry_engine.simulation_results:
                membrane_type = sim_result['membrane_type']
                
                for contaminant, data in sim_result['contaminants'].items():
                    # Extract removal efficiency
                    efficiency = 0
                    if 'removal_efficiency' in data:
                        efficiency = data['removal_efficiency']
                    elif 'kill_efficiency' in data:
                        efficiency = data['kill_efficiency']
                    elif 'rejection_percent' in data:
                        efficiency = max(0, data['rejection_percent'])
                    
                    record = {
                        'membrane_type': membrane_type,
                        'contaminant': contaminant,
                        'contaminant_type': data.get('type', 'unknown'),
                        'removal_efficiency_percent': efficiency,
                        'simulation_time_min': sim_result.get('reaction_time', 180),
                        'pH': sim_result.get('pH', 7.0),
                        'phase': 'Phase4_Chemical_Biological'
                    }
                    
                    # Add contaminant-specific data
                    if 'q_max' in data:
                        record['adsorption_capacity_mg_g'] = data['q_max']
                    if 'k2' in data:
                        record['kinetic_constant'] = data['k2']
                    if 'kill_log' in data:
                        record['log_reduction'] = data['kill_log']
                    
                    phase4_records.append(record)
            
            self.phase4_data = pd.DataFrame(phase4_records)
        else:
            print("Warning: No Phase 4 chemistry results provided")
            self.phase4_data = pd.DataFrame()
    
    def create_unified_dataset_with_phase4(self):
        """
        Create unified dataset including all four phases.
        """
        datasets = []
        
        # Add existing phase data
        if self.phase1_data is not None:
            phase1_marked = self.phase1_data.copy()
            phase1_marked['phase'] = 'Phase1_Macroscale'
            datasets.append(phase1_marked)
        
        if self.phase2_data is not None:
            phase2_marked = self.phase2_data.copy()
            phase2_marked['phase'] = 'Phase2_Hybrid_Structure'
            datasets.append(phase2_marked)
        
        if self.phase3_data is not None:
            phase3_marked = self.phase3_data.copy()
            phase3_marked['phase'] = 'Phase3_Atomistic'
            datasets.append(phase3_marked)
        
        if self.phase4_data is not None and not self.phase4_data.empty:
            datasets.append(self.phase4_data)
        
        if datasets:
            self.unified_data = pd.concat(datasets, ignore_index=True, sort=False)
            print(f"✅ Unified dataset created with {len(self.unified_data)} total records")
            print(f"Phases included: {self.unified_data['phase'].unique()}")
        else:
            print("⚠️ No data loaded for any phase")
            self.unified_data = pd.DataFrame()
