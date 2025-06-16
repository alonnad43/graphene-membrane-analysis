# simulate_chemistry.py

"""
Phase 4: Chemical and Biological Simulation Engine

Calculates contaminant removal based on chemical properties of membranes and pollutants.
Considers adsorption kinetics, equilibrium saturation, regeneration capacity, and contaminant decay.

Supported interactions:
- Heavy metal adsorption (Pb2+, As3+, etc.)
- Bacterial inactivation (E. coli, etc.)
- Salt rejection (NaCl, MgSO4, etc.)
- Chemical bonding types (π–π, redox, electrostatic, etc.)
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

class ChemicalSimulationEngine:
    """
    Main engine for processing contaminant-membrane interactions.
    """
    
    def __init__(self, contaminant_data_path="data/contaminant_data.json"):
        self.contaminant_data_path = contaminant_data_path
        self.contaminant_data = {}
        self.membrane_profiles = {}
        self.simulation_results = []
        self.load_contaminant_data()
    
    def load_contaminant_data(self):
        """Load contaminant database from JSON file."""
        try:
            if os.path.exists(self.contaminant_data_path):
                with open(self.contaminant_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                  # Handle new comprehensive JSON structure
                if 'contaminants' in data and 'membrane_types' in data:
                    # Flatten the contaminants structure for backward compatibility
                    flat_contaminants = {}
                    for category, contaminants in data['contaminants'].items():
                        for name, props in contaminants.items():
                            # Add backward compatibility fields
                            flat_props = props.copy()
                            flat_props['type'] = category
                            
                            # Handle membrane affinity vs direct membrane data
                            if 'membrane_affinity' in props:
                                flat_props['membranes'] = props['membrane_affinity']
                            elif 'membranes' in props:
                                flat_props['membranes'] = props['membranes']
                            else:
                                # Create default membrane data for pathogens
                                if category == 'pathogens':
                                    flat_props['membranes'] = {
                                        'GO': {'kill_log': 4, 'exposure_time_min': 90},
                                        'rGO': {'kill_log': 5, 'exposure_time_min': 60},
                                        'hybrid': {'kill_log': 6, 'exposure_time_min': 45}
                                    }
                                else:
                                    flat_props['membranes'] = {}
                            
                            flat_contaminants[name] = flat_props
                    
                    self.contaminant_data = flat_contaminants
                    self.membrane_profiles = data['membrane_types']
                    print(f"Loaded {len(self.contaminant_data)} contaminants from comprehensive format")
                    
                elif 'contaminants' in data and 'membrane_profiles' in data:
                    # Old format with separate contaminants and membrane_profiles
                    self.contaminant_data = data['contaminants']
                    self.membrane_profiles = data['membrane_profiles']
                    print(f"Loaded {len(self.contaminant_data)} contaminants and {len(self.membrane_profiles)} membrane profiles")
                else:
                    # Legacy format - contaminants at top level
                    self.contaminant_data = data
                    self.membrane_profiles = {}
                    print(f"Loaded {len(self.contaminant_data)} contaminants from legacy format")
                    
            else:
                print(f"Warning: {self.contaminant_data_path} not found. Using default data.")
                self._create_default_contaminant_data()
        except Exception as e:
            print(f"Error loading contaminant data: {e}")
            self._create_default_contaminant_data()
    
    def _create_default_contaminant_data(self):
        """Create default contaminant data if file doesn't exist."""
        self.contaminant_data = {
            "Pb2+": {
                "type": "heavy metal",
                "q_max": 250.0,
                "kinetic_model": "pseudo_second_order",
                "k2": 0.0041,
                "pH_range": [4.5, 7.0],
                "interaction": ["electrostatic", "complexation"],
                "competitive_index": 3.2,
                "regenerable": True,
                "max_cycles": 5,
                "membranes": {
                    "GO": {"q_max": 220, "k2": 0.0035},
                    "rGO": {"q_max": 180, "k2": 0.0028},
                    "hybrid": {"q_max": 250, "k2": 0.0041}
                },
                "regeneration_efficiency": 0.85,
                "regeneration_cycles": 3,
                "reaction_time": 120,
                "source": "H9w7okUgoCWPFTNkkbfUdf, 2024"
            },
            "E_coli": {
                "type": "bacteria",
                "kill_log": 5,
                "exposure_time": 60,
                "competitive_index": 1.5,
                "regenerable": True,
                "max_cycles": 8,
                "membranes": {
                    "GO": {"kill_log": 4, "exposure_time": 90},
                    "rGO": {"kill_log": 5, "exposure_time": 60},
                    "hybrid": {"kill_log": 6, "exposure_time": 45}
                },
                "mechanism": ["oxidative_stress", "membrane_damage"],
                "regeneration_efficiency": 0.90,
                "regeneration_cycles": 5,
                "source": "schmidt-et-al-2023, 2023"
            }
        }
        
        self.membrane_profiles = {
            "GO": {
                "name": "Graphene Oxide",
                "functional_groups": ["hydroxyl", "epoxy", "carboxyl"],
                "interlayer_spacing": {"dry": 0.8, "wet": 1.2},
                "surface_charge": -42.5,
                "porosity": 0.15
            },
            "rGO": {
                "name": "Reduced Graphene Oxide", 
                "functional_groups": ["hydroxyl", "carboxyl"],
                "interlayer_spacing": {"dry": 0.35, "wet": 0.45},
                "surface_charge": -28.1,
                "porosity": 0.25
            },
            "hybrid": {
                "name": "GO-rGO Hybrid",
                "functional_groups": ["hydroxyl", "epoxy", "carboxyl"],
                "interlayer_spacing": {"dry": 0.6, "wet": 0.85},
                "surface_charge": -35.3,
                "porosity": 0.20
            }        }
    
    def calculate_thermodynamic_favorability(self, contaminant_name, membrane_type, temperature_K=298.15):
        """
        Calculate thermodynamic favorability of adsorption using Gibbs free energy.
        
        Args:
            contaminant_name (str): Name of contaminant
            membrane_type (str): Membrane type
            temperature_K (float): Temperature in Kelvin
            
        Returns:
            dict: Thermodynamic analysis results
        """
        contaminant_data = self.contaminant_data.get(contaminant_name, {})
        membrane_params = contaminant_data.get('membranes', {}).get(membrane_type, {})
        
        # Check for thermodynamic data
        thermo_data = membrane_params.get('thermodynamics', {})
        
        if thermo_data:
            delta_G = thermo_data.get('delta_G_kJ_mol')
            delta_H = thermo_data.get('delta_H_kJ_mol')
            delta_S = thermo_data.get('delta_S_J_mol_K')
            
            # Calculate thermodynamic favorability
            favorability = {
                'delta_G_kJ_mol': delta_G,
                'delta_H_kJ_mol': delta_H,
                'delta_S_J_mol_K': delta_S,
                'spontaneous': delta_G < 0 if delta_G else None,
                'exothermic': delta_H < 0 if delta_H else None,
                'entropy_favorable': delta_S > 0 if delta_S else None
            }
            
            # Temperature dependency (if we have ΔH and ΔS)
            if delta_H is not None and delta_S is not None:
                # ΔG = ΔH - TΔS
                delta_G_calc = delta_H - (temperature_K * delta_S / 1000)  # Convert J to kJ
                favorability['delta_G_calculated'] = delta_G_calc
                favorability['temperature_dependent'] = True
            
            return favorability
        
        return {'thermodynamic_data_available': False}
    
    def apply_diffusion_effects(self, contaminant_name, membrane_type, concentration_profile, time_points):
        """
        Apply diffusion limitations to concentration profiles.
        
        Args:
            contaminant_name (str): Name of contaminant
            membrane_type (str): Membrane type
            concentration_profile (np.array): Initial concentration profile
            time_points (np.array): Time points
            
        Returns:
            np.array: Modified concentration profile with diffusion effects
        """
        contaminant_data = self.contaminant_data.get(contaminant_name, {})
        membrane_params = contaminant_data.get('membranes', {}).get(membrane_type, {})
        
        # Get effective diffusion coefficient
        D_eff = membrane_params.get('effective_diffusion_coefficient_m2s')
        
        if D_eff:
            # Apply Fick's law diffusion correction
            # Simplified: exponential approach to equilibrium
            diffusion_factor = 1 - np.exp(-D_eff * time_points * 1e9)  # Scale factor for minutes
            modified_profile = concentration_profile * diffusion_factor
            
            return modified_profile, D_eff
        
        return concentration_profile, None
    
    def simulate_contaminant_removal(self, membrane_type, contaminants, initial_concentrations, 
                                   reaction_time=180, timestep=1.0, pH=6.5):
        """
        Simulate contaminant removal for a specific membrane type.
        
        Args:
            membrane_type (str): 'GO', 'rGO', or 'hybrid'
            contaminants (list): List of contaminant names
            initial_concentrations (dict): Initial concentrations {contaminant: mg/L}
            reaction_time (float): Total simulation time in minutes
            timestep (float): Time step in minutes
            pH (float): Solution pH
            
        Returns:
            dict: Simulation results with time series data
        """
        print(f"\n🧪 Starting Phase 4 Chemical Simulation")
        print(f"Membrane: {membrane_type}")
        print(f"Contaminants: {contaminants}")
        print(f"Reaction time: {reaction_time} min")
        
        # Initialize time array
        time_points = np.arange(0, reaction_time + timestep, timestep)
        results = {
            'time_min': time_points,
            'membrane_type': membrane_type,
            'pH': pH,
            'contaminants': {}
        }
        
        # Process each contaminant
        for contaminant in contaminants:
            if contaminant not in self.contaminant_data:
                print(f"Warning: {contaminant} not found in database. Skipping.")
                continue
            
            print(f"  Processing {contaminant}...")
            contaminant_result = self._simulate_single_contaminant(
                contaminant, membrane_type, initial_concentrations.get(contaminant, 100.0),
                time_points, pH
            )
            results['contaminants'][contaminant] = contaminant_result
        
        # Store results
        self.simulation_results.append(results)
        print(f"✅ Phase 4 simulation completed for {membrane_type}")
        
        return results
    
    def _simulate_single_contaminant(self, contaminant_name, membrane_type, initial_conc, 
                                   time_points, pH):
        """
        Simulate removal kinetics for a single contaminant.
        
        Args:
            contaminant_name (str): Name of contaminant
            membrane_type (str): Membrane type
            initial_conc (float): Initial concentration (mg/L)
            time_points (np.array): Time points for simulation
            pH (float): Solution pH
              Returns:
            dict: Time series results for this contaminant
        """
        data = self.contaminant_data[contaminant_name]
        contaminant_type = data['type']
        
        # Get membrane-specific parameters
        membrane_params = data['membranes'].get(membrane_type, {})
        
        if contaminant_type in ['heavy metal', 'heavy_metal']:
            return self._simulate_adsorption(data, membrane_params, initial_conc, time_points, pH, membrane_type)
        elif contaminant_type == 'bacteria':
            return self._simulate_bacterial_inactivation(data, membrane_params, initial_conc, time_points)
        elif contaminant_type == 'virus':
            return self._simulate_bacterial_inactivation(data, membrane_params, initial_conc, time_points)  # Same as bacteria
        elif contaminant_type == 'salt':
            return self._simulate_salt_rejection(data, membrane_params, initial_conc, time_points)
        else:
            return self._simulate_generic_removal(data, membrane_params, initial_conc, time_points)
    
    def _simulate_adsorption(self, data, membrane_params, initial_conc, time_points, pH, membrane_type):
        """
        Simulate adsorption kinetics using pseudo-second-order model.
        
        Model: dq/dt = k2 * (q_max - q)^2
        """        # Get parameters
        q_max = membrane_params.get('q_max')  # mg/g
        k2 = membrane_params.get('k2')  # g/(mg·min)
        
        if q_max is None or k2 is None:
            raise ValueError(f"Missing q_max or k2 parameters for membrane {membrane_type}")
        
          # Check pH range
        pH_range = data.get('pH_range', [0, 14])
        if isinstance(pH_range, dict):
            # New format: {"min": 5.5, "max": 6.5}
            pH_min, pH_max = pH_range.get('min', 0), pH_range.get('max', 14)
        else:
            # Legacy format: [5.5, 6.5]
            pH_min, pH_max = pH_range[0], pH_range[1]
            
        if not (pH_min <= pH <= pH_max):
            print(f"    Warning: pH {pH} outside optimal range [{pH_min}, {pH_max}]")
            # Reduce efficiency outside pH range
            efficiency_factor = 0.5
            q_max *= efficiency_factor
            k2 *= efficiency_factor
        
        # Initialize arrays
        concentration = np.zeros_like(time_points)
        adsorbed = np.zeros_like(time_points)
        saturation = np.zeros_like(time_points)
        
        concentration[0] = initial_conc
        adsorbed[0] = 0.0
        
        # Simulate adsorption kinetics
        for i in range(1, len(time_points)):
            dt = time_points[i] - time_points[i-1]
            q_current = adsorbed[i-1]
            
            # Pseudo-second-order kinetics: dq/dt = k2 * (q_max - q)^2
            if q_current < q_max:
                dq_dt = k2 * (q_max - q_current)**2
                q_new = q_current + dq_dt * dt
                q_new = min(q_new, q_max)  # Don't exceed maximum capacity
            else:
                q_new = q_max
            
            adsorbed[i] = q_new
            
            # Calculate remaining concentration (simplified mass balance)
            # Assuming 1 g membrane per 1 L solution
            concentration[i] = max(0, initial_conc - q_new)
            saturation[i] = (q_new / q_max) * 100
        
        return {
            'type': 'adsorption',
            'concentration_mg_L': concentration,
            'adsorbed_mg_g': adsorbed,
            'saturation_percent': saturation,
            'removal_efficiency': ((initial_conc - concentration[-1]) / initial_conc) * 100,
            'q_max': q_max,
            'k2': k2,
            'interaction_mechanisms': data.get('interaction', []),
            'equilibrium_reached': saturation[-1] > 95
        }
    
    def _simulate_bacterial_inactivation(self, data, membrane_params, initial_cfu, time_points):
        """
        Simulate bacterial inactivation with log reduction.
        
        Model: log reduction over exposure time
        """        # Get parameters
        kill_log = membrane_params.get('kill_log')
        exposure_time_min = membrane_params.get('exposure_time_min')  # minutes
        
        if kill_log is None or exposure_time_min is None:
            raise ValueError(f"Missing kill_log or exposure_time_min parameters for bacteria")
        
        # Initialize arrays
        cfu_ml = np.zeros_like(time_points)
        log_reduction = np.zeros_like(time_points)
        
        cfu_ml[0] = initial_cfu
          # Simulate bacterial kill kinetics
        for i in range(1, len(time_points)):
            t = time_points[i]
            
            if t >= exposure_time_min:
                # Full inactivation achieved
                reduction = kill_log
            else:
                # Linear kill rate until exposure time
                reduction = (kill_log * t) / exposure_time_min
            
            log_reduction[i] = reduction
            cfu_ml[i] = initial_cfu / (10**reduction)
        
        return {
            'type': 'bacterial_inactivation',
            'cfu_ml': cfu_ml,
            'log_reduction': log_reduction,
            'kill_efficiency': ((initial_cfu - cfu_ml[-1]) / initial_cfu) * 100,
            'kill_log': kill_log,
            'exposure_time': exposure_time_min,
            'mechanisms': data.get('mechanism', []),
            'target_achieved': log_reduction[-1] >= kill_log * 0.9
        }
    
    def _simulate_salt_rejection(self, data, membrane_params, initial_conc, time_points):
        """
        Simulate salt rejection (steady-state process).
        """        # Get rejection percentage
        rejection_percent = membrane_params.get('rejection_percent')
        
        if rejection_percent is None:
            raise ValueError(f"Missing rejection_percent parameter for salt")
        
        # Handle negative rejection (permeate concentration higher than feed)
        if rejection_percent < 0:
            permeate_conc = initial_conc * (1 + abs(rejection_percent) / 100)
            actual_rejection = rejection_percent
        else:
            permeate_conc = initial_conc * (1 - rejection_percent / 100)
            actual_rejection = rejection_percent
        
        # Constant rejection over time
        concentration = np.full_like(time_points, permeate_conc)
        rejected_conc = np.full_like(time_points, initial_conc - permeate_conc)
        
        return {
            'type': 'salt_rejection',
            'permeate_concentration_mg_L': concentration,
            'rejected_concentration_mg_L': rejected_conc,
            'rejection_percent': actual_rejection,
            'mechanisms': data.get('mechanism', []),
            'steady_state': True
        }
    
    def _simulate_generic_removal(self, data, membrane_params, initial_conc, time_points):
        """
        Generic removal simulation for unknown contaminant types.
        """
        # Default to 50% removal over reaction time
        removal_efficiency = 50.0
        final_conc = initial_conc * (1 - removal_efficiency / 100)
        
        concentration = np.linspace(initial_conc, final_conc, len(time_points))
        
        return {
            'type': 'generic_removal',
            'concentration_mg_L': concentration,
            'removal_efficiency': removal_efficiency,
            'note': 'Generic removal model used - contaminant type not fully characterized'
        }
    
    def apply_regeneration(self, results, cycle_number=1):
        """
        Apply regeneration effects to membrane capacity.
        
        Args:
            results (dict): Simulation results to modify
            cycle_number (int): Current regeneration cycle
            
        Returns:
            dict: Modified results with regeneration effects
        """
        print(f"🔄 Applying regeneration cycle {cycle_number}")
        
        for contaminant_name, contaminant_result in results['contaminants'].items():
            if contaminant_name not in self.contaminant_data:
                continue
                
            data = self.contaminant_data[contaminant_name]
            regen_efficiency = data.get('regeneration_efficiency', 0.8)
            max_cycles = data.get('regeneration_cycles', 3)
            
            if cycle_number <= max_cycles:
                # Reduce capacity based on regeneration efficiency
                capacity_loss = (1 - regen_efficiency) * cycle_number
                remaining_capacity = max(0.1, 1 - capacity_loss)  # Minimum 10% capacity
                
                if 'q_max' in contaminant_result:
                    original_qmax = contaminant_result['q_max']
                    contaminant_result['q_max'] *= remaining_capacity
                    contaminant_result['regeneration_factor'] = remaining_capacity
                    print(f"  {contaminant_name}: q_max reduced from {original_qmax:.1f} to {contaminant_result['q_max']:.1f} mg/g")
            else:
                print(f"  {contaminant_name}: Maximum regeneration cycles ({max_cycles}) exceeded")
                contaminant_result['regeneration_exhausted'] = True
        
        return results
    
    def apply_regeneration(self, simulation_results, cycle_number=1):
        """
        Apply regeneration effects to reduce membrane capacity.
        
        Args:
            simulation_results (dict): Original simulation results
            cycle_number (int): Number of regeneration cycles applied
            
        Returns:
            dict: Updated simulation results with reduced capacity
        """
        # Create a copy of the original results
        import copy
        regenerated_results = copy.deepcopy(simulation_results)
          # Apply regeneration degradation to each contaminant
        for contaminant_name, contaminant_data in regenerated_results['contaminants'].items():
            if contaminant_name in self.contaminant_data:
                regen_efficiency = self.contaminant_data[contaminant_name].get('regeneration_efficiency', 0.9)
                
                # Calculate cumulative degradation factor
                degradation_factor = regen_efficiency ** cycle_number
                
                # Reduce q_max capacity
                original_qmax = contaminant_data.get('q_max', contaminant_data.get('original_q_max', 100))
                new_qmax = original_qmax * degradation_factor
                
                # Update the results
                contaminant_data['q_max'] = new_qmax
                contaminant_data['regeneration_factor'] = degradation_factor
                contaminant_data['regeneration_cycle'] = cycle_number
                contaminant_data['original_q_max'] = original_qmax
                  # Update metadata
        if 'metadata' not in regenerated_results:
            regenerated_results['metadata'] = {}
        regenerated_results['metadata']['regeneration_applied'] = True
        regenerated_results['metadata']['regeneration_cycle'] = cycle_number
        
        return regenerated_results

    def export_results(self, output_dir="output", filename_prefix="phase4_chemistry"):
        """
        Export simulation results to CSV and JSON files.
        
        Args:
            output_dir (str): Output directory path
            filename_prefix (str): Prefix for output files
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, results in enumerate(self.simulation_results):
            membrane_type = results['membrane_type']
            
            # Export time series data to CSV
            csv_data = {'time_min': results['time_min']}
            
            for contaminant, data in results['contaminants'].items():
                if 'concentration_mg_L' in data:
                    csv_data[f'{contaminant}_conc_mg_L'] = data['concentration_mg_L']
                if 'adsorbed_mg_g' in data:
                    csv_data[f'{contaminant}_adsorbed_mg_g'] = data['adsorbed_mg_g']
                if 'saturation_percent' in data:
                    csv_data[f'{contaminant}_saturation_pct'] = data['saturation_percent']
                if 'cfu_ml' in data:
                    csv_data[f'{contaminant}_cfu_ml'] = data['cfu_ml']
                if 'log_reduction' in data:
                    csv_data[f'{contaminant}_log_reduction'] = data['log_reduction']
            
            csv_df = pd.DataFrame(csv_data)
            csv_path = os.path.join(output_dir, f"{filename_prefix}_{membrane_type}_{timestamp}.csv")
            csv_df.to_csv(csv_path, index=False)
            print(f"CSV exported: {csv_path}")
            
            # Export complete results to JSON
            json_path = os.path.join(output_dir, f"{filename_prefix}_{membrane_type}_{timestamp}.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"JSON exported: {json_path}")
    
    def get_summary_statistics(self):
        """
        Generate summary statistics for all simulations.
        
        Returns:
            dict: Summary statistics across all simulations
        """
        if not self.simulation_results:
            return {"error": "No simulation results available"}
        
        summary = {
            'total_simulations': len(self.simulation_results),
            'membrane_types': [],
            'contaminant_summary': {},
            'overall_performance': {}
        }
        
        for results in self.simulation_results:
            membrane_type = results['membrane_type']
            if membrane_type not in summary['membrane_types']:
                summary['membrane_types'].append(membrane_type)
            
            for contaminant, data in results['contaminants'].items():
                if contaminant not in summary['contaminant_summary']:
                    summary['contaminant_summary'][contaminant] = {
                        'membrane_performance': {},
                        'best_membrane': None,
                        'best_efficiency': 0
                    }
                
                # Get removal efficiency
                efficiency = 0
                if 'removal_efficiency' in data:
                    efficiency = data['removal_efficiency']
                elif 'kill_efficiency' in data:
                    efficiency = data['kill_efficiency']
                elif 'rejection_percent' in data:
                    efficiency = max(0, data['rejection_percent'])
                
                summary['contaminant_summary'][contaminant]['membrane_performance'][membrane_type] = efficiency
                
                if efficiency > summary['contaminant_summary'][contaminant]['best_efficiency']:
                    summary['contaminant_summary'][contaminant]['best_efficiency'] = efficiency
                    summary['contaminant_summary'][contaminant]['best_membrane'] = membrane_type
        
        return summary

    def load_lake_victoria_parameters(self):
        """Load Lake Victoria field study parameters."""
        try:
            from properties import LAKE_VICTORIA_PARAMETERS
            self.lake_victoria_data = LAKE_VICTORIA_PARAMETERS
            print(f"✅ Lake Victoria field parameters loaded")
        except ImportError:
            print("⚠️ Lake Victoria parameters not available")
            self.lake_victoria_data = {}
    
    def apply_lake_victoria_conditions(self, membrane_type, contaminant_name, base_efficiency):
        """
        Apply Lake Victoria field study corrections to removal efficiency.
        
        Args:
            membrane_type (str): Membrane type
            contaminant_name (str): Contaminant name
            base_efficiency (float): Base removal efficiency
            
        Returns:
            float: Field-corrected efficiency
        """
        if not hasattr(self, 'lake_victoria_data'):
            self.load_lake_victoria_parameters()
        
        # Apply pathogen-specific corrections
        if contaminant_name in ['E_coli', 'Giardia', 'Salmonella', 'Rotavirus', 'Adenovirus']:
            pathogen_data = self.lake_victoria_data.get('pathogen_removal', {}).get(contaminant_name, {})
            if pathogen_data:
                # Apply tropical temperature correction (improved performance at 25-37°C)
                temp_factor = 1.1 if pathogen_data.get('optimal_temp_C', 25) > 30 else 1.0
                base_efficiency *= temp_factor
                
        # Apply PFAS-specific corrections  
        elif contaminant_name in ['PFOA', 'PFOS', 'GenX_HFPO-DA', 'PFBS']:
            pfas_data = self.lake_victoria_data.get('pfas_removal', {}).get(contaminant_name, {})
            if pfas_data and membrane_type in ['GO-CTAC', 'MAGO_Magnetic_Amine-GO']:
                field_efficiency = pfas_data.get('removal_efficiency_percent', base_efficiency)
                base_efficiency = max(base_efficiency, field_efficiency / 100.0)
                
        # Apply ion transport corrections
        elif contaminant_name in ['Na+', 'Cl-', 'Ca2+', 'Mg2+', 'SO4^2-']:
            ion_data = self.lake_victoria_data.get('ion_transport', {}).get(contaminant_name, {})
            if ion_data:
                donnan_efficiency = ion_data.get('donnan_exclusion_percent', 50) / 100.0
                base_efficiency = max(base_efficiency, donnan_efficiency)
        
        return min(base_efficiency, 1.0)  # Cap at 100%
        
def run_phase4_simulation(membrane_types=['GO', 'rGO', 'hybrid'], 
                         contaminants=['Pb2+', 'E_coli'], 
                         initial_concentrations=None,
                         reaction_time=180):
    """
    Convenience function to run Phase 4 chemical simulation.
    
    Args:
        membrane_types (list): List of membrane types to simulate
        contaminants (list): List of contaminants to test
        initial_concentrations (dict): Initial concentrations {contaminant: mg/L}
        reaction_time (float): Total reaction time in minutes
        
    Returns:
        ChemicalSimulationEngine: Engine with completed simulations
    """
    if initial_concentrations is None:
        initial_concentrations = {'Pb2+': 100.0, 'E_coli': 1e6}  # mg/L or CFU/mL
    
    print(f"\n🧪 PHASE 4: CHEMICAL AND BIOLOGICAL SIMULATION")
    print(f"==================================================")
    
    # Initialize simulation engine
    engine = ChemicalSimulationEngine()
    
    # Run simulations for each membrane type
    for membrane_type in membrane_types:
        results = engine.simulate_contaminant_removal(
            membrane_type=membrane_type,
            contaminants=contaminants,
            initial_concentrations=initial_concentrations,
            reaction_time=reaction_time
        )
    
    # Export results
    engine.export_results()
    
    # Print summary
    summary = engine.get_summary_statistics()
    print(f"\n📊 Phase 4 Summary:")
    print(f"Total simulations: {summary['total_simulations']}")
    print(f"Membrane types tested: {summary['membrane_types']}")
    
    for contaminant, info in summary['contaminant_summary'].items():
        best_membrane = info['best_membrane']
        best_efficiency = info['best_efficiency']
        print(f"{contaminant}: Best removal by {best_membrane} ({best_efficiency:.1f}%)")
    
    return engine

if __name__ == "__main__":
    # Example usage
    engine = run_phase4_simulation(
        membrane_types=['GO', 'rGO', 'hybrid'],
        contaminants=['Pb2+', 'E_coli'],
        initial_concentrations={'Pb2+': 50.0, 'E_coli': 1e5},
        reaction_time=120
    )
