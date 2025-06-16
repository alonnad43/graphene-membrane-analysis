# output_parser.py

"""
Parses LAMMPS output to extract water flux, deformation, and pressure data.

# TODO: Extract pressure-vs-time plot
# TODO: Parse atomic coordinates for visual debugging
"""

import numpy as np
import pandas as pd
import os
import re

class LAMMPSOutputParser:
    """
    Parses LAMMPS simulation output files to extract physical properties.
    """
    
    def __init__(self):
        self.supported_files = ['log', 'lammpstrj', 'dat']
    
    def parse_log_file(self, log_path):
        """
        Parse LAMMPS log file to extract thermodynamic data.
        
        Args:
            log_path (str): Path to LAMMPS log file
        
        Returns:
            pd.DataFrame: Thermodynamic data with timesteps
        """
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Log file not found: {log_path}")
        
        thermo_data = []
        columns = []
        reading_data = False
        
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Look for thermo_style header
                if line.startswith('Step'):
                    columns = line.split()
                    reading_data = True
                    continue
                
                # Stop reading at end of run
                if 'Loop time' in line or line.startswith('WARNING') or line.startswith('ERROR'):
                    reading_data = False
                    continue
                
                # Parse data lines
                if reading_data and line and not line.startswith('#'):
                    try:
                        values = [float(x) for x in line.split()]
                        if len(values) == len(columns):
                            thermo_data.append(values)
                    except ValueError:
                        reading_data = False
        
        if thermo_data and columns:
            df = pd.DataFrame(thermo_data, columns=columns)
            return df
        else:
            return pd.DataFrame()
    
    def parse_flux_data(self, flux_file):
        """
        Parse flux calculation output.
        
        Args:
            flux_file (str): Path to flux data file
        
        Returns:
            dict: Flux analysis results
        """
        if not os.path.exists(flux_file):
            return {"error": f"Flux file not found: {flux_file}"}
        
        try:
            # Read flux data (assuming z-positions of water molecules)
            data = np.loadtxt(flux_file, skiprows=1)
            
            if data.size == 0:
                return {"error": "Empty flux data file"}
            
            # Calculate water flux metrics
            if data.ndim == 1:
                z_positions = data
            else:
                # Multiple columns - assume first is timestep, rest are z-positions
                z_positions = data[:, 1:].flatten()
            
            # Basic flux analysis
            z_mean = np.mean(z_positions)
            z_std = np.std(z_positions)
            z_drift = z_positions[-100:].mean() - z_positions[:100].mean() if len(z_positions) > 200 else 0
            
            # Estimate flux rate (simplified)
            flux_rate = abs(z_drift) * 1000  # Convert to approximate flux units
            
            return {
                "mean_z_position": z_mean,
                "z_position_std": z_std,
                "z_drift": z_drift,
                "estimated_flux_rate": flux_rate,
                "n_water_measurements": len(z_positions)
            }
            
        except Exception as e:
            return {"error": f"Error parsing flux data: {str(e)}"}
    
    def parse_trajectory_file(self, traj_path, frame_skip=10):
        """
        Parse LAMMPS trajectory file to extract atomic positions.
        
        Args:
            traj_path (str): Path to trajectory file
            frame_skip (int): Only read every N-th frame
        
        Returns:
            dict: Trajectory analysis summary
        """
        if not os.path.exists(traj_path):
            return {"error": f"Trajectory file not found: {traj_path}"}
        
        try:
            frame_count = 0
            atom_counts = []
            box_sizes = []
            
            with open(traj_path, 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Look for frame headers
                if line == "ITEM: TIMESTEP":
                    frame_count += 1
                    
                    # Skip frames if needed
                    if frame_count % frame_skip != 0:
                        i += 1
                        continue
                    
                    # Read number of atoms
                    while i < len(lines) and not lines[i].strip() == "ITEM: NUMBER OF ATOMS":
                        i += 1
                    if i + 1 < len(lines):
                        n_atoms = int(lines[i + 1].strip())
                        atom_counts.append(n_atoms)
                    
                    # Read box bounds
                    while i < len(lines) and not lines[i].strip().startswith("ITEM: BOX BOUNDS"):
                        i += 1
                    if i + 3 < len(lines):
                        # Parse box dimensions
                        x_bounds = [float(x) for x in lines[i + 1].strip().split()[:2]]
                        y_bounds = [float(x) for x in lines[i + 2].strip().split()[:2]]
                        z_bounds = [float(x) for x in lines[i + 3].strip().split()[:2]]
                        
                        box_volume = (x_bounds[1] - x_bounds[0]) * (y_bounds[1] - y_bounds[0]) * (z_bounds[1] - z_bounds[0])
                        box_sizes.append(box_volume)
                
                i += 1
            
            return {
                "total_frames": frame_count,
                "frames_analyzed": len(atom_counts),
                "avg_atoms_per_frame": np.mean(atom_counts) if atom_counts else 0,
                "avg_box_volume": np.mean(box_sizes) if box_sizes else 0,
                "box_volume_change": (box_sizes[-1] - box_sizes[0]) / box_sizes[0] if len(box_sizes) > 1 else 0
            }
            
        except Exception as e:
            return {"error": f"Error parsing trajectory: {str(e)}"}
    
    def extract_membrane_properties(self, simulation_dir):
        """
        Extract all relevant properties from a complete simulation.
        
        Args:
            simulation_dir (str): Directory containing simulation outputs
        
        Returns:
            dict: Complete analysis results
        """
        results = {
            "simulation_dir": simulation_dir,
            "thermodynamics": {},
            "flux_analysis": {},
            "trajectory_summary": {},
            "success": False
        }
        
        # Find output files
        log_files = [f for f in os.listdir(simulation_dir) if f.endswith('.log')]
        flux_files = [f for f in os.listdir(simulation_dir) if f.endswith('_flux.dat')]
        traj_files = [f for f in os.listdir(simulation_dir) if f.endswith('.lammpstrj')]
        
        try:
            # Parse log file (production run preferred)
            prod_log = next((f for f in log_files if 'prod' in f or 'production' in f), None)
            if not prod_log and log_files:
                prod_log = log_files[0]  # Use any available log
            
            if prod_log:
                log_path = os.path.join(simulation_dir, prod_log)
                thermo_df = self.parse_log_file(log_path)
                if not thermo_df.empty:
                    results["thermodynamics"] = {
                        "final_temperature": thermo_df['Temp'].iloc[-1] if 'Temp' in thermo_df.columns else None,
                        "final_pressure": thermo_df['Press'].iloc[-1] if 'Press' in thermo_df.columns else None,
                        "avg_potential_energy": thermo_df['PotEng'].mean() if 'PotEng' in thermo_df.columns else None,
                        "final_volume": thermo_df['Volume'].iloc[-1] if 'Volume' in thermo_df.columns else None,
                        "simulation_steps": len(thermo_df)
                    }
            
            # Parse flux data
            if flux_files:
                flux_path = os.path.join(simulation_dir, flux_files[0])
                results["flux_analysis"] = self.parse_flux_data(flux_path)
            
            # Parse trajectory
            if traj_files:
                traj_path = os.path.join(simulation_dir, traj_files[0])
                results["trajectory_summary"] = self.parse_trajectory_file(traj_path)
            
            results["success"] = True
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def batch_analysis(self, base_dir):
        """
        Analyze multiple simulation directories.
        
        Args:
            base_dir (str): Base directory containing simulation subdirectories
        
        Returns:
            dict: Results for each simulation
        """
        batch_results = {}
        
        if not os.path.exists(base_dir):
            return {"error": f"Base directory not found: {base_dir}"}
        
        # Find simulation directories
        sim_dirs = [d for d in os.listdir(base_dir) 
                   if os.path.isdir(os.path.join(base_dir, d))]
        
        for sim_dir in sim_dirs:
            full_path = os.path.join(base_dir, sim_dir)
            batch_results[sim_dir] = self.extract_membrane_properties(full_path)
        
        return batch_results

def calculate_water_permeability(flux_analysis, membrane_thickness_nm, pressure_bar):
    """
    Calculate water permeability from simulation results.
    
    Args:
        flux_analysis (dict): Results from flux analysis
        membrane_thickness_nm (float): Membrane thickness
        pressure_bar (float): Applied pressure
    
    Returns:
        dict: Permeability calculations
    """
    if "estimated_flux_rate" not in flux_analysis:
        return {"error": "No flux rate data available"}
    
    flux_rate = flux_analysis["estimated_flux_rate"]
    
    # Convert units and calculate permeability
    # Simplified calculation - actual implementation would need proper unit conversions
    permeability = flux_rate * membrane_thickness_nm / pressure_bar
    
    return {
        "water_permeability": permeability,
        "flux_rate": flux_rate,
        "membrane_thickness": membrane_thickness_nm,
        "applied_pressure": pressure_bar,
        "units": "arbitrary (needs proper unit conversion)"
    }

def parse_water_flux(dump_path, membrane_area_nm2=100.0, simulation_time_ns=50.0):
    """
    Parse water flux from LAMMPS dump file by counting water molecules crossing membrane.
    
    Args:
        dump_path (str): Path to LAMMPS dump.xyz file
        membrane_area_nm2 (float): Membrane area in nm²
        simulation_time_ns (float): Simulation time in nanoseconds
        
    Returns:
        dict: Flux analysis results
    """
    if not os.path.exists(dump_path):
        return {"error": f"Dump file not found: {dump_path}", "flux_lmh": 0.0}
    
    try:
        # Read dump file and count water molecules crossing z-threshold
        water_crossings = count_z_crossings(dump_path, z_threshold=0.0)
        
        # Calculate flux using physical constants
        water_molar_volume_l = 0.018015  # L/mol at 300K
        avogadro = 6.02214076e23
        area_m2 = membrane_area_nm2 * 1e-18  # nm² to m²
        sim_time_hr = simulation_time_ns / 3.6e12  # ns to hours
        
        # Flux calculation: crossings → moles → liters → flux rate
        moles_crossed = water_crossings / avogadro
        volume_crossed_l = moles_crossed * water_molar_volume_l
        flux_lmh = volume_crossed_l / (area_m2 * sim_time_hr)
        
        return {
            "flux_lmh": flux_lmh,
            "water_crossings": water_crossings,
            "simulation_time_ns": simulation_time_ns,
            "membrane_area_nm2": membrane_area_nm2
        }
        
    except Exception as e:
        return {"error": str(e), "flux_lmh": 0.0}

def count_z_crossings(dump_path, z_threshold=0.0):
    """
    Count water molecules crossing z-threshold between frames.
    
    Args:
        dump_path (str): Path to dump file
        z_threshold (float): Z-coordinate threshold for crossing detection
        
    Returns:
        int: Number of water molecule crossings
    """
    crossings = 0
    previous_z_positions = {}
    
    try:
        with open(dump_path, 'r') as f:
            current_frame = []
            reading_atoms = False
            
            for line in f:
                line = line.strip()
                
                if line.startswith('ITEM: TIMESTEP'):
                    # Process previous frame if exists
                    if current_frame:
                        crossings += analyze_frame_crossings(current_frame, previous_z_positions, z_threshold)
                    current_frame = []
                    reading_atoms = False
                    
                elif line.startswith('ITEM: ATOMS'):
                    reading_atoms = True
                    
                elif reading_atoms and line and not line.startswith('ITEM:'):
                    # Parse atom data: ID type x y z ...
                    parts = line.split()
                    if len(parts) >= 5:
                        atom_id = int(parts[0])
                        atom_type = int(parts[1])
                        z_pos = float(parts[4])
                        
                        # Only track water oxygen atoms (type 4 in typical setup)
                        if atom_type == 4:
                            current_frame.append((atom_id, z_pos))
            
            # Process final frame
            if current_frame:
                crossings += analyze_frame_crossings(current_frame, previous_z_positions, z_threshold)
                
    except Exception as e:
        print(f"Error reading dump file: {e}")
        return 0
    
    return crossings

def analyze_frame_crossings(current_frame, previous_z_positions, z_threshold):
    """
    Analyze crossings in current frame compared to previous positions.
    
    Args:
        current_frame (list): List of (atom_id, z_pos) tuples
        previous_z_positions (dict): Previous z-positions by atom_id
        z_threshold (float): Crossing threshold
        
    Returns:
        int: Number of crossings detected in this frame
    """
    crossings = 0
    
    for atom_id, z_pos in current_frame:
        if atom_id in previous_z_positions:
            prev_z = previous_z_positions[atom_id]
            
            # Check for crossing
            if (prev_z < z_threshold <= z_pos) or (prev_z > z_threshold >= z_pos):
                crossings += 1
        
        # Update position for next frame
        previous_z_positions[atom_id] = z_pos
    
    return crossings

def parse_realistic_lammps_output(sim_dir):
    """
    Parse all LAMMPS output files to extract realistic physical properties.
    
    Args:
        sim_dir (str): Simulation directory containing LAMMPS output files
        
    Returns:
        dict: Comprehensive simulation results
    """
    results = {
        "water_flux_lmh": 0.0,
        "youngs_modulus_gpa": None,
        "ultimate_strength_mpa": None,
        "contact_angle_deg": None,
        "final_temperature": None,
        "final_pressure": None,
        "simulation_success": False
    }
    
    try:
        # Parse water flux from trajectory
        dump_file = os.path.join(sim_dir, "dump.xyz")
        if os.path.exists(dump_file):
            flux_results = parse_water_flux(dump_file)
            results["water_flux_lmh"] = flux_results.get("flux_lmh", 0.0)
        
        # Parse thermodynamic data from log
        log_file = os.path.join(sim_dir, "log.lammps")
        if os.path.exists(log_file):
            thermo_data = extract_final_thermodynamics(log_file)
            results.update(thermo_data)
        
        # Mark as successful if we got basic results
        if results["water_flux_lmh"] > 0 or results["final_temperature"] is not None:
            results["simulation_success"] = True
            
    except Exception as e:
        results["error"] = str(e)
    
    return results

def extract_final_thermodynamics(log_path):
    """
    Extract final thermodynamic properties from LAMMPS log file.
    
    Args:
        log_path (str): Path to LAMMPS log file
        
    Returns:
        dict: Final thermodynamic properties
    """
    results = {}
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            
            # Extract final values using regex
            temp_match = re.search(r'Final temperature:\s*([\d.]+)', content)
            if temp_match:
                results["final_temperature"] = float(temp_match.group(1))
            
            press_match = re.search(r'Final pressure:\s*([\d.-]+)', content)
            if press_match:
                results["final_pressure"] = float(press_match.group(1))
                
            pe_match = re.search(r'Final potential energy:\s*([\d.-]+)', content)
            if pe_match:
                results["final_potential_energy"] = float(pe_match.group(1))
                
    except Exception as e:
        results["parse_error"] = str(e)
    
    return results
