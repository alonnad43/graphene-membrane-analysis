# lammps_runner.py

"""
Phase 3: Runs atomistic LAMMPS simulation from a hybrid membrane.

# TODO: Support multiple runs in pressure sweep mode
# TODO: Add success/failure logging per job
"""

import os
import subprocess
import numpy as np
from datetime import datetime

class LAMMPSRunner:
    """
    Manages LAMMPS simulation execution for membrane systems.
    
    Attributes:
        lammps_exe (str): Path to LAMMPS executable
        working_dir (str): Directory for simulation files
        log_file (str): Path to log file
    """
    
    def __init__(self, lammps_exe="lmp", working_dir="./output/phase3"):
        self.lammps_exe = lammps_exe
        self.working_dir = working_dir
        self.log_file = os.path.join(working_dir, "lammps_runner.log")
        os.makedirs(working_dir, exist_ok=True)
    
    def run_simulation(self, input_file, data_file, output_prefix="sim"):
        """
        Execute a single LAMMPS simulation.
        
        Args:
            input_file (str): Path to LAMMPS input file
            data_file (str): Path to LAMMPS data file
            output_prefix (str): Prefix for output files
        
        Returns:
            dict: Simulation results and status
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_dir = os.path.join(self.working_dir, f"{output_prefix}_{timestamp}")
        os.makedirs(sim_dir, exist_ok=True)
        
        # Copy input files to simulation directory
        import shutil
        local_input = os.path.join(sim_dir, "input.in")
        local_data = os.path.join(sim_dir, "data.lammps")
        shutil.copy(input_file, local_input)
        shutil.copy(data_file, local_data)
        
        # Run LAMMPS
        cmd = [self.lammps_exe, "-in", "input.in", "-log", "lammps.log"]
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=sim_dir, 
                capture_output=True, 
                text=True, 
                timeout=3600  # 1 hour timeout
            )
            
            success = result.returncode == 0
            
            simulation_result = {
                "success": success,
                "timestamp": timestamp,
                "sim_dir": sim_dir,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
            self._log_simulation(simulation_result)
            return simulation_result
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Simulation timeout (1 hour)",
                "timestamp": timestamp,
                "sim_dir": sim_dir
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": timestamp,
                "sim_dir": sim_dir
            }
    
    def run_pressure_sweep(self, input_template, data_file, pressures, output_prefix="sweep"):
        """
        Run multiple simulations across different pressures.
        
        Args:
            input_template (str): Template input file with pressure placeholder
            data_file (str): Path to LAMMPS data file
            pressures (list): List of pressures to simulate
            output_prefix (str): Prefix for output files
        
        Returns:
            list: List of simulation results
        """
        results = []
        
        for i, pressure in enumerate(pressures):
            # Create pressure-specific input file
            with open(input_template, 'r') as f:
                template_content = f.read()
            
            input_content = template_content.replace("PRESSURE_PLACEHOLDER", str(pressure))
            pressure_input = os.path.join(self.working_dir, f"input_P{pressure:.1f}.in")
            
            with open(pressure_input, 'w') as f:
                f.write(input_content)
            
            # Run simulation
            sim_prefix = f"{output_prefix}_P{pressure:.1f}"
            result = self.run_simulation(pressure_input, data_file, sim_prefix)
            result["pressure"] = pressure
            results.append(result)
            
            print(f"Completed simulation {i+1}/{len(pressures)} at P={pressure:.1f} bar")
        
        return results
    
    def run_membrane_simulation(self, membrane):
        """
        Run complete LAMMPS simulation for a membrane using realistic molecular dynamics.
        
        Args:
            membrane: Membrane object from Phase 1
            
        Returns:
            dict: Simulation results and status
        """
        from src.data_builder import LAMMPSDataBuilder
        from src.input_writer import LAMMPSInputWriter
        import os
        
        sim_name = f"lammps_{membrane.name.replace(' ', '_')}"
        sim_dir = os.path.join(self.working_dir, sim_name)
        os.makedirs(sim_dir, exist_ok=True)
        
        try:
            print(f"  Building atomic structure for {membrane.name}...")
            
            # Build atomic structure
            builder = LAMMPSDataBuilder()
            atoms = builder.create_membrane_structure(membrane)
            
            # Use a consistent data file name
            data_file = os.path.join(sim_dir, "data.lammps")
            builder.write_lammps_data(atoms, data_file)
            
            # Create realistic LAMMPS input file (based on Schmidt et al. [17])
            writer = LAMMPSInputWriter()
            input_file = os.path.join(sim_dir, "input.in")
            writer.write_realistic_membrane_input(input_file, data_file, membrane)
            
            # Check that both files exist before running LAMMPS
            if not os.path.isfile(data_file):
                raise FileNotFoundError(f"LAMMPS data file not found: {data_file}")
            if not os.path.isfile(input_file):
                raise FileNotFoundError(f"LAMMPS input file not found: {input_file}")
            
            print(f"  Running LAMMPS simulation...")
            
            # Run LAMMPS simulation
            result = self._execute_lammps(input_file, sim_dir)
            
            if result['success']:
                print(f"  ✅ Simulation completed successfully")
                
                # Parse realistic results from LAMMPS output
                from src.output_parser import parse_realistic_lammps_output
                parsed_results = parse_realistic_lammps_output(sim_dir)
                result.update(parsed_results)
                
            else:
                print(f"  ❌ Simulation failed: {result.get('error', 'Unknown error')}")
            
            return {
                'success': result['success'],
                'output_dir': sim_dir,
                'log_file': os.path.join(sim_dir, 'log.lammps'),
                'water_flux': result.get('water_flux_lmh', None),
                'youngs_modulus': result.get('youngs_modulus_gpa', None),
                'ultimate_strength': result.get('ultimate_strength_mpa', None),
                'contact_angle': result.get('contact_angle_deg', None),
                'error': result.get('error')
            }
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'output_dir': sim_dir
            }
    
    def _execute_lammps(self, input_file, sim_dir):
        """
        Execute LAMMPS simulation using Python interface.
        """
        try:
            from lammps import lammps
            
            # Change to simulation directory
            original_dir = os.getcwd()
            os.chdir(sim_dir)
            
            try:
                # Initialize LAMMPS
                lmp = lammps()
                
                # Run input file
                lmp.file(os.path.basename(input_file))
                
                # Close LAMMPS
                lmp.close()
                
                return {'success': True}
                
            except Exception as e:
                return {'success': False, 'error': f"LAMMPS execution error: {str(e)}"}
            finally:
                os.chdir(original_dir)
                
        except ImportError:
            return {'success': False, 'error': 'LAMMPS Python interface not available'}
    
    def _log_simulation(self, result):
        """Log simulation results to file."""
        with open(self.log_file, 'a') as f:
            f.write(f"\n=== Simulation {result['timestamp']} ===\n")
            f.write(f"Success: {result['success']}\n")
            f.write(f"Directory: {result['sim_dir']}\n")
            if not result['success']:
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                f.write(f"Return code: {result.get('returncode', 'N/A')}\n")
            f.write("=" * 50 + "\n")

def check_lammps_installation():
    """
    Check if LAMMPS is available and return version info.
    
    Returns:
        dict: Installation status and version info
    """
    try:
        result = subprocess.run(["lmp", "-help"], capture_output=True, text=True, timeout=10)
        return {
            "available": True,
            "version_info": result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout
        }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {
            "available": False,
            "message": "LAMMPS not found. Please install LAMMPS and ensure 'lmp' is in PATH."
        }
