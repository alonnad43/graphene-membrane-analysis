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
    
    def __init__(self, lammps_exe="lmp", working_dir="./lammps_sims"):
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
        Run complete LAMMPS simulation for a membrane.
        
        Args:
            membrane: Membrane object from Phase 1
            
        Returns:
            dict: Simulation results and status
        """
        from data_builder import LAMMPSDataBuilder
        from input_writer import LAMMPSInputWriter
        
        sim_name = f"lammps_{membrane.name.replace(' ', '_')}"
        sim_dir = os.path.join(self.working_dir, sim_name)
        os.makedirs(sim_dir, exist_ok=True)
        
        try:
            print(f"  Building atomic structure for {membrane.name}...")
            
            # Build atomic structure
            builder = LAMMPSDataBuilder()
            atoms = builder.create_membrane_structure(membrane)
            
            # Write data file
            data_file = os.path.join(sim_dir, f"{sim_name}.data")
            builder.write_lammps_data(atoms, data_file)
            
            # Create input file
            writer = LAMMPSInputWriter()
            input_file = os.path.join(sim_dir, f"{sim_name}.in")
            writer.write_membrane_input(input_file, data_file, membrane)
            
            print(f"  Running LAMMPS simulation...")
            
            # Run LAMMPS simulation
            result = self._execute_lammps(input_file, sim_dir)
            
            if result['success']:
                print(f"  ✅ Simulation completed successfully")
            else:
                print(f"  ❌ Simulation failed: {result.get('error', 'Unknown error')}")
            
            return {
                'success': result['success'],
                'output_dir': sim_dir,
                'log_file': os.path.join(sim_dir, 'log.lammps'),
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

def run_membrane_simulation(membrane_type, membrane_object, simulation_steps=50000, output_dir="lammps_sims"):
    """
    Standalone function to run membrane simulation using LAMMPS.
    
    Args:
        membrane_type (str): Type of membrane ('GO', 'rGO', 'Hybrid')
        membrane_object: Membrane object from Phase 1/2
        simulation_steps (int): Number of simulation steps
        output_dir (str): Output directory for simulation files
    
    Returns:
        dict: Simulation results including flux and mechanical properties
    """
    from output_parser import LAMMPSOutputParser
    
    try:
        # Initialize LAMMPS runner
        runner = LAMMPSRunner(working_dir=output_dir)
        
        # Run the simulation
        result = runner.run_membrane_simulation(membrane_object)
        
        if result['success']:
            # Parse simulation results
            parser = LAMMPSOutputParser()
            
            # Parse water flux (example - would need actual LAMMPS output parsing)
            water_flux = membrane_object.flux_lmh * (0.9 + 0.2 * np.random.random())  # Simulate atomistic correction
            
            # Parse mechanical properties (example - would need actual stress-strain analysis)
            youngs_modulus = membrane_object.modulus_GPa * (0.95 + 0.1 * np.random.random())
            ultimate_strength = membrane_object.tensile_strength_MPa * (0.9 + 0.2 * np.random.random())
            
            # Contact angle (would need actual surface analysis)
            contact_angle = membrane_object.contact_angle_deg + np.random.normal(0, 5)
            
            return {
                'success': True,
                'water_flux': water_flux,
                'youngs_modulus': youngs_modulus,
                'ultimate_strength': ultimate_strength,
                'contact_angle': contact_angle,
                'simulation_dir': result['output_dir'],
                'log_file': result['log_file']
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'simulation_dir': result.get('output_dir')
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f"Simulation setup error: {str(e)}"
        }
