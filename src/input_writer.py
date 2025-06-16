# input_writer.py

"""
Creates LAMMPS input (.in) files for MD simulations.

# TODO: Support multiple material models (GO, rGO)
# TODO: Add simulation control presets (low-pressure, fast-test, long-run)
"""

import os
from datetime import datetime

class LAMMPSInputWriter:
    """
    Generates LAMMPS input files for membrane simulations.
    """
    
    def __init__(self):
        self.templates = {
            'equilibration': self._equilibration_template(),
            'pressure_ramp': self._pressure_ramp_template(),
            'production': self._production_template()
        }
    
    def write_input_file(self, filename, simulation_type='production', **kwargs):
        """
        Write a LAMMPS input file.
        
        Args:
            filename (str): Output filename
            simulation_type (str): Type of simulation ('equilibration', 'pressure_ramp', 'production')
            **kwargs: Simulation parameters
        """
        template = self.templates.get(simulation_type, self.templates['production'])
        
        # Default parameters
        defaults = {
            'data_file': 'data.lammps',
            'timestep': 1.0,
            'temperature': 300.0,
            'pressure': 1.0,
            'run_steps': 100000,
            'dump_freq': 1000,
            'thermo_freq': 1000,
            'output_prefix': 'sim'
        }
        
        # Merge defaults with provided parameters
        params = {**defaults, **kwargs}
        
        # Replace placeholders in template
        content = template
        for key, value in params.items():
            placeholder = f"{{{key.upper()}}}"
            content = content.replace(placeholder, str(value))
        
        with open(filename, 'w') as f:
            f.write(content)
    
    def _equilibration_template(self):
        """Template for system equilibration."""
        return """# LAMMPS input file for membrane equilibration
# Generated on {timestamp}

units           real
dimension       3
boundary        p p f
atom_style      atomic

# Read data file
read_data       {DATA_FILE}

# Define potential parameters
pair_style      lj/cut 10.0
pair_coeff      1 1 0.070 3.550  # C-C (graphene)
pair_coeff      2 2 0.210 3.118  # O-O
pair_coeff      3 3 0.030 2.500  # H-H
pair_coeff      4 4 0.155 3.166  # OW-OW (water)
pair_coeff      5 5 0.030 2.500  # HW-HW (water)

# Mixed terms (Lorentz-Berthelot)
pair_modify     mix arithmetic

# Define groups
group           membrane type 1 2 3
group           water type 4 5

# Initial velocity
velocity        all create {TEMPERATURE} 12345 dist gaussian

# Fixes
fix             1 all nvt temp {TEMPERATURE} {TEMPERATURE} 100.0
fix             2 membrane setforce 0.0 0.0 0.0  # Fix membrane

# Output
thermo          {THERMO_FREQ}
thermo_style    custom step temp press pe ke etotal vol

dump            1 all atom {DUMP_FREQ} {OUTPUT_PREFIX}_equil.lammpstrj

# Timestep
timestep        {TIMESTEP}

# Run equilibration
run             {RUN_STEPS}

# Write restart
write_restart   {OUTPUT_PREFIX}_equil.restart
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def _pressure_ramp_template(self):
        """Template for pressure ramp simulation."""
        return """# LAMMPS input file for pressure ramp
# Generated on {timestamp}

units           real
dimension       3
boundary        p p f
atom_style      atomic

# Read restart file
read_restart    {OUTPUT_PREFIX}_equil.restart

# Define potential parameters
pair_style      lj/cut 10.0
pair_coeff      1 1 0.070 3.550  # C-C (graphene)
pair_coeff      2 2 0.210 3.118  # O-O
pair_coeff      3 3 0.030 2.500  # H-H
pair_coeff      4 4 0.155 3.166  # OW-OW (water)
pair_coeff      5 5 0.030 2.500  # HW-HW (water)

# Mixed terms
pair_modify     mix arithmetic

# Define groups
group           membrane type 1 2 3
group           water type 4 5

# Apply pressure
fix             1 water npt temp {TEMPERATURE} {TEMPERATURE} 100.0 z 0.0 {PRESSURE} 1000.0
fix             2 membrane setforce 0.0 0.0 0.0  # Fix membrane

# Output
thermo          {THERMO_FREQ}
thermo_style    custom step temp press pe ke etotal vol lz

dump            1 all atom {DUMP_FREQ} {OUTPUT_PREFIX}_ramp.lammpstrj

# Timestep
timestep        {TIMESTEP}

# Run pressure ramp
run             {RUN_STEPS}

# Write restart
write_restart   {OUTPUT_PREFIX}_ramp.restart
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def _production_template(self):
        """Template for production simulation."""
        return """# LAMMPS input file for production run
# Generated on {timestamp}

units           real
dimension       3
boundary        p p f
atom_style      atomic

# Read restart file
read_restart    {OUTPUT_PREFIX}_ramp.restart

# Define potential parameters
pair_style      lj/cut 10.0
pair_coeff      1 1 0.070 3.550  # C-C (graphene)
pair_coeff      2 2 0.210 3.118  # O-O
pair_coeff      3 3 0.030 2.500  # H-H
pair_coeff      4 4 0.155 3.166  # OW-OW (water)
pair_coeff      5 5 0.030 2.500  # HW-HW (water)

# Mixed terms
pair_modify     mix arithmetic

# Define groups
group           membrane type 1 2 3
group           water type 4 5

# Maintain pressure
fix             1 water npt temp {TEMPERATURE} {TEMPERATURE} 100.0 z 0.0 {PRESSURE} 1000.0
fix             2 membrane setforce 0.0 0.0 0.0  # Fix membrane

# Compute water flux
compute         water_pos water property/atom z
fix             flux_calc water ave/time 100 10 {THERMO_FREQ} c_water_pos[*] file {OUTPUT_PREFIX}_flux.dat

# Output
thermo          {THERMO_FREQ}
thermo_style    custom step temp press pe ke etotal vol lz

dump            1 all atom {DUMP_FREQ} {OUTPUT_PREFIX}_prod.lammpstrj

# Timestep
timestep        {TIMESTEP}

# Production run
run             {RUN_STEPS}

# Final output
write_restart   {OUTPUT_PREFIX}_final.restart
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def create_simulation_sequence(self, output_dir, base_params=None):
        """
        Create a complete simulation sequence (equilibration -> ramp -> production).
        
        Args:
            output_dir (str): Directory for output files
            base_params (dict): Base simulation parameters
        
        Returns:
            list: List of created input files
        """
        if base_params is None:
            base_params = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        input_files = []
        
        # Equilibration
        equil_file = os.path.join(output_dir, "equilibration.in")
        equil_params = {**base_params, 'run_steps': 50000, 'output_prefix': 'equil'}
        self.write_input_file(equil_file, 'equilibration', **equil_params)
        input_files.append(equil_file)
        
        # Pressure ramp
        ramp_file = os.path.join(output_dir, "pressure_ramp.in")
        ramp_params = {**base_params, 'run_steps': 100000, 'output_prefix': 'ramp'}
        self.write_input_file(ramp_file, 'pressure_ramp', **ramp_params)
        input_files.append(ramp_file)
        
        # Production
        prod_file = os.path.join(output_dir, "production.in")
        prod_params = {**base_params, 'run_steps': 500000, 'output_prefix': 'prod'}
        self.write_input_file(prod_file, 'production', **prod_params)
        input_files.append(prod_file)
        
        return input_files

    def write_membrane_input(self, filename, data_file, membrane):
        """
        Write LAMMPS input file specifically for membrane simulation.
        
        Args:
            filename (str): Output input filename
            data_file (str): Path to data file
            membrane: Membrane object from Phase 1
        """
        params = {
            'data_file': os.path.basename(data_file),
            'timestep': 1.0,
            'temperature': 300.0,
            'pressure': membrane.pore_size_nm,  # Use pore size as pressure proxy
            'run_steps': 50000,  # Shorter run for demo
            'dump_freq': 1000,
            'thermo_freq': 1000,
            'output_prefix': f"membrane_{membrane.name.replace(' ', '_')}"
        }
        
        # Use production template for membrane simulation
        self.write_input_file(filename, 'production', **params)

    def write_realistic_membrane_input(self, filename, data_file, membrane):
        """
        Generate realistic LAMMPS input script based on literature (Schmidt et al. [17]).
        
        Args:
            filename (str): Output LAMMPS input file path
            data_file (str): Path to LAMMPS data file
            membrane: Membrane object with properties
        """
        # Generate realistic LAMMPS script based on Schmidt et al. [17]
        realistic_input = f"""# LAMMPS input file for {membrane.name} membrane simulation
# Based on Schmidt et al. [17] for GO-water interactions
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

units real
atom_style full
boundary p p f

# Read data file
read_data {os.path.basename(data_file)}

# Pair style and coefficients based on literature
pair_style lj/cut 10.0
pair_coeff * * 0.1553 3.166  # GO-water interactions from Schmidt et al.

# Bond styles for molecular water
bond_style harmonic
angle_style harmonic

# Molecular constraints
fix rigid_water water shake 1e-4 20 0 b 1 a 1

# Temperature and pressure control
fix 1 all nvt temp 300.0 300.0 100.0
fix 2 all aveforce NULL NULL -1.0  # Applied pressure gradient

# Output settings
compute temp_water water temp
compute pressure_system all pressure temp_water

thermo_style custom step temp c_temp_water press c_pressure_system pe ke etotal vol
thermo 1000

# Trajectory dump for flux analysis
dump 1 all atom 100 dump.xyz
dump_modify 1 scale no

# Production run for water flux measurement
timestep 1.0
run 50000

# Calculate and output final properties
variable final_temp equal c_temp_water
variable final_press equal c_pressure_system
variable final_pe equal pe
variable final_ke equal ke

print "Final temperature: ${{final_temp}} K"
print "Final pressure: ${{final_press}} atm" 
print "Final potential energy: ${{final_pe}} kcal/mol"
print "Final kinetic energy: ${{final_ke}} kcal/mol"
"""
        
        with open(filename, 'w') as f:
            f.write(realistic_input)
        
        print(f"  Realistic LAMMPS input written to {filename}")

def create_batch_inputs(membrane_list, output_base_dir):
    """
    Create LAMMPS input files for a batch of membranes.
    
    Args:
        membrane_list (list): List of membrane configurations
        output_base_dir (str): Base directory for outputs
    
    Returns:
        dict: Dictionary mapping membrane names to input file lists
    """
    writer = LAMMPSInputWriter()
    batch_files = {}
    
    for membrane in membrane_list:
        mem_dir = os.path.join(output_base_dir, membrane.name.replace(' ', '_'))
        params = {
            'temperature': 300.0,
            'pressure': 1.0,  # Will be modified for pressure sweeps
            'data_file': f"{membrane.name.replace(' ', '_')}.lammps"
        }
        
        input_files = writer.create_simulation_sequence(mem_dir, params)
        batch_files[membrane.name] = input_files
    
    return batch_files
