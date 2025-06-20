# input_writer.py

"""
Creates LAMMPS input (.in) files for MD simulations.

# TODO: Support multiple material models (GO, rGO)
# TODO: Add simulation control presets (low-pressure, fast-test, long-run)
"""

import os
from datetime import datetime
import json

class LAMMPSInputWriter:
    """
    Generates LAMMPS input files for membrane simulations.
    """
    
    def __init__(self, forcefield_path="data/forcefield_params.json"):
        self.forcefield = None
        try:
            with open(forcefield_path, "r") as f:
                self.forcefield = json.load(f)
        except Exception:
            pass
        
        self.templates = {
            'equilibration': self._equilibration_template(),
            'pressure_ramp': self._pressure_ramp_template(),
            'production': self._production_template()
        }
    
    def _equilibration_template(self):
        """Template for system equilibration."""
        return """# LAMMPS input file for membrane equilibration
# Generated on {TIMESTAMP}

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
"""

    def _pressure_ramp_template(self):
        """Template for pressure ramp simulation."""
        return """# LAMMPS input file for pressure ramp
# Generated on {TIMESTAMP}

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
"""

    def _production_template(self):
        """Template for production simulation."""
        return """# LAMMPS input file for production run
# Generated on {TIMESTAMP}

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
"""

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
            'output_prefix': 'sim',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Merge defaults with provided parameters
        params = {**defaults, **kwargs}
        
        # Replace all placeholders
        content = template
        for key, value in params.items():
            placeholder = f"{{{key.upper()}}}"
            content = content.replace(placeholder, str(value))
        
        with open(filename, 'w') as f:
            f.write(content)
    
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
        
        # Always use 'data.lammps' as the data file name
        data_file = base_params.get('data_file', 'data.lammps')
        
        # Equilibration
        equil_file = os.path.join(output_dir, "equilibration.in")
        equil_params = {**base_params, 'run_steps': 50000, 'output_prefix': 'equil', 'data_file': data_file}
        self.write_input_file(equil_file, 'equilibration', **equil_params)
        input_files.append(equil_file)
        
        # Pressure ramp
        ramp_file = os.path.join(output_dir, "pressure_ramp.in")
        ramp_params = {**base_params, 'run_steps': 100000, 'output_prefix': 'ramp', 'data_file': data_file}
        self.write_input_file(ramp_file, 'pressure_ramp', **ramp_params)
        input_files.append(ramp_file)
        
        # Production
        prod_file = os.path.join(output_dir, "production.in")
        prod_params = {**base_params, 'run_steps': 500000, 'output_prefix': 'prod', 'data_file': data_file}
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
            'data_file': os.path.basename(data_file) if data_file else 'data.lammps',
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
        Generate realistic LAMMPS input script with carboxyl, edge, N, S, and all new types/parameters.
        """
        data_file_basename = os.path.basename(data_file)
        lines = [
            f"# LAMMPS input file for {membrane.name} membrane simulation",
            f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "units real",
            "atom_style full",
            "boundary p p f",
            f"read_data {data_file_basename}",
            "group water type 4 5",
            "# group membrane type ... (add as needed)",
        ]
        # Determine pair style and kspace style based on boundary
        boundary_line = next((l for l in lines if l.strip().startswith('boundary')), 'boundary p p f')
        b = boundary_line.strip().split()
        if b[-3:] == ['p', 'p', 'p']:
            pair_style_line = 'pair_style lj/cut/coul/long 10.0'
            kspace_line = 'kspace_style pppm 1.0e-4'
        else:
            pair_style_line = 'pair_style lj/cut/coul/cut 10.0'
            kspace_line = 'kspace_style none'
        # Replace or insert pair_style line
        lines = [l for l in lines if not l.strip().startswith('pair_style')]
        # Insert pair_style after boundary
        boundary_idx = next((i for i, l in enumerate(lines) if l.strip().startswith('boundary')), None)
        if boundary_idx is not None:
            lines.insert(boundary_idx + 1, pair_style_line)
            lines.insert(boundary_idx + 2, kspace_line)
        else:
            lines.insert(0, pair_style_line)
            lines.insert(1, kspace_line)
        # Remove any previous kspace_style lines elsewhere
        lines = [l for l in lines if not (l.strip().startswith('kspace_style') and l not in [kspace_line])]
        # Add pair_coeffs from forcefield
        if self.forcefield:
            atom_types = self.forcefield["atom_types"]
            for k1, v1 in atom_types.items():
                for k2, v2 in atom_types.items():
                    if v1["id"] <= v2["id"]:
                        lines.append(f"pair_coeff {v1['id']} {v2['id']} {v1['epsilon']} {v1['sigma']}  # {k1}-{k2}")
        lines.append("kspace_style pppm 1.0e-4")
        lines.append("bond_style harmonic")
        lines.append("angle_style harmonic")
        # Only write dihedral section if present in data file
        write_dihedrals = self._should_write_dihedrals(data_file)
        if write_dihedrals:
            lines.append("dihedral_style opls")
            if self.forcefield and "dihedral_types" in self.forcefield:
                for k, v in self.forcefield["dihedral_types"].items():
                    # OPLS expects K1 K2 K3 K4 (all floats, default 0.0 if missing)
                    K1 = v.get("K1", 0.0)
                    K2 = v.get("K2", 0.0)
                    K3 = v.get("K3", 0.0)
                    K4 = v.get("K4", 0.0)
                    lines.append(f"dihedral_coeff {v['id']} {K1} {K2} {K3} {K4}  # {k}")
        # Add custom cross-term pair_coeffs
        if self.forcefield and "cross_terms" in self.forcefield:
            for k, v in self.forcefield["cross_terms"].items():
                # Parse atom type names from cross-term key
                a1, a2 = k.split('-')
                id1 = self.forcefield["atom_types"][a1]["id"] if a1 in self.forcefield["atom_types"] else None
                id2 = self.forcefield["atom_types"][a2]["id"] if a2 in self.forcefield["atom_types"] else None
                if id1 and id2:
                    lines.append(f"pair_coeff {id1} {id2} {v['epsilon']} {v['sigma']}  # {k}")
        # --- Robust section parser for LAMMPS data file ---
        def parse_types_from_data(data_file, section_name):
            types = set()
            try:
                with open(data_file, 'r') as f:
                    lines = f.readlines()
                in_section = False
                for idx, line in enumerate(lines):
                    if line.strip() == section_name:
                        in_section = True
                        header_skipped = False
                        continue
                    if in_section:
                        if not header_skipped:
                            header_skipped = True  # skip the header line (e.g., '1 2 33 51')
                            continue
                        if not line.strip() or line[0].isalpha():
                            break
                        parts = line.strip().split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            types.add(int(parts[1]))
            except Exception as e:
                print(f"[DEBUG] Error parsing {section_name}: {e}")
            return types
        bond_types_in_data = parse_types_from_data(data_file, 'Bonds')
        angle_types_in_data = parse_types_from_data(data_file, 'Angles')
        dihedral_types_in_data = parse_types_from_data(data_file, 'Dihedrals')
        # Add bond_coeffs for all types in data file
        if self.forcefield:
            ff_bond_types = {v['id']: (k, v) for k, v in self.forcefield['bond_types'].items()}
            for bond_type in sorted(bond_types_in_data):
                if bond_type in ff_bond_types:
                    k, v = ff_bond_types[bond_type]
                    lines.append(f"bond_coeff {v['id']} {v['k']} {v['r0']}  # {k}")
                else:
                    lines.append(f"bond_coeff {bond_type} 100.0 1.0  # MISSING TYPE")
        # --- PATCH: Guarantee all bond_coeffs for all defined types ---
        def parse_num_types_from_data(data_file, type_name):
            # e.g., type_name='bond types' -> line: '18 bond types'
            try:
                with open(data_file, 'r') as f:
                    for line in f:
                        if line.strip().endswith(type_name):
                            return int(line.strip().split()[0])
            except Exception as e:
                print(f"[DEBUG] Error parsing num {type_name}: {e}")
            return 0
        num_bond_types = parse_num_types_from_data(data_file, 'bond types')
        # Build a dict of bond_coeff lines for all types 1..N
        bond_coeff_dict = {}
        if self.forcefield:
            ff_bond_types = {v['id']: (k, v) for k, v in self.forcefield['bond_types'].items()}
            for bond_type in range(1, num_bond_types + 1):
                if bond_type in ff_bond_types:
                    k, v = ff_bond_types[bond_type]
                    bond_coeff_dict[bond_type] = f"bond_coeff {v['id']} {v['k']} {v['r0']}  # {k}"
                else:
                    bond_coeff_dict[bond_type] = f"bond_coeff {bond_type} 100.0 1.0  # MISSING TYPE"
        else:
            for bond_type in range(1, num_bond_types + 1):
                bond_coeff_dict[bond_type] = f"bond_coeff {bond_type} 100.0 1.0  # MISSING TYPE"
        # Remove any previously added bond_coeff lines
        lines = [l for l in lines if not l.strip().startswith('bond_coeff')]
        # Insert all bond_coeffs in order after bond_style
        bond_style_idx = next((i for i, l in enumerate(lines) if l.strip().startswith('bond_style')), None)
        bond_coeff_lines = [bond_coeff_dict[i] for i in range(1, num_bond_types + 1)]
        if bond_style_idx is not None:
            for i, coeff_line in enumerate(bond_coeff_lines):
                lines.insert(bond_style_idx + 1 + i, coeff_line)
        else:
            lines += bond_coeff_lines
        # --- END PATCH ---
        # --- DEBUG/VALIDATION BLOCK ---
        # --- Ensure all debug variables are always defined at the top of the block ---
        ff_bond_types = {v['id']: (k, v) for k, v in self.forcefield['bond_types'].items()} if self.forcefield and 'bond_types' in self.forcefield else {}
        ff_angle_types = {v['id']: (k, v) for k, v in self.forcefield['angle_types'].items()} if self.forcefield and 'angle_types' in self.forcefield else {}
        ff_dihedral_types = {v['id']: (k, v) for k, v in self.forcefield['dihedral_types'].items()} if self.forcefield and 'dihedral_types' in self.forcefield else {}
        bond_coeff_lines = [l for l in lines if l.strip().startswith('bond_coeff')]
        angle_coeff_lines = [l for l in lines if l.strip().startswith('angle_coeff')]
        dihedral_coeff_lines = [l for l in lines if l.strip().startswith('dihedral_coeff')]
        # Print all type IDs found and in forcefield
        print(f"[DEBUG] Bond type IDs in data file: {sorted(bond_types_in_data)}")
        print(f"[DEBUG] Bond type IDs in forcefield: {sorted(ff_bond_types.keys())}")
        print(f"[DEBUG] Angle type IDs in data file: {sorted(angle_types_in_data)}")
        print(f"[DEBUG] Angle type IDs in forcefield: {sorted(ff_angle_types.keys())}")
        if write_dihedrals:
            print(f"[DEBUG] Dihedral type IDs in data file: {sorted(dihedral_types_in_data)}")
            print(f"[DEBUG] Dihedral type IDs in forcefield: {sorted(ff_dihedral_types.keys())}")
        # Only print angle/dihedral debug output if those coeff lines are non-empty
        if angle_coeff_lines:
            print(f"[DEBUG] Angle_coeff lines written:")
            for l in angle_coeff_lines:
                print(l)
            unused_angles = [aid for aid in ff_angle_types if aid not in angle_types_in_data]
            if unused_angles:
                print(f"[WARNING] Unused angle types in forcefield: {unused_angles}")
            print(f"[SUMMARY] Angles: {len(angle_types_in_data)} in data, {len(ff_angle_types)} in forcefield.")
        if dihedral_coeff_lines:
            print(f"[DEBUG] Dihedral_coeff lines written:")
            for l in dihedral_coeff_lines:
                print(l)
            unused_dihedrals = [did for did in ff_dihedral_types if did not in dihedral_types_in_data]
            if unused_dihedrals:
                print(f"[WARNING] Unused dihedral types in forcefield: {unused_dihedrals}")
            print(f"[SUMMARY] Dihedrals: {len(dihedral_types_in_data)} in data, {len(ff_dihedral_types)} in forcefield.")
        # Print summary
        print(f"[SUMMARY] Bonds: {len(bond_types_in_data)} in data, {len(ff_bond_types)} in forcefield.")
        print(f"[SUMMARY] Angles: {len(angle_types_in_data)} in data, {len(ff_angle_types)} in forcefield.")
        if write_dihedrals:
            print(f"[SUMMARY] Dihedrals: {len(dihedral_types_in_data)} in data, {len(ff_dihedral_types)} in forcefield.")
        # Print first 20 lines of generated input
        print(f"[DEBUG] First 20 lines of generated LAMMPS input:")
        for l in lines[:20]:
            print(l)
        # --- END DEBUG/VALIDATION BLOCK ---
        # Add angle_coeffs for all types in data file
        if self.forcefield:
            ff_angle_types = {v['id']: (k, v) for k, v in self.forcefield['angle_types'].items()}
            for angle_type in sorted(angle_types_in_data):
                if angle_type in ff_angle_types:
                    k, v = ff_angle_types[angle_type]
                    lines.append(f"angle_coeff {v['id']} {v['k']} {v['theta0']}  # {k}")
                else:
                    lines.append(f"angle_coeff {angle_type} 50.0 120.0  # MISSING TYPE")
        # Add dihedral_coeffs for all types in data file (if present)
        if write_dihedrals and self.forcefield and 'dihedral_types' in self.forcefield:
            ff_dihedral_types = {v['id']: (k, v) for k, v in self.forcefield['dihedral_types'].items()}
            for dihedral_type in sorted(dihedral_types_in_data):
                if dihedral_type in ff_dihedral_types:
                    k, v = ff_dihedral_types[dihedral_type]
                    K1 = v.get('K1', 0.0)
                    K2 = v.get('K2', 0.0)
                    K3 = v.get('K3', 0.0)
                    K4 = v.get('K4', 0.0)
                    lines.append(f"dihedral_coeff {v['id']} {K1} {K2} {K3} {K4}  # {k}")
                else:
                    lines.append(f"dihedral_coeff {dihedral_type} 0.0 0.0 0.0 0.0  # MISSING TYPE")
        # Remove any previous kspace_style lines
        lines = [l for l in lines if not l.strip().startswith('kspace_style')]
        # Find the boundary line from the template
        boundary_line = next((l for l in lines if l.strip().startswith('boundary')), 'boundary p p f')
        # Insert kspace_style immediately after boundary
        kspace_line = self._get_kspace_style(boundary_line)
        # Find index of boundary line
        boundary_idx = next((i for i, l in enumerate(lines) if l.strip().startswith('boundary')), None)
        if boundary_idx is not None:
            lines.insert(boundary_idx + 1, kspace_line)
        else:
            lines.append(kspace_line)
        # Insert bond_style and bond_coeffs in correct order
        # Find index of bond_style
        bond_style_idx = next((i for i, l in enumerate(lines) if l.strip().startswith('bond_style')), None)
        bond_coeff_lines = [l for l in lines if l.strip().startswith('bond_coeff')]
        # Remove all bond_coeff lines from lines
        lines = [l for l in lines if not l.strip().startswith('bond_coeff')]
        # Insert bond_coeffs immediately after bond_style
        if bond_style_idx is not None:
            for i, coeff_line in enumerate(bond_coeff_lines):
                lines.insert(bond_style_idx + 1 + i, coeff_line)
        else:
            # If bond_style not found, append at end
            lines += bond_coeff_lines
        # Repeat for angle_style/angle_coeff and dihedral_style/dihedral_coeff
        angle_style_idx = next((i for i, l in enumerate(lines) if l.strip().startswith('angle_style')), None)
        angle_coeff_lines = [l for l in lines if l.strip().startswith('angle_coeff')]
        lines = [l for l in lines if not l.strip().startswith('angle_coeff')]
        if angle_style_idx is not None:
            for i, coeff_line in enumerate(angle_coeff_lines):
                lines.insert(angle_style_idx + 1 + i, coeff_line)
        else:
            lines += angle_coeff_lines
        dihedral_style_idx = next((i for i, l in enumerate(lines) if l.strip().startswith('dihedral_style')), None)
        dihedral_coeff_lines = [l for l in lines if l.strip().startswith('dihedral_coeff')]
        lines = [l for l in lines if not l.strip().startswith('dihedral_coeff')]
        if dihedral_style_idx is not None:
            for i, coeff_line in enumerate(dihedral_coeff_lines):
                lines.insert(dihedral_style_idx + 1 + i, coeff_line)
        else:
            lines += dihedral_coeff_lines
        # Add remaining fix, output, and run commands
        lines += [
            "fix rigid_water water shake 1e-4 20 0 b 4 a 4",
            "fix 1 all nvt temp 300.0 300.0 100.0",
            "fix 2 all aveforce NULL NULL -1.0",
            "compute temp_water water temp",
            "compute pressure_system all pressure temp_water",
            "thermo_style custom step temp c_temp_water press c_pressure_system pe ke etotal vol",
            "thermo 1000",
            "dump 1 all atom 100 dump.xyz",
            "dump_modify 1 scale no",
            "timestep 1.0",
            "run 50000",
            "variable final_temp equal c_temp_water",
            "variable final_press equal c_pressure_system",
            "variable final_pe equal pe",
            "variable final_ke equal ke",
            'print "Final temperature: ${final_temp} K"',
            'print "Final pressure: ${final_press} atm"',
            'print "Final potential energy: ${final_pe} kcal/mol"',
            'print "Final kinetic energy: ${final_ke} kcal/mol"',
        ]
        # --- FINAL PATCH: Guarantee all angle_coeffs for all defined types (overwrite any previous logic) ---
        num_angle_types = parse_num_types_from_data(data_file, 'angle types')
        angle_coeff_dict = {}
        if self.forcefield and 'angle_types' in self.forcefield:
            ff_angle_types = {v['id']: (k, v) for k, v in self.forcefield['angle_types'].items()}
            for angle_type in range(1, num_angle_types + 1):
                if angle_type in ff_angle_types:
                    k, v = ff_angle_types[angle_type]
                    angle_coeff_dict[angle_type] = f"angle_coeff {v['id']} {v['k']} {v['theta0']}  # {k}"
                else:
                    angle_coeff_dict[angle_type] = f"angle_coeff {angle_type} 50.0 120.0  # MISSING TYPE"
        else:
            for angle_type in range(1, num_angle_types + 1):
                angle_coeff_dict[angle_type] = f"angle_coeff {angle_type} 50.0 120.0  # MISSING TYPE"
        # Remove any previously added angle_coeff lines
        lines = [l for l in lines if not l.strip().startswith('angle_coeff')]
        # Insert all angle_coeffs in order after angle_style
        angle_style_idx = next((i for i, l in enumerate(lines) if l.strip().startswith('angle_style')), None)
        angle_coeff_lines = [angle_coeff_dict[i] for i in range(1, num_angle_types + 1)]
        if angle_style_idx is not None:
            for i, coeff_line in enumerate(angle_coeff_lines):
                lines.insert(angle_style_idx + 1 + i, coeff_line)
        else:
            lines += angle_coeff_lines
        # --- END FINAL PATCH ---
        # Add minimization step before dynamics in LAMMPS input
        # Find the index of the line with 'run' and insert minimization before it
        run_idx = next((i for i, l in enumerate(lines) if l.strip().startswith('run ')), None)
        if run_idx is not None:
            lines.insert(run_idx, 'minimize 1.0e-4 1.0e-6 1000 10000')
        else:
            lines.append('minimize 1.0e-4 1.0e-6 1000 10000')
        # Reduce timestep for more stable dynamics
        for i, line in enumerate(lines):
            if line.strip().startswith('timestep '):
                lines[i] = 'timestep 0.5'  # Reduce from 1.0 to 0.5
                break
        # Further reduce timestep for maximum stability
        for i, line in enumerate(lines):
            if line.strip().startswith('timestep '):
                lines[i] = 'timestep 0.25'  # Reduce from 0.5 to 0.25
                break
        # Add further minimization and short NVT equilibration before production MD in LAMMPS input
        # Insert after minimization, before main run
        min_idx = next((i for i, l in enumerate(lines) if l.strip().startswith('minimize ')), None)
        if min_idx is not None:
            # Insert NVT equilibration after minimization
            lines.insert(min_idx + 1, 'fix nvt_eq all nvt temp 300.0 300.0 100.0')
            lines.insert(min_idx + 2, 'run 2000')
            lines.insert(min_idx + 3, 'unfix nvt_eq')
        with open(filename, 'w') as f:
            f.write("\n".join(lines))
        print(f"  Realistic LAMMPS input written to {filename}")

    def _should_write_dihedrals(self, data_file_path):
        """
        Check the LAMMPS data file for nonzero dihedrals and dihedral types.
        Returns True if both are >0, else False.
        """
        try:
            with open(data_file_path, 'r') as f:
                lines = f.readlines()
            dihedrals = 0
            dihedral_types = 0
            for line in lines:
                if 'dihedrals' in line and 'types' not in line:
                    dihedrals = int(line.strip().split()[0])
                if 'dihedral types' in line:
                    dihedral_types = int(line.strip().split()[0])
            return dihedrals > 0 and dihedral_types > 0
        except Exception:
            return False

    def _get_kspace_style(self, boundary_line):
        """
        Returns the appropriate kspace_style for the given boundary.
        For non-periodic boundaries, we avoid kspace entirely by using cutoff-based pair styles.
        """
        b = boundary_line.strip().split()
        if b[-3:] == ['p', 'p', 'p']:
            return 'kspace_style pppm 1.0e-4'
        else:
            # For any non-periodic boundary, use no kspace (will use lj/cut/coul/cut)
            return 'kspace_style none'

    def get_run_steps(self, default_steps):
        """
        Return a short number of steps if LAMMPS_TEST_RUN is set, else default.
        """
        test_env = os.environ.get('LAMMPS_TEST_RUN', '').lower()
        if test_env in ['1', 'true', 'yes']:
            return 1000  # Short test run
        return default_steps

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
