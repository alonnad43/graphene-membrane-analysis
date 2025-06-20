# Technical Code Structure and Developer Reference

This document provides a comprehensive technical overview of the graphene membrane simulation codebase, including all major scripts, classes, functions, and data/JSON files. Use this as a reference for development, extension, or integration.

---

## 1. Main Simulation Workflow

### `src/main.py`
- **Role:** Orchestrates the full four-phase simulation workflow:
  1. Macroscale simulation (flux, rejection, mechanical properties)
  2. Hybrid structure modeling and optimization
  3. Atomistic simulation (LAMMPS)
  4. Chemical/biological simulation (contaminant removal, regeneration)
- **Key Functions:**
  - `validate_forcefield_params()` – Checks completeness of forcefield JSON
  - `generate_membrane_variants()` – Generates all membrane variants with variability
  - `main()` – Entry point, runs all phases in order
- **Imports:** Membrane models, simulators, plotting, properties, structure builders, and utilities

---

## 2. Core Simulation Modules

### `src/membrane_model.py`
- **Classes:**
  - `MembraneVariant` – Represents a single membrane variant (properties, methods for property inference)
  - `Membrane` – Main membrane object (pore size, thickness, flux, modulus, etc.)
- **Key Functions:**
  - `get_variant_properties`, `infer_variant_from_spacing`, `to_dict`, `from_dict`, `to_lammps_data`, `get_interlayer_spacing`, `get_variant_info`, etc.
  - `compute_interface_penalty(layers)` – Computes interface penalty for hybrid structures
  - `generate_membrane_variants_with_variability(...)` – Generates variants with statistical variability

### `src/data_builder.py`
- **Class:** `LAMMPSDataBuilder`
  - `__init__`, `create_membrane_structure`, `create_graphene_sheet`, `add_go_functional_groups`, `add_pei_branches`, `add_ions`, `add_contaminants`, `add_water_layer`, `get_atom_charge`, `write_lammps_data`
- **Role:** Builds atomic structures for LAMMPS, adds functional groups, ions, contaminants, water

### `src/input_writer.py`
- **Class:** `LAMMPSInputWriter`
  - `__init__`, `_equilibration_template`, `_pressure_ramp_template`, `_production_template`, `write_input_file`, `create_simulation_sequence`, `write_membrane_input`, `write_realistic_membrane_input`
  - `create_batch_inputs(membrane_list, output_base_dir)`
- **Role:** Generates LAMMPS input scripts for all simulation types

### `src/lammps_runner.py`
- **Class:** `LAMMPSRunner`
  - `__init__`, `run_simulation`, `run_pressure_sweep`, `run_membrane_simulation`, `_execute_lammps`, `_log_simulation`
  - `check_lammps_installation()`
- **Role:** Runs LAMMPS simulations, manages input/output, error handling

### `src/simulate_chemistry.py`
- **Class:** `ChemicalSimulationEngine`
  - `validate_forcefield_params`, `__init__`, `load_contaminant_data`, `_create_default_contaminant_data`, `calculate_thermodynamic_favorability`, `apply_diffusion_effects`, `simulate_contaminant_removal`, `load_lab_characterization_data`, `apply_lab_validated_membrane_properties`, `get_summary_statistics`, `apply_regeneration`, `export_results`
  - `run_phase4_simulation(...)` (standalone)
- **Role:** Simulates chemical/biological removal, regeneration, and exports results

---

## 3. Analysis, Plotting, and Utilities

- `src/plot_chemistry.py` – `ChemistryPlotter` (plots for phase 4, time series, saturation, comparative performance, regeneration, summary)
- `src/plot_utils.py` – Utility plotting functions for flux, rejection, etc.
- `src/plot_all_results.py` – `ComprehensivePlotter` (aggregates and visualizes all-phase results)
- `src/unify_results.py` – `ResultsUnifier` (merges results from all phases, exports unified datasets)
- `src/output_parser.py` – `LAMMPSOutputParser` (parses LAMMPS output files)
- `src/io_utils.py` – File and data utilities

---

## 4. Advanced/Ultra-Efficient Modules

- `src/ultra_efficient_chemistry.py` – `UltraEfficientChemicalEngine` (fast chemistry simulation)
- `src/ultra_efficient_flux.py` – `UltraEfficientFluxSimulator`
- `src/ultra_efficient_membrane_generation.py` – `UltraEfficientMembraneGenerator`
- `src/ultra_efficient_oil_rejection.py` – `UltraEfficientOilRejectionSimulator`
- `src/ultra_efficient_plotting.py` – `UltraEfficientPlotter` (batch and high-speed plotting)

---

## 5. Data and JSON Files

### Main Data Files
- `forcefield_and_simulation_data.json` (root):
  - **Sections:** atom_types, bond_types, angle_types, dihedral_types, cross_terms, microplastic_hybrid, coarse_grained_beads, regeneration_chemistry, chemical_config, contaminant_data, synthesis_recommender, experimental_conditions, validation_and_field_data
  - **Purpose:** Single source for all simulation, force-field, contaminant, synthesis, field, and validation parameters
- `data/contaminant_data.json`, `data/chemical_config.json`, `data/forcefield_params.json` (legacy/compatibility)
- `data/lab_characterization_example.json` – Template for lab/field data
- `output/phase4_chemistry_*.json` – Simulation results (per run)
- `output/unified_results_*.csv/.xlsx` – Unified results for all phases
- `output/plots/` – All generated plots, organized by analysis type (e.g. `output/plots/oil_rejection_summary/`, `output/plots/flux_vs_thickness_per_pressure/`, etc.)

### Monitoring and Parameter Files
- `monitoring/chemistry_simulation_params.json`, `monitoring/contaminant_membrane_properties.json`, `monitoring/references.py`, etc.
- `monitoring/models.py` – Data models for simulation/monitoring

---

## 6. Notebooks and Tools

- `notebooks/implement_all_suggestions.ipynb` – Demonstration and extension notebook
- `notebooks/custom_scenarios_and_validation.ipynb` – Custom scenario and validation notebook
- `notebooks/workflow_validation_and_extension.ipynb` – Workflow validation/extension
- `tools/lab_export.py` – Exports simulation specs/results to lab-ready CSV

---

## 7. How to Extend and Integrate
- Add new membrane types, contaminants, or field data by updating `forcefield_and_simulation_data.json` (placeholders provided)
- Add new simulation/analysis modules in `src/` and reference them in `main.py` or notebooks
- Use `monitoring/` for advanced options, result templates, and parameter management
- For new lab/field data, use the template in `data/lab_characterization_example.json` and log results in the JSON
- For batch or high-throughput runs, use the ultra-efficient modules and batch input/output utilities

---

## 8. Summary Table: Key Classes and Functions

| File                        | Class/Function                        | Purpose/Role                                    |
|-----------------------------|---------------------------------------|-------------------------------------------------|
| main.py                     | main(), validate_forcefield_params()   | Orchestrates workflow, validates data           |
| membrane_model.py           | Membrane, MembraneVariant              | Membrane objects, property inference            |
| data_builder.py             | LAMMPSDataBuilder                      | Builds atomic structures for LAMMPS             |
| input_writer.py             | LAMMPSInputWriter                      | Writes LAMMPS input scripts                     |
| lammps_runner.py            | LAMMPSRunner, check_lammps_installation| Runs LAMMPS, manages I/O                        |
| simulate_chemistry.py       | ChemicalSimulationEngine, run_phase4   | Phase 4 simulation, export, validation          |
| plot_chemistry.py           | ChemistryPlotter                       | Phase 4 plotting                                |
| unify_results.py            | ResultsUnifier, unify_all_results      | Merges and exports unified results              |
| output_parser.py            | LAMMPSOutputParser                     | Parses LAMMPS output                            |
| ultra_efficient_*.py        | UltraEfficient* classes                | Fast/batch simulation and plotting              |

---

## 9. Data Flow and Extensibility
- All simulation phases and modules read/write to the consolidated JSON and output folders
- Results are unified and validated against lab/field data
- All code is modular and ready for extension (add new classes, functions, or data as needed)

---

## Ultra-Optimized Orchestrator & Output (2025)
- The codebase includes a final ultra-optimized orchestrator supporting batch simulation, vectorized flux/rejection, and advanced plotting.
- All results and plots are saved under `output/` with subfolders for each phase and plot type (e.g., `output/phase1/`, `output/plots/oil_rejection_summary/`).
- Robust error handling and serialization are built in, with performance summaries for each run.
- All modules use numpy, pandas, matplotlib, and scipy, and support CSV/JSON export.
- The workflow is validated end-to-end and is ready for extension and integration.

---

For further details, see code comments in each file, and refer to the demonstration notebooks for usage examples and extension patterns.
