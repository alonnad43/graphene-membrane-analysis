# GO/rGO/Hybrid Membrane Simulation Framework

**Executive Summary**
This simulation framework enables the engineering and optimization of GO/rGO-based membranes for water purification. It integrates physical modeling, atomistic validation, and chemical filtration kinetics, supporting both research and practical deployment.

---

## 🦦 Simulation Workflow Overview

The system runs a four-phase workflow:

### 1. **Macroscale Simulation (Phase 1)**
- Physics-based flux and oil rejection modeling using empirical and literature-derived parameters.
- All membrane constants are loaded from `properties.py` and the consolidated JSON data file.

### 2. **Hybrid Structure Modeling (Phase 2)**
- Designs and optimizes hybrid GO/rGO membrane structures.
- Blends mechanical and transport properties, models interface effects, and selects top-performing candidates for atomistic simulation.

### 3. **Atomistic Simulation (Phase 3)**
- Builds atomic structures and generates LAMMPS input files using `data_builder.py` and `input_writer.py`.
- Runs LAMMPS MD simulations for the top hybrid structure.
- **If any phase fails, the workflow halts and prints an error explanation.**

### 4. **Chemical/Biological Simulation (Phase 4)**
- Simulates contaminant removal and biological interactions using empirical/physical models.
- Integrates results and generates summary plots.

---

## 🚀 Ultra-Optimized Pipeline & Output Structure (2025)
- The simulation pipeline is now fully ultra-optimized, supporting batch simulation, vectorized flux/rejection, and advanced plotting.
- All results and plots are saved under `output/` with subfolders for each phase and plot type (e.g., `output/phase1/`, `output/plots/oil_rejection_summary/`).
- The orchestrator and all modules use robust error handling and serialization, with performance summaries saved for each run.
- All code is modular, uses numpy, pandas, matplotlib, and scipy, and supports CSV/JSON export for all data.
- The workflow is validated end-to-end and produces publication-quality results and figures.

## 🗂️ Code and Data Structure

- **`main.py`**: Orchestrates the full workflow, error handling, and result integration.
- **`data_builder.py`**: Builds atomic structures for LAMMPS, adds functional groups, ions, contaminants, and water.
- **`input_writer.py`**: Generates LAMMPS input scripts, handles OPLS dihedral coefficients, and writes all force field terms.
- **`properties.py`**: Stores all physical constants, empirical values, and simulation parameters.
- **`forcefield_and_simulation_data.json`**: The single, consolidated source for all atom, bond, angle, dihedral, cross-term, bead, contaminant, field, synthesis, and validation parameters. Used by all simulation phases and the recommender. Now includes:
  - Advanced dihedral/torsion terms (PEI, sulfone, etc.)
  - Cross-terms for PFAS–membrane, NOM–membrane, microplastic–membrane
  - Coarse-grained beads for pathogens (E. coli, rotavirus, norovirus, adenovirus)
  - Microplastic types (PET, PS, PE, PP)
  - Regeneration chemistry and kinetics (hypochlorite, peroxide)
  - NOM subtypes and interaction parameters
  - Advanced pore-size/defect statistics
  - Explicit validation/field data integration
- **`output/`**: Stores all simulation results and generated plots, organized by phase and plot type.
- **`EXPERIMENT_INTEGRATION.md`**: How to use this code with experiments and the larger project.
- **`CODE_STRUCTURE.md`**: Technical breakdown of each module and file.

### Output Folder Structure
- **Phase 1 output:** `output/phase1/` (macroscale simulation results)
- **Phase 2 output:** `output/phase2/` (hybrid structure results)
- **Phase 3 output:** `output/phase3/` (LAMMPS atomistic simulation results, per-membrane subfolders)
- **Phase 4 output:** `output/phase4/` (chemical/biological simulation results)
- **All-phase summary:** `output/simulation_results_all_phases.xlsx`
- **Plots:** `output/plots/` (organized by analysis type, e.g. `output/plots/oil_rejection_summary/`, `output/plots/flux_vs_thickness_per_pressure/`, etc.)
- **Data files:** See `forcefield_and_simulation_data.json` for all simulation parameters and force field data.
- **Experiment integration:** See `EXPERIMENT_INTEGRATION.md` for how to connect code and lab data.
- **Technical code structure:** See `CODE_STRUCTURE.md` for module and file details.

---

## 📊 Comprehensive Data Integration
- **All force-field, contaminant, and field parameters** are now in `forcefield_and_simulation_data.json`.
- **New/expanded sections:**
  - Dihedral/torsion terms for carboxyl, epoxy, PEI, sulfone, etc.
  - Cross-terms for PFAS–membrane, NOM–membrane, microplastic–membrane
  - Coarse-grained beads for pathogens (E. coli, rotavirus, norovirus, adenovirus)
  - Microplastic types (PET, PS, PE, PP)
  - Regeneration chemistry and kinetics (hypochlorite, peroxide)
  - NOM subtypes and interaction parameters
  - Advanced pore-size/defect statistics
  - Explicit validation/field data integration
- **All simulation scripts and modules** reference this file for parameters, ensuring consistency and extensibility.
- **To extend:** Add new contaminants, membrane types, or experimental/field data by updating the relevant section in the JSON. Placeholders are provided for easy extension.

---

## 📈 Scientific Rigor, Validation, and Data Sufficiency
- All models and parameters are sourced from peer-reviewed literature (see `monitoring/references.py`).
- Validation is tracked in `validation_log.json`, comparing simulation results to lab data (flux, C/O, ID/IG, etc.).
- Data sufficiency is regularly reviewed; see `CODE_STRUCTURE.md` for required and recommended datasets.
- Gaps (e.g., NOM, regeneration kinetics, nanochannel statistics) are noted and can be filled as new data becomes available.
- The JSON includes explicit placeholders for future data and validation/field integration.

---

## 🔁 Simulation ↔ Experiment Integration
- Use the code to select promising membrane designs before lab work.
- Update the JSON with new experimental or field data as it becomes available.
- See `EXPERIMENT_INTEGRATION.md` for details on workflow and data feedback.

---

## 📚 References and Further Details
- See `CODE_STRUCTURE.md` for a technical breakdown of each module and all data sections.
- See inline code comments for implementation details.
- For user-facing instructions and workflow overview, see this README and the code docstrings.

_Last updated: June 2025, with full error handling, OPLS dihedral support, and comprehensive data integration._
