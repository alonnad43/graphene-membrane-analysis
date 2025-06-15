# GO/rGO/Hybrid Membrane Simulation Framework

This project simulates and compares the water flux, mechanical properties, and oil rejection performance of Graphene Oxide (GO), Reduced Graphene Oxide (rGO), and hybrid membranes for oil–water separation. The framework supports a complete multi-phase workflow:

- **Phase 1:** Macroscale/empirical simulation of flux, rejection, and mechanical properties
- **Phase 2:** Hybrid membrane structure design and performance prediction
- **Phase 3:** Atomistic (LAMMPS) simulation for selected membranes

---

## Features
- Modular, well-documented Python code (numpy, pandas, matplotlib, scipy)
- CSV/JSON/Excel data import/export
- Publication-quality plots and summary tables
- LAMMPS integration for atomistic simulation (Phase 3)
- All physical/empirical constants in `src/properties.py`

---

## Requirements
- Python 3.8+
- numpy, pandas, matplotlib, scipy, seaborn, openpyxl
- LAMMPS Python package (for Phase 3)

Install Python dependencies:
```powershell
pip install -r requirements.txt
```

Install LAMMPS (for atomistic simulation):
```powershell
pip install lammps
```

---

## Usage

### Run All Phases (Recommended)
```powershell
python src/main.py
```
- You will be prompted to run Phase 2 (hybrid design) and Phase 3 (LAMMPS atomistic simulation).
- Results are exported to `output/` and plots to `graphs/`.

### Phase 3 (LAMMPS) Details
- When prompted, select a membrane for atomistic simulation.
- LAMMPS input/data files and logs are saved in `lammps_sims/`.
- Atomistic results are parsed and merged with macroscale results.

---

## Project Structure
```
├── src/
│   ├── main.py              # Main workflow (all phases)
│   ├── properties.py        # Physical/empirical constants
│   ├── membrane_model.py    # Membrane class
│   ├── flux_simulator.py    # Macroscale flux models
│   ├── oil_rejection.py     # Oil rejection models
│   ├── hybrid_structure.py  # Hybrid structure design (Phase 2)
│   ├── data_builder.py      # LAMMPS atomic structure builder
│   ├── input_writer.py      # LAMMPS input file generator
│   ├── lammps_runner.py     # LAMMPS simulation runner (Phase 3)
│   ├── output_parser.py     # LAMMPS output parser
│   ├── unify_results.py     # Combine results from all phases
│   ├── plot_all_results.py  # Comprehensive plotting
│   ├── plot_utils.py        # Plotting utilities
│   └── io_utils.py          # Data import/export helpers
├── data/                    # Input/experimental data
├── output/                  # Results (Excel, CSV, JSON)
├── graphs/                  # Plots and figures
├── lammps_sims/             # LAMMPS simulation files
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Phase-by-Phase Details

### Phase 1: Macroscale Simulation
**What it does:**
- Simulates water flux, oil rejection, and mechanical properties for GO, rGO, and hybrid membranes using empirical/physical models.
- Generates summary tables and plots for all membrane types and parameter sweeps.

**Inputs required:**
- No user input required; all parameters are defined in `src/properties.py` (thickness, pore size, pressure, etc.).

**Outputs:**
- Results table (`output/simulation_results.xlsx`)
- Plots in `graphs/` (e.g., flux vs. thickness, flux vs. pore size, oil rejection summary)

---

### Phase 2: Hybrid Structure Design
**What it does:**
- Designs and analyzes hybrid GO/rGO membrane structures by combining layers in different configurations.
- Predicts performance (flux, rejection, mechanical properties) for each structure.
- Ranks and visualizes top-performing hybrid designs.

**Inputs required:**
- User is prompted:
  - `target_flux`: Desired minimum water flux (L·m⁻²·h⁻¹). Enter a value or press Enter for automatic/optimal selection.
  - `target_rejection`: Desired minimum oil rejection (%). Enter a value or press Enter for automatic/optimal selection.

**Example prompt:**
```
membrane_type: hybrid
Target flux (L·m⁻²·h⁻¹) [Enter for auto]: 5000
Target rejection (%) [Enter for auto]: 99
```

**Outputs:**
- Visualizations of hybrid structures (`output/phase2_structures/`)
- Console summary, e.g.:
  ```
  PHASE 2: HYBRID STRUCTURE DESIGN
  ==================================================
  Phase 2 visualizations saved to: .../output/phase2_structures
  Phase 2 Complete: 15 structures analyzed
  Top structure: Alt_4L_GO
  Predicted flux: 5698.5 L·m⁻²·h⁻¹
  Predicted rejection: 92.0%
  Phase 2 structures added to results. Total entries: 195
  ```
- Top structures and their predicted properties are added to the main results table for further analysis or Phase 3 simulation.

**Explanation of output text:**
- `membrane_type: hybrid` — Indicates the type of membrane being designed (hybrid = combination of GO and rGO layers).
- `target_flux` — The minimum water flux you want the hybrid structure to achieve.
- `target_rejection` — The minimum oil rejection you want the hybrid structure to achieve.
- `Top structure: Alt_4L_GO` — The best-performing structure found (e.g., alternating 4-layer GO/rGO stack).
- `Predicted flux` and `Predicted rejection` — Model-predicted performance for the top structure.
- `Phase 2 structures added to results. Total entries: ...` — Number of structures/results now in the main dataset.

---

### Phase 3: Atomistic (LAMMPS) Simulation
**What it does:**
- Runs detailed molecular dynamics simulations for a selected membrane using LAMMPS.
- Generates atomistic-level predictions for water flux, mechanical properties, and (optionally) contact angle.
- Integrates atomistic results with macroscale data for unified analysis.

**Inputs required:**
- User is prompted to select a membrane (GO, rGO, or hybrid) for atomistic simulation.
- No manual editing of files is needed; all input/data files are generated automatically.

**Outputs:**
- LAMMPS input/data files and logs in `lammps_sims/`
- Atomistic simulation results (flux, modulus, strength, etc.) merged into `output/simulation_results.xlsx`
- Console summary of LAMMPS results and any errors

---

## Example User Prompts and Output Explained

When running the main script, you may see prompts and output like:
```
Run Phase 2 (Hybrid Structure Design)? [yes/no]: yes
Target flux (L·m⁻²·h⁻¹) [Enter for auto]: 5000
Target rejection (%) [Enter for auto]: 99
PHASE 2: HYBRID STRUCTURE DESIGN
==================================================
Phase 2 visualizations saved to: C:\Users\ramaa\Documents\graphene_mebraine\output\phase2_structures
Phase 2 Complete: 15 structures analyzed
Top structure: Alt_4L_GO
Predicted flux: 5698.5 L·m⁻²·h⁻¹
Predicted rejection: 92.0%
Phase 2 structures added to results. Total entries: 195
```
- This means you requested a hybrid membrane with at least 5000 L·m⁻²·h⁻¹ flux and 99% rejection. The code analyzed 15 possible structures, and the best one found is `Alt_4L_GO` with predicted flux and rejection shown. All results are now included in the main output file.

---

## Notes
- No chemical modeling; only physical/empirical properties are used.
- All simulation parameters are easily adjustable in `src/properties.py`.
- For troubleshooting LAMMPS, check logs in `lammps_sims/`.

---

For questions or issues, please refer to the code comments or open an issue.
