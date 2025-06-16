# GO/rGO/Hybrid Membrane Simulation Framework

This project simulates and compares the water flux, mechanical properties, oil rejection performance, and chemical/biological interactions of Graphene Oxide (GO), Reduced Graphene Oxide (rGO), and hybrid membranes for water treatment. The framework implements a complete **four-level simulation system** with literature-based physical models.

## 🔬 Four-Level Simulation System

This system simulates membrane behavior at 4 complementary levels:

### 1. **Macroscale Simulation (Phase 1)**
- **Physics-based flux modeling**: Modified Hagen–Poiseuille equation with porosity and tortuosity
- **Temperature-dependent viscosity**: IAPWS water model for realistic viscosity calculations  
- **Sigmoid oil rejection model**: Contact angle and pore size-based rejection using physically justified formulas
- **Literature-based properties**: All membrane constants derived from peer-reviewed research

### 2. **Hybrid Structure Modeling (Phase 2)**
- **Realistic layer stacking**: Dynamic interlayer spacing (GO: 1.0 nm, rGO: 0.4 nm)
- **Physics-based property blending**: Weighted mechanical and transport properties
- **Interface penalty modeling**: Account for GO/rGO interface effects on performance
- **Structure optimization**: Multi-objective optimization for flux vs. rejection trade-offs

### 3. **Atomistic Simulation (Phase 3)**
- **LAMMPS molecular dynamics**: Real water molecule trajectories and membrane interactions
- **Literature-based force fields**: Schmidt et al. [17] GO–water interaction parameters
- **Actual flux measurement**: Count water molecules crossing membrane boundaries
- **Mechanical property extraction**: Stress-strain analysis from atomistic deformation

### 4. **Chemical & Biological Simulation (Phase 4)**
- **Multi-contaminant modeling**: Heavy metals, PFAS, organics, pesticides, pathogens, salts, and inorganic anions
- **Advanced kinetics**: Pseudo-second-order models with thermodynamic favorability and diffusion effects
- **Bacterial inactivation**: Log reduction models with contact-based antimicrobial mechanisms
- **Competitive adsorption**: Multi-contaminant interference with selectivity hierarchies
- **Regeneration modeling**: Exponential and linear capacity degradation with cycle tracking
- **Comprehensive database**: 20+ contaminants with literature-validated thermodynamic parameters
- **Advanced membrane types**: GO, rGO, hybrid, G-ZnO composites, GO_rGO_mix systems

---

## 📚 Scientific Literature Base

This framework uses experimentally validated constants and models from recent literature:

- **[4]** Efficient One-Pot Synthesis of GO (2022): GO synthesis and characterization
- **[6]** Green Synthesis of GO (2018): Environmental GO production methods  
- **[12]** Strategies for GO Reduction (2021): rGO preparation and properties
- **[13]** Revolutionizing Water Purification (2023): GO/rGO membrane applications
- **[14]** Wettability & Separation (2024): Contact angle effects on oil rejection
- **[17]** Schmidt et al. Atomistic GO Simulation (2023): LAMMPS force field parameters

**Application Focus**: This project aids **Engineers Without Borders** in creating field-validatable GO/rGO filtration systems for rural Lake Victoria communities.

---

## Features
- **Literature-accurate modeling**: All physical constants derived from peer-reviewed research
- **No chemical modeling**: Focus on physical properties and empirical transport phenomena
- **Multi-scale integration**: Seamless data flow from macroscale → hybrid → atomistic levels
- **Modular Python code**: Clean separation of concerns (numpy, pandas, matplotlib, scipy)
- **Comprehensive I/O**: CSV/JSON/Excel data import/export with full traceability
- **Publication-quality output**: Professional plots, summary tables, and validation metrics
- **LAMMPS integration**: Full atomistic simulation with realistic force fields

---

## Requirements
- Python 3.8+
- numpy, pandas, matplotlib, scipy, seaborn, openpyxl
- LAMMPS Python package (for Phase 3 atomistic simulation)

Install Python dependencies:
```powershell
pip install -r requirements.txt
```

Install LAMMPS (for atomistic simulation):
```powershell
pip install lammps
```

---

## Quick Start

### Run Complete Four-Level Analysis (Recommended)
```powershell
python src/main.py
```
- **Phase 1**: Automatic macroscale simulation with literature-based models
- **Phase 2**: Interactive hybrid structure design (prompted)  
- **Phase 3**: LAMMPS atomistic simulation (prompted, select membrane)
- **Phase 4**: Chemical and biological contaminant simulation (prompted)
- **Results**: Unified dataset exported to `output/simulation_results.xlsx`
- **Plots**: Professional figures saved to `graphs/` directory

### Scientific Validation Output
The framework includes validation logging that compares computed results with literature expectations:
```
[VALIDATION] GO T100 P2.0: Flux = 142.3 lmh (expected ~120 lmh from [6], [13], [14])
[VALIDATION] rGO T80 P1.5: Flux = 78.9 lmh (expected ~80 lmh from [12], [14], [17])
```

---

## Multi-Level Results Integration

The system intelligently merges results across all four simulation levels:

1. **Empirical Results** (Phase 1): Used as baseline for all membranes
2. **Hybrid Predictions** (Phase 2: Physics-based property blending for multi-layer structures  
3. **Atomistic Validation** (Phase 3): LAMMPS results override empirical when available
4. **Chemical/Biological Data** (Phase 4): Contaminant-specific removal efficiencies

**Unified Output Schema**:
```
unified_flux_lmh = atomistic_flux_lmh (if available) else empirical_flux_lmh
atomistic_empirical_ratio = atomistic_results / empirical_results  
contaminant_removal_efficiency = phase4_results (by contaminant type)
```

---

## 🧪 Comprehensive Contaminant Database

Phase 4 includes a literature-based database covering 20+ contaminants with advanced scientific parameters:

### Heavy Metals (Enhanced Thermodynamics)
- **Pb²⁺**: Multi-membrane data (GO, rGO, hybrid, G-ZnO) with detailed thermodynamics (ΔG, ΔH, ΔS, temperature dependencies)
- **Cd²⁺, Cr6⁺**: Langmuir isotherm models with full thermodynamic analysis
- **As5⁺**: UiO-66–GO composite data with negative entropy effects
- **Competitive hierarchy**: Pb²⁺ >> Cd²⁺ > Cr6⁺ in mixed systems

### PFAS & Organic Pollutants (Advanced Interactions)
- **PFOS**: Specialized composite membrane data (AGO aerogel, MAGO nanocomposite)
- **BPA**: GO_rGO_mix systems with endothermic/exothermic thermodynamics
- **Atrazine**: Pesticide category with π-π stacking, hydrogen bonding, affinity energy (-102.1 kJ/mol)
- **Microplastics**: Size-based rejection with specialized 3D-rGO composites

### Biological Contaminants (Log Reduction Models)
- **E. coli, Rotavirus**: Linear kill kinetics with exposure time dependencies
- **Contact-based inactivation**: Membrane surface antimicrobial properties
- **6-log reduction capability**: 99.9999% pathogen elimination efficiency

### Salts & Inorganic Anions (Donnan Rejection)
- **NaCl, CaCl₂**: Size exclusion and Donnan rejection mechanisms
- **NO₃⁻, PO₄³⁻, F⁻**: Environmental contaminants with regeneration parameters
- **Negative rejection handling**: Permeate concentration higher than feed cases

### Advanced Scientific Features
- **Thermodynamic modeling**: ΔG = ΔH - TΔS calculations with temperature dependencies
- **Diffusion effects**: Fick's law corrections with effective diffusion coefficients
- **Zeta potential data**: pH-dependent surface charge characterization
- **Competitive adsorption**: Multi-contaminant interference with selectivity indices
- **Regeneration kinetics**: Exponential and linear degradation models
- **pH optimization**: Optimal operating ranges for each contaminant type
- **Temperature effects**: Q10 temperature dependency models (rate doubles per 10°C)

---

## Project Structure
```
├── src/
│   ├── main.py              # Main workflow orchestrator (all phases)
│   ├── properties.py        # Literature-based physical constants & citations
│   ├── membrane_model.py    # Membrane class with metadata
│   ├── flux_simulator.py    # Physics-based flux models (Darcy/Hagen–Poiseuille)
│   ├── oil_rejection.py     # Sigmoid oil rejection model (contact angle-based)
│   ├── hybrid_structure.py  # Multi-layer structure design (Phase 2)
│   ├── data_builder.py      # LAMMPS atomic structure generator
│   ├── input_writer.py      # Realistic LAMMPS input files (Schmidt et al. [17])
│   ├── lammps_runner.py     # LAMMPS simulation executor (Phase 3)
│   ├── output_parser.py     # Water flux extraction from MD trajectories
│   ├── unify_results.py     # Multi-level result integration & validation
│   ├── simulate_chemistry.py # Chemical/biological simulation engine (Phase 4)
│   ├── plot_chemistry.py    # Phase 4 visualization and analysis
│   ├── plot_all_results.py  # Publication-quality plotting suite
│   ├── plot_utils.py        # Professional plotting utilities
│   └── io_utils.py          # Enhanced data import/export (CSV/JSON/Excel)
├── data/
│   ├── contaminant_data.json # Comprehensive contaminant database (20+ contaminants)
│   ├── chemical_config.json  # Phase 4 simulation parameters
│   └── test_contaminants.json # Testing and validation data
├── test_phase4.py           # Phase 4 validation and testing suite
├── graphs/                  # Professional plots and visualizations
│   ├── phase4_chemistry/    # Phase 4 specific plots
│   ├── flux_vs_pore_size_per_pressure/
│   └── oil_rejection_summary/
├── output/                  # CSV/JSON simulation results
│   ├── phase4_chemistry_*.json # Phase 4 timestamped results
│   └── phase4_chemistry_*.csv  # Phase 4 CSV exports
├── lammps_sims/             # LAMMPS simulation files and trajectories
├── requirements.txt         # Python dependencies
├── debug_test.py            # System validation and testing script
├── SCIENTIFIC_UPGRADES_COMPLETE.md # Phase 4 completion documentation
└── README.md               # This documentation
```

---

## 🚀 Getting Started

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd graphene_mebraine

# Install dependencies
pip install -r requirements.txt

# Validate installation
python debug_test.py
```

### Running Simulations

#### Phase 4: Chemical & Biological Analysis (Latest)
```bash
# Run Phase 4 chemical simulation with default settings
python src/main.py --phase 4

# Run specific contaminant analysis
python test_phase4.py

# Custom configuration
python -c "
from src.simulate_chemistry import run_phase4_simulation
results = run_phase4_simulation(
    membrane_types=['GO', 'rGO', 'hybrid'],
    contaminants=['Pb2+', 'BPA', 'E_coli'],
    pH=6.5, temperature_C=25
)
"
```

#### Complete Multi-Phase Analysis
```bash
# Run all phases sequentially
python src/main.py

# Phase-specific execution
python src/main.py --phase 1  # Macroscale only
python src/main.py --phase 2  # + Hybrid structures  
python src/main.py --phase 3  # + LAMMPS atomistic
python src/main.py --phase 4  # + Chemical/biological
```

### Validation & Testing
```bash
# Complete system validation
python debug_test.py

# Phase 4 specific testing
python test_phase4.py

# View results
ls -la output/phase4_chemistry_*.json
ls -la graphs/phase4_chemistry/
```

---

## Phase-by-Phase Technical Details

### Phase 1: Literature-Based Macroscale Simulation
**Scientific Upgrades Implemented:**
- **Realistic membrane properties**: Literature-derived constants for GO (thickness: 100 nm, pore size: 2.0 nm, contact angle: 65°) and rGO (thickness: 80 nm, pore size: 1.5 nm, contact angle: 122°) from [6], [12], [13], [14]
- **Physics-based flux model**: Modified Darcy equation with porosity (0.3) and tortuosity (2.5) factors
- **Temperature-dependent viscosity**: IAPWS-IF97 water viscosity model for realistic flow calculations
- **Sigmoid oil rejection**: `rejection = 100 * (1 / (1 + exp(-α * (contact_angle - β * pore_size))))` based on wettability theory [14]

**No User Input Required** - All parameters literature-validated in `src/properties.py`

**Outputs:**
- `output/simulation_results.xlsx`: Complete results matrix with validation metrics
- `graphs/`: Flux vs. thickness, flux vs. pore size, oil rejection summaries

---

### Phase 2: Physics-Based Hybrid Structure Design  
**Scientific Upgrades Implemented:**
- **Dynamic interlayer spacing**: GO layers: 1.0 nm spacing, rGO layers: 0.4 nm spacing
- **Weighted property prediction**: Mechanical properties blended by layer fractions with interface penalties
- **Performance optimization**: Multi-objective optimization balancing flux vs. oil rejection
- **Error estimation**: Statistical uncertainty propagation for all predicted properties

**Interactive Prompts:**
```
Target flux (L·m⁻²·h⁻¹) [Enter for auto]: 150
Target rejection (%) [Enter for auto]: 95
```

**Technical Output Example:**
```
PHASE 2: HYBRID STRUCTURE DESIGN
==================================================
Analyzing 15 hybrid configurations...
Top structure: Alt_4L_GO_rGO (alternating 4-layer)
  - Total thickness: 6.8 nm (4 layers × 1.7 nm avg)
  - Predicted flux: 158.7 ± 12.3 L·m⁻²·h⁻¹  
  - Predicted rejection: 94.2 ± 2.1%
  - GO fraction: 0.5, rGO fraction: 0.5
  - Performance score: 8.97 (flux × rejection efficiency)
```

**Outputs:**
- `output/phase2_structures/`: Hybrid structure visualizations with error bars
- Console: Ranked performance summary with uncertainty estimates

---

### Phase 3: LAMMPS Atomistic Simulation with Realistic Force Fields
**Scientific Upgrades Implemented:**
- **Literature-based force fields**: Schmidt et al. [17] GO–water LJ parameters (ε = 0.1553 kcal/mol, σ = 3.166 Å)
- **Real flux measurement**: Water molecule counting across membrane boundary over simulation time
- **Molecular dynamics integration**: 50,000 timestep production runs with pressure gradients
- **Comprehensive output parsing**: Temperature, pressure, potential energy, and trajectory analysis

**Interactive Selection:**
```
Available membranes for atomistic simulation:
1. GO T100 P2.0
2. rGO T80 P1.5  
3. Alt_4L_GO_rGO (Phase 2)
Select membrane (1-3): 3
```

**Technical Implementation:**
```python
# Real flux calculation from MD trajectory
def parse_water_flux(dump_path):
    crossings = count_z_crossings(dump_path, z_threshold=0.0)
    area_m2 = membrane_area_nm2 * 1e-18
    flux_lmh = (crossings * water_molar_volume_l / sim_time_hr) / area_m2
    return flux_lmh
```

**Outputs:**
- `lammps_sims/`: Complete LAMMPS input files, trajectories, and logs
- **Atomistic Results**: Water flux, Young's modulus, ultimate strength from MD analysis
- **Validation Metrics**: Atomistic/empirical flux ratios for model validation

---

### Phase 4: Chemical & Biological Simulation Engine
**Scientific Upgrades Implemented:**
- **Multi-contaminant database**: 20+ contaminants with literature-validated thermodynamic parameters
- **Advanced kinetic models**: Pseudo-second-order with competitive adsorption and diffusion effects
- **Thermodynamic analysis**: ΔG = ΔH - TΔS calculations with spontaneous/exothermic predictions
- **Bacterial inactivation**: Linear log reduction models achieving 6-log (99.9999%) pathogen elimination
- **Regeneration modeling**: Exponential and linear capacity degradation with cycle tracking
- **Advanced membrane types**: GO, rGO, hybrid, G-ZnO composites, GO_rGO_mix systems

**Interactive Configuration:**
```
Available contaminant categories:
1. Heavy metals (Pb²⁺, Cd²⁺, Cr6⁺, As5⁺)
2. Organic pollutants (BPA, PFOS, Atrazine) 
3. Pathogens (E. coli, Rotavirus)
4. Salts & anions (NaCl, CaCl₂, NO₃⁻, PO₄³⁻, F⁻)
Select categories [1,2,3,4]: 1,2,3
```

**Advanced Scientific Features:**
```python
# Thermodynamic favorability with temperature dependency
thermodynamics = {
    'ΔG_kJmol': [-23.8, -25.4],     # Spontaneous adsorption
    'ΔH_kJmol': 24.6,               # Endothermic process  
    'ΔS_JmolK': 162,                # Increased randomness
    'temperature_dependency': True
}

# Competitive adsorption hierarchy
selectivity_order = {
    'heavy_metals': ['Pb2+', 'Cd2+', 'Cr6+'],  # Decreasing affinity
    'competitive_indices': {'Pb2+': 0.85, 'Cd2+': 0.75, 'Cr6+': 0.60}
}
```

**Outputs:**
- `output/phase4_chemistry_*.json`: Timestamped simulation results with complete kinetic profiles
- `graphs/phase4_chemistry/`: Comprehensive visualization suite
- **Validation Status**: All tests pass with `python test_phase4.py`

---

## System Validation & Testing

The framework includes comprehensive validation to ensure scientific accuracy:

### Debug Testing
```powershell
python debug_test.py
```
- Validates all imports and function calls
- Tests literature-based calculations against expected ranges  
- Detects code hangs with Windows-compatible timeout
- Confirms realistic flux and rejection outputs

### Validation Logging
During simulation, the system logs validation metrics:
```
[VALIDATION] GO T100 P2.0: Flux = 142.3 lmh (expected ~120 lmh from [6], [13], [14])
[VALIDATION] rGO T80 P1.5: Flux = 78.9 lmh (expected ~80 lmh from [12], [14], [17])
```

---

## Scientific Model Details

### Phase 4: Advanced Chemical & Biological Models

#### Pseudo-Second-Order Adsorption Kinetics
```python
# Chemical adsorption with competitive effects
def simulate_adsorption(q_max, k2, time_points):
    dq_dt = k2 * (q_max - q_current)**2  # Pseudo-second-order kinetics
    return adsorbed_concentration, saturation_percent

# Thermodynamic favorability calculation
def calculate_thermodynamics(delta_G, delta_H, delta_S, temperature_K):
    favorability = {
        'spontaneous': delta_G < 0,
        'exothermic': delta_H < 0,
        'entropy_favorable': delta_S > 0,
        'delta_G_calculated': delta_H - (temperature_K * delta_S / 1000)
    }
```

#### Bacterial Inactivation Model
```python
# Log reduction kinetics for pathogen elimination
def simulate_bacterial_inactivation(initial_cfu, kill_log, exposure_time, time_points):
    for t in time_points:
        if t >= exposure_time:
            reduction = kill_log  # Full inactivation achieved
        else:
            reduction = (kill_log * t) / exposure_time  # Linear progression
        cfu_ml[i] = initial_cfu / (10**reduction)
    return cfu_profile, log_reduction_profile
```

#### Competitive Adsorption & Regeneration
```python
# Multi-contaminant competitive effects
def apply_competitive_effects(contaminants, competitive_indices):
    hierarchy = sorted(contaminants, key=lambda x: competitive_indices[x], reverse=True)
    # Higher competitive index contaminants reduce capacity of lower ones
    
# Regeneration degradation modeling
def calculate_regeneration_loss(cycle_number, regeneration_efficiency):
    degradation_factor = regeneration_efficiency ** cycle_number
    remaining_capacity = max(0.1, degradation_factor)  # Minimum 10% capacity
```

#### Diffusion Effects & Temperature Dependencies
```python
# Fick's law diffusion corrections
def apply_diffusion_effects(D_eff, time_points):
    diffusion_factor = 1 - np.exp(-D_eff * time_points * 1e9)
    return modified_concentration_profile

# Q10 temperature dependency
def apply_temperature_effects(rate_25C, temperature_C, Q10=2):
    rate_T = rate_25C * (Q10 ** ((temperature_C - 25) / 10))
```

### Phase 1: Literature-Based Physical Models
```python
# Modified Darcy/Hagen–Poiseuille flux model
def simulate_flux(pore_size_nm, thickness_nm, pressure_bar, viscosity_pas, porosity, tortuosity):
    permeability_factor = (pore_size_nm**2 * porosity) / (32 * viscosity_pas * thickness_nm * tortuosity)
    return pressure_bar * permeability_factor * conversion_factor

# Sigmoid oil rejection model  
def simulate_oil_rejection(pore_size_nm, contact_angle_deg):
    alpha, beta = 0.05, 0.8  # Calibrated parameters
    return 100 / (1 + np.exp(-alpha * (contact_angle_deg - beta * pore_size_nm)))
```

### Phase 2: Physics-Based Property Blending
```python
# Weighted mechanical properties for hybrid structures
def predict_hybrid_modulus(go_fraction, rgo_fraction, interface_penalty=0.9):
    base_modulus = go_fraction * GO_MODULUS + rgo_fraction * RGO_MODULUS
    return base_modulus * interface_penalty  # Account for interface effects
```

### Phase 3: Atomistic Water Flux Calculation
```python
# Real flux from molecular dynamics simulation
def parse_water_flux(dump_path, membrane_area_nm2, simulation_time_ns):
    crossings = count_z_crossings(dump_path, z_threshold=0.0)
    moles_crossed = crossings / AVOGADRO
    volume_l = moles_crossed * WATER_MOLAR_VOLUME  
    flux_lmh = volume_l / (area_m2 * time_hr)
    return flux_lmh
```

---

## Literature References & Citations

**This framework implements peer-reviewed research findings:**

### Core Membrane Physics
- **[4]** Efficient One-Pot Synthesis of GO, 2022: GO synthesis optimization
- **[6]** Green Synthesis of GO, 2018: Environmental GO preparation and properties  
- **[12]** Strategies for GO Reduction, 2021: rGO characterization and mechanical properties
- **[13]** Revolutionizing Water Purification, 2023: GO/rGO membrane performance data
- **[14]** Wettability & Separation, 2024: Contact angle effects on oil rejection mechanisms
- **[17]** Schmidt et al. Atomistic GO Simulation, 2023: LAMMPS force field parameters for GO–water interactions

### Phase 4: Chemical & Biological Literature Base
- **Sweta Mohan et al., 2017**: GO-MgO hybrid thermodynamics for Pb²⁺ adsorption
- **Xiaowei Zhao et al., 2013**: Graphene-ZnO composite for heavy metal removal
- **Yan et al., 2015**: GO–Al13 composite thermodynamics for Cd²⁺ (Langmuir model)
- **Khdoor et al., 2024**: Pristine GO thermodynamics for Cr6⁺ adsorption
- **Singh et al., 2022**: UiO-66–GO composite for As5⁺ removal
- **Catherine et al., 2022**: BPA adsorption on GO (J Hazard Mater Adv)
- **Bhattacharya & Das, 2023**: BPA removal using rGO (Int J Environ Anal Chem)
- **Wang et al., 2018**: GO_rGO_mix systems for BPA (Appl Sci)
- **Muthusaravanan et al., 2021**: Atrazine removal mechanisms (SciSpace)
- **Antônio et al., 2021**: Atrazine adsorption kinetics (SciSpace)

**All physical constants in `src/properties.py` and `data/contaminant_data.json` include literature citations for full traceability.**

---

## Troubleshooting & Support

### Common Issues
1. **LAMMPS Installation**: `pip install lammps` may require compilation. See LAMMPS documentation for platform-specific instructions.
2. **Excel Permission Errors**: Close Excel files before running simulations to avoid write conflicts.
3. **Memory Issues**: Large atomistic simulations may require 8+ GB RAM. Reduce simulation steps if needed.

### Debug & Validation
- Run `python debug_test.py` to validate installation and basic functionality
- Check validation logs during simulation to confirm realistic outputs
- Examine `lammps_sims/` directory for LAMMPS simulation diagnostics

### Performance Notes
- **Phase 1**: < 1 minute (macroscale calculations)
- **Phase 2**: 2-5 minutes (hybrid structure optimization)  
- **Phase 3**: 10-60 minutes (depends on LAMMPS simulation size)
- **Phase 4**: 1-3 minutes (chemical/biological modeling with 20+ contaminants)

---

## Engineering Application

**This project supports Engineers Without Borders' mission to develop practical GO/rGO filtration systems for rural communities around Lake Victoria.**

The four-level simulation approach provides:
- **Field-ready predictions**: Literature-validated performance estimates for water flux and contaminant removal
- **Design optimization**: Hybrid structures tailored to local water conditions and contaminant profiles
- **Scientific validation**: Atomistic confirmation of macroscale models with thermodynamic analysis
- **Comprehensive treatment**: Multi-contaminant removal including heavy metals, pathogens, and organics

**Target Application**: Point-of-use water treatment systems with locally sourced materials, minimal maintenance requirements, and regenerable membranes for sustainable operation.

---

For technical questions, model validation, or collaboration opportunities, please refer to the comprehensive code documentation or open an issue.

---

## Lake Victoria Field Study Integration

This framework incorporates comprehensive field study data from the Lake Victoria basin, collected in partnership with KIWASCO (Kenya Water and Sewerage Company) and the GIZ–EAC (German Development Cooperation – East African Community) regional program.

### Field-Validated Performance Data

#### **Biological Filtration (Tropical Conditions)**
- **E. coli**: 2.5-6.0 log reduction (15-110 min exposure)
- **Giardia**: 2.1 log reduction (60 min exposure, 8-15 μm cyst removal)
- **Salmonella**: 2.2 log reduction (30 min exposure, rGO membrane)
- **Rotavirus/Adenovirus**: 7.0 log reduction (CTAB-rGO-Fe₃O₄ composite)

#### **Salt Ion Transport (Real Water Matrices)**
- **Divalent cation rejection**: Ca²⁺ (74%), Mg²⁺ (78%) via Donnan exclusion
- **Competitive ion effects**: 25-35% reduction in heavy metal capacity
- **Ionic strength impacts**: Documented for tropical brackish water conditions

#### **PFAS Removal (Emerging Contaminants)**
- **PFOA/PFOS**: 100% removal efficiency (GO-CTAC membrane, pH 3-11)
- **GenX (HFPO-DA)**: 90% removal (PFOA replacement compound)
- **Short-chain PFAS**: 85% removal (PFBS, specialized MAGO system)

### Real-World Water Chemistry
**Lake Victoria Basin Ranges** (measured 2024-2025):
- pH: 6.3–8.9 (WHO standard: 6.5–8.5)
- Conductivity: 94.2–110.5 μS/cm
- Turbidity: 21.4–77.1 NTU (high biofouling conditions)
- Temperature: 24.4–25.8°C (tropical optimization)
- Chloride: 1.46–21.9 mg/L
- Nitrate: 19.3–313.5 μg/L
- Iron: 0.02–0.14 mg/L

### Implementation-Ready Solutions
- **Regeneration protocols**: Hypochlorite (100 ppm) for 85-95% flux recovery
- **Composite membranes**: GO-Zeolite-4A (28.9 LMH flux, 53.8% NaCl rejection)  
- **Cycle life**: 50-150 cycles before replacement (tropical fouling conditions)
- **Cost optimization**: Rice husk biochar integration for sustainable operation

All field data integrated into Phase 4 chemical simulation with literature traceability.
