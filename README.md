# GO/rGO Membrane Simulation

This project simulates and compares the water flux, mechanical properties, and oil rejection performance of Graphene Oxide (GO), Reduced Graphene Oxide (rGO), and hybrid membranes for oil–water separation.

## Features
- Simulate water flux as a function of thickness and pore size
- Compare oil rejection efficiency
- Analyze mechanical stability under pressure
- Import/export data (CSV/JSON)
- Plot results using matplotlib

## File Structure
- /src: Source code
- /data: Experimental input data
- /models: (Optional) SolidWorks reference files

## Requirements
- Python 3.8+
- numpy, pandas, matplotlib, scipy

## Usage
1. Place your data files in the /data directory.
2. Run the simulation: `python src/main.py`
3. View plots and results in the output directory or console.

## Notes
- No chemical modeling is performed.
- All parameters are based on published literature.

## Project Structure

The project is organized for clarity and modularity. Each file has a specific role:

```
go_rgo_simulation/
│
├── main.py                  # Main entry point for simulations. Loads data, runs simulations, and generates plots.
├── properties.py            # Holds all constants and experimental values used throughout the project.
├── membrane_model.py        # Defines the Membrane class and related data structures (geometry, layers, combinations).
├── flux_simulator.py        # Functions to compute water flux as a function of thickness, pore size, and pressure.
├── oil_rejection.py         # Functions to estimate oil rejection efficiency for each membrane type.
├── plot_utils.py            # Utility functions for plotting flux, rejection, and other results using matplotlib.
├── io_utils.py              # Utility functions for loading and saving data in CSV/JSON formats.
├── README.md                # Project overview, instructions, and file descriptions.
├── copilot_instructions.md  # Custom instructions for GitHub Copilot to ensure code quality and project standards.
├── data/                    # Directory for experimental input data (CSV/JSON files).
└── models/                  # (Optional) Directory for SolidWorks or reference model files.
```

### File Descriptions
- **main.py**: Orchestrates the simulation workflow. Loads membrane data, runs water flux and oil rejection simulations, and generates comparative plots.
- **properties.py**: Stores all physical constants, empirical values, and simulation ranges (e.g., pore sizes, thicknesses, flux, mechanical properties).
- **membrane_model.py**: Contains the `Membrane` class, encapsulating all relevant membrane properties and enabling easy data handling.
- **flux_simulator.py**: Implements functions to simulate water flux through membranes, including thickness and pore size effects.
- **oil_rejection.py**: Provides logic to estimate oil rejection efficiency based on membrane type and empirical data.
- **plot_utils.py**: Contains all plotting functions for visualizing simulation results (e.g., flux vs. thickness, oil rejection bar charts).
- **io_utils.py**: Handles data import/export, supporting both CSV and JSON formats for flexibility.
- **README.md**: This file. Explains the project, structure, and usage.
- **copilot_instructions.md**: Contains workspace-specific instructions for GitHub Copilot to ensure code quality and adherence to project guidelines.
- **data/**: Place your experimental or simulation input data here (CSV/JSON).
- **models/**: (Optional) Store reference or CAD model files here.
