# How to Use This Code with Experiments and the Larger Project

This guide explains how to connect the simulation framework to your experimental workflow and to other projects, ensuring a smooth loop between simulation, lab work, and data integration.

---

## 1. Before Experiments: Use Simulation to Guide Lab Work
- Run `main.py` to simulate water flux, oil rejection, and mechanical properties for a wide range of GO/rGO/hybrid membranes.
- Review the output in the `output/` and `graphs/` folders to identify the most promising membrane designs.
- Select a small number of top-performing variants for fabrication in the lab.
- If testing a new GO/rGO synthesis route (e.g., electrochemical, biological, or algae-based), use Phase 4 to simulate expected C/O ratios and reduction effectiveness based on literature data.

---

## 2. During Experiments: Measure Real Properties
- Fabricate the selected membranes using your standard protocols (e.g., drop-casting, LbL).
- Measure:
  - **Porosity, interlayer spacing:** SEM/XRD
  - **Contact angle:** Goniometer
  - **Flux & rejection:** Filtration test bench
  - **C/O ratio (for rGO):** XPS or Raman
  - **Mechanical strength:** Tensile test
- Record all measurements in a structured format (CSV or JSON). See `data/lab_characterization_example.json` for a template.
- For rural or off-grid labs (e.g., Kenya), ensure reduction conditions match field-appropriate techniques:
  - Use solar or electrochemical methods (see `EXPERIMENTAL_SYNTHESIS_NOTES.md`).
  - For electrical conductivity validation in the field, use a portable four-point probe or multimeter.

---

## 3. After Experiments: Feed Real Data Back into the Simulation
- Update the relevant data files with your measured values:
  - **Membrane properties:** Edit `forcefield_and_simulation_data.json` (all membrane, contaminant, and field data are now consolidated here).
  - **Experimental results:** Add to `data/lab_characterization_example.json` or create a new file in the same format.
- Re-run `main.py` to see how the new data affects simulation predictions.
- After updating force field or membrane properties, validate atomistic-level simulation stability using Phase 3 LAMMPS logs (`output/phase3/log.lammps`).
- If new contaminants, pathogens, microplastics, or NOM subtypes are tested, add them to the relevant section in `forcefield_and_simulation_data.json`.
- Use the `validation_and_field_data` section in the JSON to log new lab/field results and validation notes.

---

## 4. Data File Reference (2025+)
- **Membrane, contaminant, force field, synthesis, and validation data:** `forcefield_and_simulation_data.json` (single consolidated source; includes all advanced sections and placeholders for extension)
- **Experimental/lab data:** `data/lab_characterization_example.json`
- **Simulation results and plots:** `output/`, `graphs/`
- **Atomistic simulation logs and structures:** `output/phase3/[membrane_name]/`

---

## 5. How to Use, Extend, and Validate
- To add new functional groups, ions, contaminants, microplastics, NOM subtypes, or field/validation data, update the relevant section in `forcefield_and_simulation_data.json`.
- For new experimental data, add to `data/lab_characterization_example.json` or create a new file in the same format.
- For technical details on how these new sections are parsed and used, see `CODE_STRUCTURE.md`.
- To test a new synthesis method (e.g., algae-based reduction), create a JSON file under `data/experimental_synthesis_profiles/` describing the process, and link it with `monitoring/references.py`.
- For batch simulation comparisons across multiple membranes, you can script `main.py` with different config presets.
- Use `synthesis_recommender.py` to select the optimal synthesis method for your field or lab conditions:
  - Run: `python synthesis_recommender.py <temp_C> <sunlight_Wm2> <biomass_kg> [actual_CO actual_IDIG]`
  - The script will recommend the best method (electrochemical, solar, or biological) and, if you provide actual C/O and ID/IG ratios, will log and compare them to target specs for quality control.
- After updating force field or membrane properties, validate atomistic-level simulation stability using Phase 3 LAMMPS logs (`output/phase3/log.lammps`).
- If new contaminants, pathogens, microplastics, or NOM subtypes are tested, ensure their properties are added to the relevant section in `forcefield_and_simulation_data.json`.
- Use the `validation_and_field_data` section in the JSON to track and compare lab/field results.

---

## 6. Extensions & Future Work
- Consider creating `EXPERIMENTAL_SYNTHESIS_NOTES.md` as a structured companion to this guide, based on practical steps from your electrochemical lab manual and field protocols. This can help standardize synthesis and measurement procedures for all collaborators.
- For a technical breakdown of each code module, see `CODE_STRUCTURE.md`.
- For user-facing instructions and workflow overview, see `README.md`.
- For details on new/expanded JSON sections and placeholders, see the top of `forcefield_and_simulation_data.json`.

---
