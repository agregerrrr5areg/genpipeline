# Sim Settings Panel + OpenLSTO Topology Solver Design
**Date**: 2026-02-28
**Status**: Approved

## Goal
Two parallel tracks:
1. **Sim Settings Panel** — configurable material, load, objective weights, structural constraints with safety factor; saved to `sim_config.json`; wired into FreeCAD FEM script and BO objective
2. **OpenLSTO Topology Solver** — level-set topology optimisation producing voxel density fields fed into the existing FEM/VAE/BO pipeline

---

## Track A: Sim Settings Panel

### UI
Collapsible `st.expander("Settings")` above the Design Browser in the left column.

**Sections:**
- **Material**: preset dropdown (Steel / Aluminium / Titanium / Custom) + manual fields: E (MPa), Poisson ratio ν, Density (kg/m³)
- **Load**: Force magnitude (N), direction fixed -Z on right face
- **Safety factor**: single float (e.g. 1.5 → allowable = yield_strength / SF)
- **Structural constraints**: max allowable stress (MPa), max displacement (mm)
- **Objective weights**: w_stress, w_compliance, w_mass (sliders 0–2)
- **[Save]** button → writes `sim_config.json`

**Material presets:**

| Preset | E (MPa) | ν | ρ (kg/m³) | Yield (MPa) |
|--------|---------|---|-----------|-------------|
| Steel | 210000 | 0.30 | 7900 | 250 |
| Aluminium | 70000 | 0.33 | 2700 | 270 |
| Titanium | 114000 | 0.34 | 4430 | 880 |

### Config file schema (`sim_config.json`)
```json
{
  "material": "Steel",
  "E_mpa": 210000,
  "poisson": 0.30,
  "density_kg_m3": 7900,
  "yield_mpa": 250,
  "force_n": 1000,
  "safety_factor": 1.5,
  "max_stress_mpa": 167,
  "max_disp_mm": 1.0,
  "w_stress": 1.0,
  "w_compliance": 0.1,
  "w_mass": 0.01
}
```
`max_stress_mpa` is auto-computed as `yield_mpa / safety_factor` but user can override.

### Pipeline wiring
- `freecad_scripts/run_fem_variant.py`: reads E, ν, ρ, force_n from config instead of hardcoded values
- `freecad_bridge.py`: passes config path to FreeCAD script
- `optimization_engine.py`: objective = `w_stress×stress + w_compliance×compliance + w_mass×mass`, plus penalty `1e6` if stress > max_stress or disp > max_disp
- `dashboard_bo_runner.py`: loads `sim_config.json` at start, passes weights/constraints to optimizer

---

## Track B: OpenLSTO Topology Solver

### Architecture
```
User sets: volume fraction, mesh resolution, loads, BCs (from sim_config.json)
    ↓
OpenLSTO C++ level-set solver  (wrapped via pybind11)
    ↓  density field φ (voxel grid, 32³)
Threshold φ > 0 → binary voxel → marching cubes → STL
    ↓
FreeCAD FEM validation  [existing]
    ↓  stress, compliance, mass → fem_variants/
VAE fine-tune on new data  [existing]
    ↓
BO searches latent space initialised from topo-opt result  [existing]
```

### Build plan
1. `sudo apt install cmake libeigen3-dev`
2. `pip install pybind11 scikit-image`  (scikit-image for marching cubes)
3. Clone OpenLSTO: `git clone https://github.com/topopt/OpenLSTO`
4. Write `topology_solver/bindings.cpp` — pybind11 wrapper exposing `solve_3d(params) → np.ndarray`
5. `CMakeLists.txt` to build shared lib `.so`
6. `topology_solver.py` — Python wrapper: `TopologySolver.run(sim_config) → voxel_grid`
7. Dashboard: "Run Topo Opt" button → runs solver → saves STL → adds to design browser

### Dashboard integration
New button in toolbar: **"Topo Opt"** — triggers a background thread running `TopologySolver.run(sim_config)`, shows progress, adds result to design browser when done.

---

## Roadmap (post this session)
- Train VAE on topology-opt results (50+ designs across varied load cases)
- U-Net surrogate replacing VAE for better spatial prediction
- Design clustering (k-means on latent space) for diverse alternatives panel
- OpenLSTO → manufacturing constraints (CNC, overhang)
- Multi-load-case optimisation
