# GenDesign FreeCAD Workbench — Design Document

**Date:** 2026-03-01
**Status:** Implemented

---

## Overview

GenDesign is a FreeCAD workbench that connects FreeCAD's geometry authoring environment to the WSL2-resident VAE + Bayesian optimisation pipeline. Users define boundary conditions and optimisation parameters directly on their FreeCAD part, then launch the pipeline with a single button click.

---

## Architecture

```
FreeCAD (Windows)                         WSL2 (Linux)
─────────────────────────────             ─────────────────────────────
GenDesign Workbench                       /home/genpipeline/
  ├── constraint_obj.py   ─┐              ├── optimization_engine.py
  ├── load_obj.py          │ document     ├── vae_design_model.py
  ├── seed_part.py         │ objects      ├── freecad_bridge.py
  ├── export_pipeline.py ──┘              └── voxel_fem.py
  └── commands.py
        │
        │  gendesign_config.json
        │  C:\Windows\Temp\gendesign_config.json
        │  /mnt/c/Windows/Temp/gendesign_config.json
        ▼
  wsl bash -c "python optimization_engine.py --config-path ..."
        │
        ▼
  stdout streamed to progress dialog
```

---

## Document Objects

Each object is a `App::FeaturePython` instance with a typed `Proxy` class and a corresponding `ViewProvider`.

### ConstraintObject (`constraint_obj.py`)

Properties stored on the document object:

| Property | Type | Description |
|----------|------|-------------|
| `ConstraintType` | String | `fixed` / `symmetry` / `preserve` / `mounting` |
| `Label` | String | Human-readable name |
| `References` | LinkSubList | (face, sub-element) pairs |

Task panel (`AddConstraintPanel`) lets users pick constraint type and select faces interactively.

### LoadObject (`load_obj.py`)

| Property | Type | Description |
|----------|------|-------------|
| `LoadType` | String | `force` / `pressure` / `acceleration` |
| `Magnitude` | Float | Force in N, pressure in MPa, or acceleration in mm/s² |
| `Direction` | Vector | Unit vector (dx, dy, dz) |
| `References` | LinkSubList | Loaded faces |

### SeedPartObject (`seed_part.py`)

| Property | Type | Description |
|----------|------|-------------|
| `SourceBody` | Link | Body to optimise |
| `GeometryType` | String | `cantilever` / `lbracket` / `tapered` / `ribbed` |
| `VolumeFraction` | Float | Target solid fraction (0–1) |
| `MaxStressMPa` | Float | Stress constraint (MPa) |
| `NoOverhang` | Bool | Enforce 45° overhang rule |
| `WSL2PipelinePath` | String | e.g. `/home/genpipeline` |
| `CheckpointPath` | String | VAE `.pth` path in WSL2 |
| `NIter` | Integer | Optimisation iterations |

---

## Config Export (`export_pipeline.py`)

`export_config(doc, output_path)` walks all document objects, collects constraints and loads, reads the seed part properties, and writes `gendesign_config.json`:

```json
{
  "geometry_type": "lbracket",
  "checkpoint_path": "/home/genpipeline/checkpoints/vae_best.pth",
  "n_iter": 50,
  "volume_fraction": 0.4,
  "max_stress_mpa": 250.0,
  "no_overhang": false,
  "wsl2_pipeline_path": "/home/genpipeline",
  "constraints": [{"type": "fixed", "label": "...", "faces": ["Body.Face1"]}],
  "loads": [{"type": "force", "magnitude": 1000.0, "direction": [0,0,-1], "label": "...", "faces": ["Body.Face5"]}],
  "fixed_face_normal": [0, 0, -1],
  "load_face_normal":  [1, 0, 0],
  "force_n": 1000.0,
  "force_direction": [0, 0, -1]
}
```

Boundary condition normals (`fixed_face_normal`, `load_face_normal`) are derived from geometry type via a lookup table:

| Geometry | fixed_face_normal | load_face_normal |
|----------|-------------------|-----------------|
| cantilever | `[-1, 0, 0]` | `[1, 0, 0]` |
| tapered | `[-1, 0, 0]` | `[1, 0, 0]` |
| ribbed | `[-1, 0, 0]` | `[1, 0, 0]` |
| lbracket | `[0, 0, -1]` | `[1, 0, 0]` |

---

## Commands (`commands.py`)

| Command ID | Menu Text | Action |
|------------|-----------|--------|
| `GenDesign_AddConstraint` | Add Constraint | Opens `AddConstraintPanel` task panel |
| `GenDesign_AddLoad` | Add Load | Opens `AddLoadPanel` task panel |
| `GenDesign_SetSeedPart` | Set Seed Part | Opens `SetSeedPartPanel` task panel |
| `GenDesign_ExportConfig` | Export Config | Calls `export_config()`, shows summary dialog |
| `GenDesign_RunOptimisation` | Run Optimisation | Exports config, shells to WSL2, streams output |
| `GenDesign_ImportResult` | Import Result | File dialog → imports STL as Mesh::Feature |

### Run Optimisation Flow

1. Export `gendesign_config.json` to `C:\Windows\Temp\`
2. Build WSL2 command:
   `cd {pipeline_path} && source venv/bin/activate && python optimization_engine.py --model-checkpoint {ckpt} --n-iterations {n} --config-path {wsl_config}`
3. Launch `wsl bash -c "..."` via `subprocess.Popen`
4. Stream stdout line-by-line to a `QPlainTextEdit` progress dialog
5. Cancel button sends `proc.terminate()`
6. On exit code 0: success dialog; otherwise warning with exit code

---

## Deployment

```bash
./freecad_workbench/deploy.sh
# Optional: specify FreeCAD location
./freecad_workbench/deploy.sh "/mnt/c/Program Files/FreeCAD 1.0"
```

Copies 7 Python files to `{FreeCAD}/Mod/GenDesign/`. FreeCAD auto-discovers workbenches in `Mod/`.

Default install path: `C:\Users\PC-PC\AppData\Local\Programs\FreeCAD 1.0\Mod\GenDesign`

---

## Integration with Pipeline

`optimization_engine.py` accepts `--config-path <json>`. When provided:
- `geometry_type` overrides `sim_cfg["geometry_type"]`
- `n_iter` overrides `--n-iter`
- `checkpoint_path` overrides `--model-checkpoint`
- `fixed_face_normal`, `load_face_normal`, `force_n`, `force_direction` flow into `sim_cfg` and are forwarded to `freecad_bridge.run_variant()` as BC overrides

This allows the workbench to fully drive the pipeline without requiring CLI argument changes.
