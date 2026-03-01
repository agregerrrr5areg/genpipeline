# Generative Design Pipeline — System Design Document

**Date:** 2026-03-01
**Status:** Implemented (training + generation in progress)

---

## 1. Problem Statement

Manually iterating on FEM-validated mechanical designs is slow. The goal is a closed-loop system where:
1. A parametric FEM simulator samples the design space
2. A generative model learns the underlying geometry distribution
3. Bayesian optimisation navigates the latent space towards Pareto-optimal designs (minimise stress + mass)
4. A FreeCAD workbench lets engineers define boundary conditions and launch the loop from the CAD tool they already use

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  FreeCAD (Windows)                                                  │
│  GenDesign Workbench                                                │
│   Add Constraint → Add Load → Set Seed Part → Export Config        │
│                         │                                           │
│                  gendesign_config.json                              │
│                    C:\Windows\Temp\                                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  wsl bash -c "python optimization_engine.py"
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  WSL2 / Linux — /home/genpipeline/                                  │
│                                                                     │
│  FEM Data Generation                                                │
│   freecad_bridge.py ──► run_fem_variant.py (Windows FreeCAD)        │
│   4 geometries: cantilever / lbracket / tapered / ribbed            │
│   Output: *_mesh.stl + *_fem_results.json → fem_data/              │
│                                                                     │
│  Dataset                                                            │
│   rebuild_dataset.py ──► FEMDataset (64³ voxels, uint8)            │
│   237–450 samples, 80/20 train/val split                            │
│                                                                     │
│  VAE (vae_design_model.py)                                          │
│   Encoder: Conv3D 1→32→64→128→256, stride-2 ×4  (64³ → 4³)        │
│   FC:      Linear 16384→1024→32 (latent)                            │
│   Decoder: FC 32→1024→16384, ConvTranspose3D 256→128→64→32→16→1   │
│   Heads:   performance (→3) + parameter (→2)                        │
│   37.7M params, BF16 conv + FP32 linear (Blackwell workaround)     │
│                                                                     │
│  Bayesian Optimisation (optimization_engine.py)                     │
│   MOBO: qEHVI on [stress, mass] Pareto front                        │
│   GP on CPU (Blackwell cuBLAS batched GEMM bug)                     │
│   Latent space: z ∈ ℝ³² → parameter_head → (h_mm, r_mm)           │
│   Per-geometry bounds via GEOM_SPACES dict                          │
│                                                                     │
│  FEM Evaluation (freecad_bridge.py + run_fem_variant.py)            │
│   4 workers × parallel variants, UUID stems to avoid collisions     │
│   Returns: stress_max, compliance, mass, bbox                       │
│                                                                     │
│  Voxel FEM (voxel_fem.py) — direct CalculiX path                   │
│   Decode z → 64³ voxels → C3D8 hex mesh → .inp → ccx               │
│   Bypasses FreeCAD; geometry-agnostic; --voxel-fem flag             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Components

### 3.1 FEM Data Generation (`freecad_bridge.py` + `freecad_scripts/run_fem_variant.py`)

Generates parametric FEM variants by calling Windows FreeCAD from WSL2 via subprocess.

**Per-geometry shapes and boundary conditions:**

| Geometry | Shape function | Fixed BC normal | Load BC normal |
|----------|----------------|-----------------|----------------|
| cantilever | `make_shape(h, r)` | `(-1,0,0)` left face | `(1,0,0)` right face |
| lbracket | `make_lbracket_shape(arm_h, thickness)` | `(0,0,-1)` bottom | `(1,0,0)` arm tip |
| tapered | `make_tapered_beam_shape(h_start)` | `(-1,0,0)` left | `(1,0,0)` right |
| ribbed | `make_ribbed_plate_shape(rib_h, plate_frac)` | `(-1,0,0)` left | `(1,0,0)` right |

**Key implementation details:**
- UUID-based stems prevent filename collisions in parallel runs
- Config JSON (including `stem`) written to Windows Temp and read by FreeCAD
- `find_face()` used for BC assignment; lbracket uses `(0,0,-1)` normal

### 3.2 Dataset (`fem_data_pipeline.py`, `rebuild_dataset.py`)

- `VoxelGrid` voxelizes STL meshes at 64³ using a CUDA kernel (~2890× faster than CPU trimesh)
- Stored as uint8 per sample, converted to float32 on `__getitem__`
- `rebuild_dataset.py` scans `fem_data/` and rebuilds `fem_dataset.pt` from scratch

**Current dataset state (2026-03-01):**
- cant: 196, lbra: 82, tape: ~70, ribb: ~70 (generation in progress)
- Target: ~450 samples with balanced geometry coverage

### 3.3 VAE (`vae_design_model.py`)

37.7M parameter 3D VAE trained at 64³ resolution.

**Blackwell RTX 5080 constraints:**
- `cublasDgemmStridedBatched` (batched 3D matmul) crashes at batch ≥ 2 — BoTorch GP stays on CPU
- `cublasGemmEx` with BF16 crashes during backward for `nn.Linear` — wrapped in `torch.autocast(enabled=False)`
- Conv3D forward/backward in BF16 works correctly

**Training:** 500 epochs, batch 128, OneCycleLR (3e-4 peak), KL ramp over 50 epochs, val loss 0.0084 (final checkpoint).

### 3.4 Bayesian Optimisation (`optimization_engine.py`)

Multi-objective BO using `qExpectedHypervolumeImprovement` over [stress, mass].

**Per-geometry BO bounds (GEOM_SPACES):**

| Geometry | h_mm range | r_mm range |
|----------|-----------|-----------|
| cantilever | [5, 20] | [0, 5] |
| tapered | [8, 25] | [2, 7] |
| ribbed | [6, 20] | [2, 6] |
| lbracket | [8, 25] | [5, 20] |

`--config-path` flag accepts `gendesign_config.json` from the FreeCAD workbench to override geometry type, BC normals, stress limit, and iteration count.

### 3.5 Voxel FEM (`voxel_fem.py`)

Direct CalculiX path that bypasses FreeCAD entirely.

- Binary voxels → C3D8 hex elements (node deduplication via dict)
- BCs applied by face index (x_min fixed, x_max loaded by default)
- `.frd` parser uses fixed-width column slicing (12 chars at offset 13+slot×12)
- Integrated via `--voxel-fem` flag in `optimization_engine.py`
- Verified: 10³ solid cube → stress_max=61.1 MPa ✓

### 3.6 FreeCAD Workbench (`freecad_workbench/`)

7-file FreeCAD workbench deployed to `Mod/GenDesign/` via `deploy.sh`.

**Commands:**
1. Add Constraint — FeaturePython ConstraintObject (fixed/symmetry/preserve/mounting)
2. Add Load — FeaturePython LoadObject (force/pressure/acceleration)
3. Set Seed Part — geometry type, volume fraction, stress limit, WSL2 paths, n_iter
4. Export Config — writes `gendesign_config.json` with BC normals derived from geometry type
5. Run Optimisation — shells to WSL2, streams stdout to progress dialog with Cancel
6. Import Result — file dialog → imports best_design.stl as Mesh::Feature

---

## 4. Data Flow

```
User defines BCs in FreeCAD
        │
        ▼  Export Config
gendesign_config.json
        │
        ▼  Run Optimisation (WSL2 subprocess)
optimization_engine.py --config-path gendesign_config.json
        │
        ├── loads VAE checkpoint
        ├── seeds latent space (random z ~ N(0,1))
        │
        └── for each BO round:
              z → parameter_head → (h_mm, r_mm, geometry)
              → BridgeEvaluator.evaluate_batch() [4 workers]
                  → freecad_bridge.run_variant() × 4
                      → FreeCAD Windows: build shape + FEM
                      → returns {stress_max, compliance, mass, bbox}
              → update GP surrogate
              → qEHVI acquisition → next z candidates
              → log Pareto front size
        │
        ▼
optimization_results/
  ├── optimization_history.json  (Pareto front + all history)
  └── fem_evaluations.json
        │
        ▼  (optional) step_5_export_design
        └── decode Pareto z → voxels → STL
```

---

## 5. What Remains

### 5.1 End-to-End Validation
The pipeline has been built and tested in parts. A full end-to-end run has not been executed:
- Retrain on balanced ~450-sample dataset is queued (auto-starts after generation)
- Run `python quickstart.py --step 4 --n-iter 50` with the production checkpoint
- Verify `optimization_history.json` contains a non-trivial Pareto front

### 5.2 Post-training Evaluation
`eval_vae.py` is ready but has not been run on the production checkpoint:
- Reconstruction IoU on validation set
- Latent space PCA plot
- Random sample STL export for visual inspection

### 5.3 Workbench End-to-End Test
The workbench has been deployed but not tested end-to-end in a live FreeCAD session:
- Open a part in FreeCAD
- Add constraints/loads via the GenDesign panels
- Export config, verify JSON correctness
- Click Run Optimisation, observe progress stream
- Import result STL

### 5.4 Voxel FEM Integration Test
`voxel_fem.py` was unit-tested on a solid cube but not integrated into a full BO loop:
- Run `python optimization_engine.py --voxel-fem --model-checkpoint checkpoints/vae_best.pth --n-iter 20`
- Verify stress/mass values are plausible
- Compare against bridge-evaluated results on the same geometry

---

## 6. Configuration Reference

`pipeline_config.json` (key fields):

| Key | Value | Effect |
|-----|-------|--------|
| `voxel_resolution` | 64 | 64³ voxels |
| `latent_dim` | 32 | Latent space dimensionality |
| `batch_size` | 128 | Training batch size |
| `epochs` | 500 | Training epochs |
| `learning_rate` | 3e-4 | OneCycleLR peak LR |
| `beta_vae` | 0.05 | KL weight |
| `n_optimization_iterations` | 1000 | BO rounds |

`gendesign_config.json` (workbench export):

| Key | Effect |
|-----|--------|
| `geometry_type` | Selects GEOM_SPACES bounds + BC normals |
| `n_iter` | Overrides `--n-iter` |
| `checkpoint_path` | Overrides `--model-checkpoint` |
| `fixed_face_normal` | BC override passed to run_fem_variant.py |
| `max_stress_mpa` | Stress constraint for penalty term |

---

## 7. Known Issues / Constraints

| Issue | Workaround |
|-------|-----------|
| Blackwell batched GEMM crash | BoTorch GP on CPU via `botorch_device` |
| Blackwell BF16 Linear backward crash | `torch.autocast(enabled=False)` around all `nn.Linear` |
| `generate_variants` was rebuilding dataset at 32³ | Fixed: default `voxel_resolution=64` |
| lbracket FEM BCs were wrong (x-face instead of z-face) | Fixed: `find_face(normal=(0,0,-1))` |
| `.frd` parser returned zeros | Fixed: 12-char fixed-width column slicing |
| Aider concurrent refactors | Worked with new architecture (config-driven BCs, MOBO) |
