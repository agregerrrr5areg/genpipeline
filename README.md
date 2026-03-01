# GenPipeline

Generative design pipeline combining FreeCAD FEM simulation, a 3D VAE, and Bayesian optimisation to discover structurally optimal geometries. Tuned for RTX 5080 (Blackwell/CUDA 12.8).

---

## Status

| Component | State | Notes |
|-----------|-------|-------|
| FEM data generation | Active | Cantilever, L-bracket, tapered, ribbed variants via FreeCAD bridge |
| VAE training | Active | 3D conv VAE, 16-dim latent space, performance + parameter heads |
| Bayesian optimisation | Active | Two-stage: GPU surrogate sweep then physical-space BO per geometry |
| Voxel FEM path | Active | Direct CalculiX C3D8 hex mesh from decoded voxels (`--voxel-fem`) |
| Blackwell workaround | Active | BoTorch GP runs on CPU; VAE/Conv on GPU |

---

## Architecture

```
freecad_bridge.py      — parametric FEM variants via FreeCAD (WSL2 -> Windows)
fem_data_pipeline.py   — voxelise STL + FEM metrics -> fem_dataset.pt
vae_design_model.py    — 3D conv VAE: encode/decode + stress/param predictor heads
optimization_engine.py — two-stage BO: latent sweep (Stage 1) + physical BO (Stage 2)
voxel_fem.py           — direct CalculiX path: voxels -> C3D8 hex mesh -> .frd parser
utils.py               — voxel/mesh helpers, geometry metrics
quickstart.py          — orchestrates all pipeline steps
blackwell_compat.py    — device routing (BoTorch on CPU, VAE on GPU)
```

### Optimisation stages

- **Stage 1** — GPU surrogate sweep: sample 100 latent vectors, score via VAE predictor, run real FEM on top candidates.
- **Stage 2a** (default) — Physical BO in (h, r) space per geometry via FreeCAD bridge.
- **Stage 2b** (`--voxel-fem`) — Latent-space UCB/GP BO with direct CalculiX hex mesh; evaluates the full 32³ voxel topology, not just (h, r).

### Geometry types and boundary conditions

| Geometry | Fixed face | Load face | BO bounds |
|----------|-----------|-----------|-----------|
| `cantilever` | x-min | x-max | h ∈ [5,20] mm, r ∈ [0,5] mm |
| `lbracket` | z-min (base of vertical arm) | x-max (tip of horizontal arm) | arm_h ∈ [8,25] mm, thickness ∈ [5,20] mm |
| `tapered` | x-min | x-max | h_start ∈ [8,25] mm, taper ∈ [2,7] |
| `ribbed` | x-min | x-max | rib_h ∈ [6,20] mm, plate_frac ∈ [2,6] |

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

FreeCAD 1.0 must be installed on Windows (not in WSL). The bridge auto-detects it under `/mnt/c/`.

---

## Usage

```bash
# Full pipeline
python quickstart.py --all --config pipeline_config.json

# Individual steps
python quickstart.py --step 2                          # extract FEM data
python quickstart.py --step 3 --epochs 200             # train VAE
python quickstart.py --step 4 --n-iter 50              # Bayesian optimisation
python quickstart.py --step 5                          # export best design

# Direct optimisation
python optimization_engine.py --model-checkpoint checkpoints/vae_best.pth --n-iterations 100
python optimization_engine.py --model-checkpoint checkpoints/vae_best.pth --voxel-fem

# Voxel FEM unit test (10x10x10 cube, verifies ccx + .frd parser)
python voxel_fem.py --test

# Synthetic end-to-end test (no FreeCAD required)
python synthetic_test.py
```

Set `"geometry_type"` in `sim_config.json` to switch geometry (`cantilever` | `lbracket` | `tapered` | `ribbed`).

---

## Configuration

| File | Purpose |
|------|---------|
| `pipeline_config.json` | VAE dims, learning rate, voxel resolution, BO acquisition |
| `sim_config.json` | Geometry type, FEM weights (stress/compliance/mass), yield limit |
| `materials.json` | Material properties (E, Poisson, density, thermal) |

---

## Hardware notes (RTX 5080 / Blackwell)

Install torch from the `cu128` index — the `cu121` build does not include sm_120 kernels.

The RTX 5080 triggers a cuBLAS batched GEMM crash for batch >= 2 in CUDA 12.8. `blackwell_compat.py` routes BoTorch GP models to CPU automatically. See `CLAUDE.md` for full details.

---

## License

Research and industrial optimisation use. Refer to [FreeCAD](https://www.freecad.org/) and [BoTorch](https://botorch.org/) for dependency licenses.
