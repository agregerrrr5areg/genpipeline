# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A generative design pipeline combining FreeCAD's **FEMbyGEN** addon (parametric FEM simulation) with a **PyTorch 3D VAE** and **Bayesian optimization** to automatically discover mechanically optimal geometries.

## Environment

- **Python**: `venv/bin/python` (Python 3.13, venv at `/home/genpipeline/venv/`)
- **PyTorch**: `2.10.0+cu128` — installed and working on RTX 5080
- **GPU**: NVIDIA RTX 5080 (sm_120, Blackwell) — requires CUDA 12.8 builds

> **Important**: The RTX 5080 is Blackwell (sm_120). Do NOT install `torch` from the `cu121` index — use `cu128`:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
> ```


The venv is already configured. Activate with:
```bash
source venv/bin/activate
```

To recreate from scratch:
```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy scikit-image scikit-learn trimesh pyvista botorch gpytorch tensorboard tensorboardX matplotlib seaborn tqdm pyyaml
```

FreeCAD must be separately installed and its Python path exported:
```bash
export PYTHONPATH=/usr/lib/freecad/lib:$PYTHONPATH
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
python -c "import FreeCAD; print(FreeCAD.__version__)"
```

## Known Bug: Blackwell cuBLAS batched GEMM crash

**Affected**: RTX 5080 (sm_120), CUDA 12.8, torch 2.10.0+cu128, gpytorch 1.15.1, botorch 0.17.0

**Symptom**: Any `torch.matmul` or `@` operator on a CUDA tensor with `dim > 2` and `batch_size >= 2` raises:
```
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling
`cublasDgemmStridedBatched(...)` / `cublasSgemmStridedBatched(...)`
```

**Root cause**: `cublasDgemmStridedBatched` (double) and `cublasSgemmStridedBatched` (float) are broken for `batch >= 2` in the CUDA 12.8 build shipping with torch 2.10.0 on Blackwell. This is a driver/cuBLAS bug, not a gpytorch/botorch issue. Patching the Python libraries (contiguous, monkey-patching `torch.matmul`) does not help because the crash also originates from the `@` operator (`__matmul__`) inside gpytorch's lazy tensor and `linear_operator` machinery.

**Affected call sites found during testing**:
- `gpytorch/kernels/kernel.py:43` — `sq_dist`: `x1_.matmul(x2_.transpose(-2, -1))` with shape `[64, 1, 4] @ [64, 4, 11]`
- `gpytorch/models/exact_prediction_strategies.py:415` — `exact_predictive_covar`: `covar_inv_quad_form_root @ covar_inv_quad_form_root.transpose(-1, -2)`

**What still works on GPU**: Conv3D, Conv1D, standard 2D matmul (dim=2), batch=1 matmul.

**Workaround**: Keep BoTorch GP models on CPU. GPs in Bayesian optimisation are small (< 1000 points) and the pipeline bottleneck is FEM simulations, not the GP. `blackwell_compat.py` exports the correct device:

```python
from blackwell_compat import botorch_device  # torch.device('cpu') on this machine

train_X = train_X.to(botorch_device)
train_Y = train_Y.to(botorch_device)
gp = SingleTaskGP(train_X, train_Y)          # no .to('cuda') for BoTorch models
```

**Device map after workaround**:

| Component | Device |
|-----------|--------|
| VAE Conv3D training (`vae_design_model.py`) | `cuda` |
| GPyTorch custom GP training | `cuda` |
| BoTorch `SingleTaskGP` + `optimize_acqf` | `cpu` (via `botorch_device`) |

**Resolution**: Monitor pytorch/pytorch for a CUDA 12.8/Blackwell cuBLAS fix. When resolved, update `botorch_device` in `blackwell_compat.py` to `torch.device('cuda')`.

## Pipeline Commands

Run the full automated pipeline:
```bash
python quickstart.py --all --config pipeline_config.json
```

Or step-by-step:
```bash
python quickstart.py --step 2                        # Extract FEM data from FreeCAD files
python quickstart.py --step 3 --epochs 100           # Train VAE
python quickstart.py --step 4 --n-iter 50            # Bayesian optimization
python quickstart.py --step 5                        # Export optimized design

# Run individual modules directly
python fem_data_pipeline.py --freecad-project ./freecad_designs --output-dir ./fem_data --voxel-resolution 32
python vae_design_model.py --dataset-path ./fem_data/fem_dataset.pt --epochs 100 --latent-dim 16 --batch-size 8
python optimization_engine.py --model-checkpoint checkpoints/vae_best.pth --n-iterations 50
```

Monitor training:
```bash
tensorboard --logdir ./logs   # http://localhost:6006
```

## Architecture

### End-to-End Data Flow

```
FreeCAD (FEMbyGEN) → parametric variants → CalculiX FEM sims
    → fem_data_pipeline.py  →  32³ voxel grids + FEM metrics  →  fem_dataset.pt
    → vae_design_model.py   →  trained 3D VAE  (checkpoints/vae_best.pth)
    → optimization_engine.py → Bayesian opt loop (GP surrogate + UCB acquisition)
    → decode best z → voxel → mesh → STL/STEP export
```

### Module Roles

- **`fem_data_pipeline.py`** — reads `.FCStd` files, extracts FEM results (stress, compliance), exports meshes as STL, voxelizes to 32³ binary grids, creates 80/20 train/val PyTorch Dataset saved to `fem_data/`.
- **`vae_design_model.py`** — 3D convolutional VAE: `Encoder (3×Conv3D→FC) → latent μ/σ (16-dim) → Decoder (FC→3×ConvTranspose3D)` with a performance predictor head outputting stress/compliance/mass. Includes KL annealing and checkpoint management.
- **`optimization_engine.py`** — wraps BoTorch GP + UCB acquisition. Each iteration: sample latent `z` → decode → run FEM in FreeCAD → update GP. Objective: `stress + 0.1×compliance + 0.01×mass`. History saved to `optimization_results/`.
- **`utils.py`** — geometry voxelization helpers, manufacturability constraint checking (min feature size, max overhang angle), geometry metrics (volume, surface area, MOI), FreeCAD integration helpers.
- **`quickstart.py`** — orchestrates all 5 steps; loads config from `pipeline_config.json`; `PipelineConfig` class merges defaults with file overrides.

### Configuration

All hyperparameters live in `pipeline_config.json`. Key values:

| Key | Default | Effect |
|-----|---------|--------|
| `voxel_resolution` | 32 | 32³ = fast; 64³ = accurate but ~8× more VRAM |
| `latent_dim` | 16 | Dimensionality of design space |
| `beta_vae` | 1.0 | KL weight — lower = more diverse samples |
| `n_optimization_iterations` | 50 | More = better optimum but slower |
| `optimization.acquisition_function` | `"UCB"` | Also supports `"EI"` |

### Generated Data Directories (not in repo)

```
freecad_designs/        # Input .FCStd files from FEMbyGEN
fem_data/               # Voxel grids, FEM metrics, fem_dataset.pt
checkpoints/            # vae_best.pth + epoch snapshots
logs/                   # TensorBoard events
optimization_results/   # optimization_history.json, exported STL/STEP
```
