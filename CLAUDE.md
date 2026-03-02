# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A generative design pipeline combining FreeCAD's **FEMbyGEN** addon (parametric FEM simulation) with a **PyTorch 3D VAE** and **Bayesian optimisation** to automatically discover mechanically optimal geometries.

## Environment

- **Python**: `venv/bin/python` (Python 3.13, venv at `/home/genpipeline/venv/`)
- **PyTorch**: `2.10.0+cu128` — installed and working on RTX 50 series card
- **GPU**: NVIDIA RTX 50 series card (sm_120, Blackwell) — requires CUDA 12.8 builds

> **Important**: The RTX 50 series card is Blackwell (sm_120). Do NOT install `torch` from the `cu121` index — use `cu128`:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
> ```


## CUDA Kernel Compilation

Custom CUDA kernels live in `cuda_kernels/`. They require CUDA 12.8 nvcc (not the system 11.8).

```bash
# Compile and benchmark GPU voxelisation kernel
source venv/bin/activate
CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH python cuda_kernels/benchmark.py
```

The kernel JIT-compiles on first use and is cached automatically. Results: **27–2890× faster** than CPU trimesh voxelisation.

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

**Affected**: RTX 50 series card (sm_120), CUDA 12.8, torch 2.10.0+cu128, gpytorch 1.15.1, botorch 0.17.0

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
| BoTorch `SingleTaskGP` + `optimise_acqf` | `cpu` (via `botorch_device`) |

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
python quickstart.py --step 4 --n-iter 50            # Bayesian optimisation
python quickstart.py --step 5                        # Export optimised design

# Run individual modules directly
python fem/data_pipeline.py --freecad-project ./freecad_designs --output-dir ./fem/data --voxel-resolution 32
python vae_design_model.py --dataset-path ./fem/data/fem_dataset.pt --epochs 100 --latent-dim 16 --batch-size 8
python optimisation_engine.py --model-checkpoint checkpoints/vae_best.pth --n-iterations 50
```

Monitor training:
```bash
tensorboard --logdir ./logs   # http://localhost:6006
```

## Architecture

### End-to-End Data Flow

```
FreeCAD (FEMbyGEN) → parametric variants → CalculiX FEM sims
    → fem/data_pipeline.py  →  64³ voxel grids + FEM metrics  →  fem/data/fem_dataset.pt
    → genpipeline/vae_design_model.py   →  trained 3D VAE  (checkpoints/vae_best.pth)
    → genpipeline/optimization_engine.py → Bayesian opt loop (GP surrogate + UCB acquisition)
    → decode best z → voxel → mesh → STL/STEP export
```

### Module Roles

- **`fem/data_pipeline.py`** — reads `.FCStd` files, extracts FEM results (stress, compliance), exports meshes as STL, voxelises to 64³ binary grids, creates 80/20 train/val PyTorch Dataset saved to `fem/data/`.
- **`genpipeline/vae_design_model.py`** — 3D convolutional VAE with a performance predictor head outputting stress/compliance/mass.
- **`genpipeline/optimization_engine.py`** — wraps BoTorch GP + UCB acquisition. MOBO loop logic.
- **`genpipeline/pipeline_utils.py`** — Shared utilities (voxelization, manufacturability, encoding).
- **`genpipeline/schema.py`** — Pydantic models for type-safe data passing and validation.
- **`genpipeline/config.py`** — Configuration loading and validation using Pydantic.
- **`fem/voxel_fem.py`** — Direct CalculiX voxel FEM path, bypassing FreeCAD.
- **`topology/topo_data_gen.py`** — SIMP-based physics-grounded dataset generator.
- **`quickstart.py`** — Integrated CLI orchestrator for all pipeline steps.

### Testing & Verification

Comprehensive test suite in `tests/`:
- `test_schema_validation.py`: Pydantic model and validator checks.
- `test_vae_model.py`: GPU integration and architecture tests.
- `test_optimization_engine.py`: MOBO loop and fallback logic.
- `test_voxel_fem.py`: CalculiX integration and meshing tests.
- `test_provenance.py`: Data integrity and 'No Synthetic Data' mandate guards.
- `test_rebuild_dataset.py`: STL scanner and dataset construction tests.
- `test_export_pipeline.py`: FreeCAD workbench export logic mocks.

### Quickstart Bootstrap
```bash
python quickstart.py --topo-data --n-samples 200  # Generate data and rebuild dataset.pt
python quickstart.py --step 3 --epochs 100        # Train VAE
python quickstart.py --step 4 --n-iter 50 --topo-refine  # Optimise with refined candidates
```

### MANDATE: No Non-Physical Synthetic Data
**Strict Rule:** All training data must originate from physics-based simulations (FreeCAD/CalculiX or SIMP). **Non-physical 'synthetic' data** (e.g., random noise-based shapes, pure geometric heuristics without FEM validation) **is forbidden.** The model must exclusively learn from physically plausible mechanics to ensure engineering validity.

### Configuration

All hyperparameters live in `pipeline_config.json`. Key values:

| Key | Default | Effect |
|-----|---------|--------|
| `voxel_resolution` | 32 | 32³ = fast; 64³ = accurate but ~8× more VRAM |
| `latent_dim` | 16 | Dimensionality of design space |
| `beta_vae` | 1.0 | KL weight — lower = more diverse samples |
| `n_optimisation_iterations` | 50 | More = better optimum but slower |
| `optimisation.acquisition_function` | `"UCB"` | Also supports `"EI"` |

### Generated Data Directories (not in repo)

```
freecad_designs/        # Input .FCStd files from FEMbyGEN
fem/                    # Unified physics folder
  data/                 # Voxel grids, FEM metrics, fem_dataset.pt
checkpoints/            # vae_best.pth + epoch snapshots
logs/                   # TensorBoard events
optimisation_results/   # optimisation_history.json, exported STL/STEP
```
