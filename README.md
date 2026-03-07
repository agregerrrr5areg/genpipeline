# Generative Design Pipeline

A physics-grounded generative design system that combines topology optimisation (SIMP), a 3D convolutional VAE, and Bayesian optimisation to discover mechanically optimal geometries.

## Overview

The pipeline runs in four stages:

1. **Data generation** — SIMP topology optimisation produces physically valid voxel designs with FEM metrics (stress, compliance, mass).
2. **VAE training** — A 3D convolutional VAE learns a continuous latent space over the design distribution.
3. **Bayesian optimisation** — A GP surrogate with UCB acquisition searches the latent space for designs that minimise stress and mass.
4. **Export** — The best latent vector is decoded to a voxel grid and exported as STL/STEP.

## Repository Layout

```
genpipeline/          Python package
  fem/                FEM evaluation (GPU vectorised + CalculiX fallback)
  topology/           SIMP solvers (CPU, GPU, Dense GPU)
  cuda_kernels/       CUDA kernel sources and JIT loaders
  blackwell_compat.py RTX 50-series (sm_120) workarounds
  optimization_engine.py  BO loop (qUCB, scalarised objective)
  optimiser.py        Top-level orchestrator
  vae_design_model.py 3D conv VAE + performance predictor

data/                 Training dataset (fem_dataset.pt, 64^3 voxels)
checkpoints/          Saved model weights (vae_best.pth)
results/              BO output (bo_checkpoint.json, best_z.npy)
logs/                 TensorBoard event files

tests/                Test suite
scripts/              Utility scripts (rebuild_dataset, eval_vae, plot_pareto)
freecad_scripts/      Scripts that run inside FreeCAD Python
freecad_workbench/    FreeCAD workbench addon for FEMbyGEN export

pipeline_config.json  All hyperparameters and paths
quickstart.py         Single entry point for all pipeline stages
materials.yaml        Material property database
```

## Requirements

- Python 3.13, CUDA 12.8, PyTorch 2.10+cu128
- GPU: tested on RTX 5080 (sm_120 / Blackwell)

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy scikit-image scikit-learn trimesh botorch gpytorch tensorboard tqdm pyyaml
```

## Usage

```bash
source venv/bin/activate

# Generate SIMP training data
python quickstart.py --topo-data --n-samples 200

# Rebuild dataset.pt from raw STL+JSON files
python scripts/rebuild_dataset.py --fem-dir data --resolution 64

# Train VAE
python quickstart.py --step 3 --epochs 200

# Run Bayesian optimisation (50 rounds, GPU FEM)
python quickstart.py --step 4 --n-iter 50

# Monitor training
tensorboard --logdir ./logs
```

## Configuration

All settings are in `pipeline_config.json`. Key parameters:

| Key | Default | Notes |
|-----|---------|-------|
| `voxel_resolution` | 64 | 32 = fast, 64 = accurate (~8x more VRAM) |
| `latent_dim` | 32 | Dimensionality of design space |
| `batch_size` | 16 | Reduce if OOM on 64^3 |
| `optimization.parallel_evaluations` | 8 | FEM workers per BO round |

## Hardware Notes

**RTX 50 series (Blackwell, sm_120):**
- `cublasSgemmStridedBatched` is broken in CUDA 12.8 — BoTorch GP runs on CPU (`blackwell_compat.py`)
- `cublasSgemv` is also broken — all solvers use cuSPARSE SpMV (`torch.mv` on CSR)
- Conv3D and BF16 forward/backward work correctly

## Current Results

- Dataset: 461 samples at 64^3 resolution
- VAE: val IoU 0.881, 37.7M parameters
- BO: 50 rounds (400 evaluations) in ~8 minutes with GPU FEM
- Best design: stress proxy 0.015 MPa, mass 0.242 kg
