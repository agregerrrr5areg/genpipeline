# FEMbyGEN + PyTorch Generative Design Pipeline

## Overview

This pipeline combines FEMbyGEN's topology optimization with a PyTorch-based generative model to:
1. Generate parametric design variations
2. Run FEM simulations on each variant
3. Train a VAE to learn the design space
4. Optimize new designs using gradient-based feedback from FEM results
5. Close the loop: generate → simulate → refine

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  FreeCAD + FEMbyGEN (Parametric Generation)        │
├─────────────────────────────────────────────────────┤
│  - Master parametric model                          │
│  - Parameter ranges (thickness, feature sizes, etc) │
│  - FEM setup (loads, materials, constraints)        │
└──────────────────┬──────────────────────────────────┘
                   │
                   v
┌─────────────────────────────────────────────────────┐
│  Data Extraction (fem_data_pipeline.py)             │
├─────────────────────────────────────────────────────┤
│  - Export FEM results from generated variations     │
│  - Mesh voxelization (voxel grids)                 │
│  - Stress/strain field extraction                   │
│  - Parameter-to-design mapping                      │
└──────────────────┬──────────────────────────────────┘
                   │
                   v
┌─────────────────────────────────────────────────────┐
│  PyTorch Dataset + DataLoader                       │
├─────────────────────────────────────────────────────┤
│  - Design voxel grids (input)                       │
│  - Performance metrics (output)                     │
│  - Parameterization (latent space)                  │
└──────────────────┬──────────────────────────────────┘
                   │
                   v
┌─────────────────────────────────────────────────────┐
│  Generative Model (vae_design_model.py)             │
├─────────────────────────────────────────────────────┤
│  - VAE Encoder: design geometry → latent vector     │
│  - VAE Decoder: latent vector → voxel design       │
│  - Performance predictor head (stress/compliance)  │
│  - CUDA-optimized training                          │
└──────────────────┬──────────────────────────────────┘
                   │
                   v
┌─────────────────────────────────────────────────────┐
│  Optimization Loop (optimization_engine.py)         │
├─────────────────────────────────────────────────────┤
│  - Sample latent space via Bayesian optimization    │
│  - Decode to design                                 │
│  - FEM simulation (via FreeCAD API)                 │
│  - Update VAE/predictor with new results           │
│  - Iterative refinement                             │
└──────────────────┬──────────────────────────────────┘
                   │
                   v
┌─────────────────────────────────────────────────────┐
│  Export Optimized Geometry (utils.py)               │
├─────────────────────────────────────────────────────┤
│  - Voxel grid → STEP/STL                            │
│  - Parameter reconstruction                        │
│  - Manufacturing constraints applied               │
└─────────────────────────────────────────────────────┘
```

## Workflow Steps

### Phase 1: Setup (FreeCAD GUI)
1. Create master parametric model (Part Design)
2. Define parameter ranges in spreadsheet
3. Set up FEM Analysis with loads/constraints
4. Switch to FEMbyGEN workbench
5. Click "Initialize" → Parameters spreadsheet created
6. Click "Generate" to create design variants
7. Click "FEA" → Run simulations on all variants

### Phase 2: Data Export
```bash
python fem_data_pipeline.py \
  --freecad-project master_design.FCStd \
  --output-dir ./fem_results \
  --voxel-resolution 32  # 32³ voxel grid
```

### Phase 3: Train Generative Model
```bash
python vae_design_model.py \
  --data-dir ./fem_results \
  --epochs 100 \
  --batch-size 32 \
  --latent-dim 16 \
  --gpu 0
```

### Phase 4: Run Optimization
```bash
python optimization_engine.py \
  --model-checkpoint vae_best.pth \
  --freecad-path /path/to/FreeCAD \
  --n-iterations 50 \
  --acquisition UCB  # Upper Confidence Bound
```

### Phase 5: Export Results
```bash
python export_geometry.py \
  --latent-vector "[0.5, -0.3, 0.2, ...]" \
  --output design_optimized.step
```

## Key Components

### 1. FEMbyGEN Setup
- Parameter naming: thickness_mm, feature_radius_mm, etc
- Range definition: min/max for each parameter
- FEM must be repeatable (same loads, constraints, materials)

### 2. Voxelization Strategy
- Convert mesh to 32³ or 64³ binary occupancy grid
- Option A: Occupancy (is point inside geometry?)
- Option B: Signed Distance Field (SDF) for smoothness
- Option C: Multi-channel (density + stress + strain)

### 3. VAE Architecture
- Encoder: 3D conv layers (32³ voxel input)
- Bottleneck: 16-dim latent space
- Decoder: 3D transposed conv
- Auxiliary head: predict max stress, compliance, mass

### 4. Optimization Objective
Multi-objective (Pareto frontier):
- Minimize: max_stress, compliance, mass
- Constraints: manufacturability, feature sizes
- Bounded: latent space [-3σ, 3σ]

## File Structure
```
project/
├── fem_data_pipeline.py      # Extract FEM results → datasets
├── vae_design_model.py        # VAE + performance predictor
├── optimization_engine.py     # Bayesian opt loop
├── freecad_interface.py       # FreeCAD API wrapper
├── utils.py                   # Voxelization, geometry utils
└── config.yaml                # Hyperparameters
```

## Data Flow Example

**Input (FEMbyGEN generates 100 designs):**
- Design 1: thickness=2.0mm, radius=5.0mm → FEM → stress=150MPa, compliance=0.3
- Design 2: thickness=2.5mm, radius=4.5mm → FEM → stress=120MPa, compliance=0.25
- ...
- Design 100: ...

**Processing:**
- Voxelize each design geometry (32³ grid)
- Normalize stress/compliance values
- Create PyTorch dataset with 100 samples

**Training:**
- VAE learns to encode geometry variations
- Predictor learns to estimate stress from voxels
- Bottleneck (16D latent) captures essential design features

**Optimization:**
- Sample latent space points (e.g., via Bayesian opt)
- Decode to voxel grid
- Convert back to parametric design
- Run FEM simulation
- Update predictor with real result
- Close loop

## Dependencies

```bash
pip install torch torchvision pytorch-cuda==12.1
pip install numpy scipy scikit-learn
pip install trimesh pyvista
pip install -U FreeCAD  # or use system FreeCAD
pip install botorch gpytorch  # Bayesian optimization
pip install tensorboard
```

## Performance Expectations

| Phase | Time | Hardware |
|-------|------|----------|
| FEMbyGEN generation | 5 min | CPU (FreeCAD) |
| FEM simulations (100) | 1-2 hrs | CPU (CalculiX) |
| Data prep | 5 min | CPU |
| VAE training | 30 min | RTX 5080 (FP32) |
| Optimization (50 iters) | 1-2 hrs | RTX 5080 + CalculiX |
| **Total** | **2-4 hrs** | **GPU + CPU parallelization** |

## Tips for Your Setup

1. **GPU Acceleration**: Use `torch.cuda.is_available()` to verify RTX 5080 detection
2. **Batch Processing**: Run multiple FEM sims in parallel via FreeCAD subprocesses
3. **Latent Space Size**: Start with 16D; increase if training loss plateaus
4. **Voxel Resolution**: 32³ = 32k params (fast); 64³ = 262k params (slow but better)
5. **Checkpoint Strategy**: Save VAE every epoch; use best-performing for inference

## Next Steps

1. Install FEMbyGEN via Addon Manager
2. Prepare parametric master model
3. Run `fem_data_pipeline.py` to extract data
4. Train VAE with `vae_design_model.py`
5. Iteratively run `optimization_engine.py` 
6. Export optimized designs

See individual Python files for detailed implementation.
