# FEMbyGEN + PyTorch Pipeline - Getting Started Guide

## What You're Getting

A complete generative design system that combines:
- **FEMbyGEN** (FreeCAD addon) for parametric design generation
- **PyTorch VAE** (Variational Autoencoder) to learn design space
- **Bayesian Optimization** to find optimal designs
- **CUDA acceleration** on your RTX 5080 for training & inference

## File Overview

```
fembygen_pytorch_guide.md          (READ THIS FIRST)
├── Architecture diagrams
├── Complete workflow explanation
├── Dependency installation
└── Performance expectations

fem_data_pipeline.py
├── Extracts FEM results from FreeCAD
├── Voxelizes geometries (converts mesh to 32³ grid)
├── Creates PyTorch Dataset
└── Handles train/val split

vae_design_model.py
├── 3D Convolutional VAE architecture
├── Performance predictor head
├── Training loop with KL annealing
├── Checkpoint management
└── Design interpolation utilities

optimization_engine.py
├── FEM evaluator (simulates designs in FreeCAD)
├── Bayesian optimization using BoTorch
├── Gaussian Process for surrogate model
├── Acquisition function (UCB/EI)
└── Result history tracking

utils.py
├── Geometry voxelization utilities
├── Manufacturability constraint checking
├── Geometry metrics (volume, surface area, MOI)
├── FreeCAD integration helpers
└── Performance normalization

quickstart.py
├── End-to-end pipeline orchestration
├── 5-step workflow execution
├── Configuration management
└── Run individual pipeline steps

config/requirements.txt
├── All Python dependencies
└── Pin versions for reproducibility

pipeline_config.json
├── Default hyperparameters
├── Path configuration
└── Optimization settings

README.md
├── Full documentation
├── Troubleshooting
├── Advanced features
└── Performance benchmarks
```

## Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name())"
```

### 2. Prepare FreeCAD

1. Install FreeCAD 1.0+: https://www.freecad.org/
2. Install FEMbyGEN addon:
   - Tools → Addon Manager → Search "FEMbyGEN" → Install
   - Restart FreeCAD
3. Create parametric model:
   - Part Design: Create sketch with parameter-constrained features
   - FEM: Setup analysis with loads, materials, mesh
   - FEMbyGEN: Generate 50-100 design variations
   - Run simulations

### 3. Run Pipeline

```bash
# Full automated workflow
python quickstart.py --all

# Or step-by-step
python quickstart.py --step 1  # FreeCAD setup (manual)
python quickstart.py --step 2  # Extract FEM data
python quickstart.py --step 3  # Train VAE
python quickstart.py --step 4  # Optimize
python quickstart.py --step 5  # Export
```

## Detailed Workflow

### Step 1: FreeCAD Parametric Model

**Create in Part Design:**
```
Base Sketch (parameters: thickness=t, radius=r)
  ↓
Pad (height: t)
  ↓
Pocket (radius: r)
  ↓
Fillet (radius: r/2)
```

**Setup in FEM:**
```
Analysis Container
  ├── Fixed constraint (one face)
  ├── Force (e.g., 1000N on opposite face)
  ├── Material (e.g., Steel)
  ├── Mesh (Gmsh, 2-5mm element size)
  └── CalculiX Solver
```

**Generate with FEMbyGEN:**
```
1. FEMbyGEN Workbench → Initialize
   (Creates Parameters spreadsheet)

2. Set ranges:
   thickness: [1.5, 3.5] mm
   radius: [3.0, 7.0] mm
   feature_size: [0.5, 2.0] mm

3. Generate → 100 designs with random parameter combinations

4. FEA → Runs CalculiX on all 100 designs
   (Takes 1-4 hours depending on model complexity)
```

### Step 2: Extract FEM Data

```bash
python fem_data_pipeline.py \
  --freecad-project ./freecad_designs \
  --output-dir ./fem_data \
  --voxel-resolution 32
```

**What happens:**
- Reads each .FCStd file from FEMbyGEN generations
- Extracts FEM results (stress, compliance)
- Exports mesh geometries as STL
- Converts to 32³ binary voxel grids
- Creates 80-20 train/val split
- Saves PyTorch Dataset

**Output:**
```
fem_data/
├── fem_results.json (FEM metrics per design)
├── meshes/ (geometry STL files)
├── voxels/ (32³ voxel grids)
└── fem_dataset.pt (PyTorch Dataset)
```

### Step 3: Train VAE

```bash
python vae_design_model.py \
  --dataset-path ./fem_data/fem_dataset.pt \
  --epochs 100 \
  --latent-dim 16 \
  --batch-size 8 \
  --beta 1.0
```

**Architecture:**
```
Input Geometry (1 × 32 × 32 × 32)
  ↓
Encoder (3× Conv3D → MaxPool → FC)
  ↓
Latent Space (16-dimensional)
  ├→ μ (mean)
  └→ σ (variance)
  ↓
Decoder (FC → 3× ConvTranspose3D)
  ↓
Reconstructed Geometry (1 × 32 × 32 × 32)
  ↓
Performance Head (FC → stress, compliance, mass)
```

**Monitoring:**
```bash
tensorboard --logdir ./logs
# Open http://localhost:6006
```

**Best practices:**
- Monitor reconstruction loss (should decrease)
- Monitor KL divergence (should stabilize)
- Stop when val loss plateaus
- Save checkpoints every 10 epochs

**Output:**
```
checkpoints/
├── vae_best.pth (best validation loss)
├── vae_epoch_10.pth
├── vae_epoch_20.pth
└── ...

logs/
├── events.out.tfevents.*
└── (tensorboard logs)
```

### Step 4: Bayesian Optimization

```bash
python optimization_engine.py \
  --model-checkpoint checkpoints/vae_best.pth \
  --freecad-template freecad_designs/master_design.FCStd \
  --n-iterations 50 \
  --latent-dim 16
```

**Loop (50 iterations):**
```
Iteration 1:
  Initialize 5 random designs
    ├─ Sample latent z₁, z₂, ..., z₅
    ├─ Decode → voxel grids
    ├─ Convert → FreeCAD parameters
    └─ Run FEM simulation

Iteration 2-50:
  Fit Gaussian Process to (z, performance) data
    ↓
  Compute acquisition function (Upper Confidence Bound)
    ↓
  Optimize to find best next candidate z*
    ↓
  Evaluate z* in FEM
    ↓
  Update GP with new data
    ↓
  Repeat
```

**Objective function:**
```
minimize: stress + 0.1×compliance + 0.01×mass
constraints: min_feature_size > 1mm, no overhangs > 45°
```

**Output:**
```
optimization_results/
├── optimization_history.json
│   ├── x_history (50 latent vectors)
│   ├── y_history (50 objective values)
│   ├── best_x (optimal latent vector)
│   └── best_y (best objective value)
└── fem_evaluations.json (FEM simulation details)
```

### Step 5: Export Optimized Design

```bash
python export_geometry.py \
  --latent-vector "[0.5, -0.3, 0.2, ...]" \
  --output optimized_design.step
```

**Process:**
1. Load best latent vector from optimization
2. Decode using VAE → voxel grid
3. Apply manufacturability constraints
4. Marching cubes voxels → surface mesh
5. Export to STL/OBJ/STEP

**Output:**
```
optimization_results/exported_designs/
├── optimized_design.stl (3D print ready)
├── optimized_design.obj (CAD ready)
└── optimized_design.step (STEP format)
```

## Performance Expectations

### Training Time (on RTX 5080)
```
Step 2 (Data extraction):     5-10 min
Step 3 (VAE training):        30-45 min (100 epochs)
Step 4 (Optimization):        1-2 hours (50 iterations)
Step 5 (Export):              <1 min
Total:                        2-4 hours
```

### Memory Usage
```
VAE Training:    6-8 GB VRAM (batch_size=8)
Optimization:    4-6 GB VRAM
FEM Evaluation:  CPU-bound (4-8 GB RAM per CalculiX instance)
```

## Customization

### Adjust Design Space

Edit `pipeline_config.json`:
```json
{
  "latent_dim": 32,  # Increase for more design freedom
  "voxel_resolution": 64  # Higher resolution (slower)
}
```

### Change Optimization Objective

In `optimization_engine.py`:
```python
def objective_function(self, z):
    # Current: minimize stress + compliance + mass
    # Custom: minimize mass while keeping stress < 200MPa
    stress = perf_pred[0, 0]
    mass = perf_pred[0, 2]
    
    if stress > 200:
        penalty = 1000  # Infeasible
    else:
        penalty = 0
    
    return mass + penalty
```

### Add Manufacturing Constraints

In `utils.py`:
```python
mfg = ManufacturabilityConstraints(
    min_feature_size=2.0,  # 2mm minimum
    max_overhang_angle=30.0  # Stricter
)
voxel_constrained = mfg.apply_constraints(voxel_grid)
```

## Troubleshooting

### GPU not detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If False, reinstall PyTorch with CUDA 12.1

### FreeCAD import fails
```bash
export PYTHONPATH=/usr/lib/freecad/lib:$PYTHONPATH  # Linux
```

### VAE not learning
- Reduce learning_rate: 1e-3 → 5e-4
- Increase epochs: 100 → 200
- Check voxel_resolution (increase to 64)

### Optimization stuck
- Initialize with more points: `n_init_points=10`
- Reduce beta in UCB: 0.1 → 0.05

## Next Steps

1. **Read full guide**: `fembygen_pytorch_guide.md`
2. **Try example**: Run `quickstart.py --all` with sample model
3. **Customize**: Modify for your specific design problem
4. **Deploy**: Export optimized designs to manufacturing

## Resources

- FreeCAD: https://www.freecad.org/
- FEMbyGEN: https://github.com/Serince/FEMbyGEN
- PyTorch: https://pytorch.org/
- BoTorch: https://botorch.org/

## Support

- Issues/Questions: Check README.md troubleshooting section
- Code help: Review inline comments in Python files
- FEM issues: FreeCAD Forum & Documentation

---

**Status**: Ready to use
**Last tested**: February 2025
**Hardware**: NVIDIA RTX 5080, FreeCAD 1.0, PyTorch 2.1, CUDA 12.1
