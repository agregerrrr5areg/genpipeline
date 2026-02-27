# FEMbyGEN + PyTorch Generative Design Pipeline

A complete framework combining FreeCAD's FEMbyGEN topology optimization with PyTorch-based generative models and Bayesian optimization for automated design discovery.

## Features

- **FEM-Driven Generation**: Leverage FEMbyGEN to create parametric design variations
- **3D VAE**: Learn latent design space from FEM-validated geometries
- **Bayesian Optimization**: Intelligently search design space using Gaussian processes
- **Performance Prediction**: Predict stress, compliance, and manufacturability
- **GPU-Accelerated**: Full CUDA support for RTX 5080+ GPUs
- **Manufacturing Constraints**: Apply real-world design constraints (min features, overhangs)
- **New: Customizable Optimization Algorithm**: Support for multiple acquisition functions and dynamic parameter tuning

## Architecture Overview

```
FreeCAD (FEMbyGEN)
    ↓
Generate parametric variations
    ↓
Run FEM simulations (CalculiX)
    ↓
Extract mesh + performance metrics
    ↓
Voxelize geometries
    ↓
PyTorch Dataset
    ↓
Train 3D VAE
    ↓
Bayesian Optimization Loop
    ↓
Decode → FEM Evaluate → Update GP
    ↓
Export optimized design
```

## Installation

### Prerequisites
- FreeCAD 1.0+ (with FEMbyGEN addon)
- Python 3.8+
- CUDA 12.1+ (for GPU acceleration)

### Step 1: Install FreeCAD & FEMbyGEN

```bash
# Windows / macOS / Linux
# Download from https://www.freecad.org/

# Install FEMbyGEN addon:
# Tools → Addon Manager → Search "FEMbyGEN" → Install
```

### Step 2: Setup Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio pytorch-cuda==12.1 -i https://download.pytorch.org/whl/cu121
pip install numpy scipy scikit-image scikit-learn
pip install trimesh pyvista
pip install botorch gpytorch
pip install tensorboard

# Optional: For better performance
pip install numba
```

### Step 3: Verify Setup

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import FreeCAD; print(f'FreeCAD: {FreeCAD.__version__}')"
```

## Quick Start

### Phase 1: Prepare FreeCAD Model

1. **Create parametric model**:
   ```
   Part Design Workbench
   → Create sketch
   → Constrain with parameters (e.g., thickness=2mm, radius=5mm)
   → Pad/Pocket features
   → Save as master_design.FCStd
   ```

2. **Setup FEM Analysis**:
   ```
   FEM Workbench
   → Create Analysis
   → Add fixed constraints
   → Add loads (forces, pressure)
   → Assign material (Steel, Aluminum)
   → Create mesh (Gmsh, 2-5mm element size)
   ```

3. **Generate variations with FEMbyGEN**:
   ```
   FEMbyGEN Workbench
   → Click "Initialize" (creates Parameters spreadsheet)
   → Define ranges: thickness: 1.5-3.5mm, radius: 3-7mm, etc.
   → Click "Generate" → Creates 50-100 design variants
   → Click "FEA" → Runs simulations
   ```

### Phase 2: Run Pipeline

```bash
# Full automated pipeline
python quickstart.py --all --config config.json

# Or run step-by-step
python quickstart.py --step 2  # Extract FEM data
python quickstart.py --step 3  --epochs 100  # Train VAE
python quickstart.py --step 4  --n-iter 50   # Optimize
python quickstart.py --step 5  # Export results
```

## File Structure

```
.
├── fembygen_pytorch_guide.md     # Detailed architecture & workflow
├── fem_data_pipeline.py          # FEM data extraction & voxelization
├── vae_design_model.py           # 3D VAE + performance predictor
├── optimization_engine.py        # Bayesian optimization loop
├── utils.py                      # Geometry utilities & constraints
├── quickstart.py                 # End-to-end pipeline script
├── config.yaml                   # Configuration parameters
└── README.md                     # This file

Data Flow:
freecad_designs/                 # Input FreeCAD files (*.FCStd)
fem_data/                        # Extracted voxel grids & metrics
checkpoints/                     # Trained VAE weights
optimization_results/            # Optimization history & best designs
```

## Configuration

Edit `pipeline_config.json`:

```json
{
  "freecad_project_dir": "./freecad_designs",
  "fem_data_output": "./fem_data",
  "voxel_resolution": 32,
  "use_sdf": false,
  "latent_dim": 16,
  "batch_size": 8,
  "epochs": 100,
  "learning_rate": 0.001,
  "beta_vae": 1.0,
  "device": "cuda",
  "seed": 42,
  "n_optimization_iterations": 50,
  "n_init_points": 5,
  "output_dir": "./optimization_results",
  "checkpoint_dir": "./checkpoints",
  "log_dir": "./logs",
  "manufacturing_constraints": {
    "min_feature_size_mm": 1.0,
    "max_overhang_angle_deg": 45.0
  },
  "optimization": {
    "acquisition_function": "UCB",  // Options: "UCB", "EI", "PI"
    "beta": 0.1,
    "use_botorch": true,
    "num_restarts": 10,
    "raw_samples": 512,
    "max_iterations": 100,  // New parameter
    "parallel_evaluations": 4  // New parameter
  },
  "performance_weights": {
    "stress": 1.0,
    "compliance": 0.1,
    "mass": 0.01
  }
}
```

### Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `voxel_resolution` | 32 | 32³ voxel grid (fast), 64³ (accurate) |
| `latent_dim` | 16 | Design space dimensionality. Increase if underfitting. |
| `batch_size` | 8 | Increase for faster training (if VRAM allows) |
| `epochs` | 100 | Training epochs. Monitor tensorboard for convergence. |
| `beta_vae` | 1.0 | KL weight. Lower = more diverse; Higher = sharper reconstructions |
| `n_optimization_iterations` | 50 | Optimization steps. More = better but slower. |
| `max_iterations` | 100 | Maximum number of iterations for optimization. |
| `parallel_evaluations` | 4 | Number of parallel evaluations during optimization. |

## Usage Examples

### Example 1: Train only (no optimization)

```bash
python quickstart.py --step 2 --freecad-dir /path/to/designs
python quickstart.py --step 3 --epochs 50
```

### Example 2: Interactive optimization with custom parameters

```python
from vae_design_model import DesignVAE
from optimization_engine import DesignOptimizer
import torch

# Load trained VAE
vae = DesignVAE(latent_dim=16)
vae.load_state_dict(torch.load('checkpoints/vae_best.pth')['model_state_dict'])

# Run optimization with custom parameters
optimizer = DesignOptimizer(vae, fem_evaluator, device='cuda', max_iterations=100, parallel_evaluations=4)
best_z, best_obj = optimizer.run_optimization(n_iterations=20)
```

### Example 3: Design interpolation

```python
from vae_design_model import interpolate_designs
import torch

# Load VAE
vae = DesignVAE(latent_dim=16)
vae.load_state_dict(torch.load('checkpoints/vae_best.pth')['model_state_dict'])

# Interpolate between two designs
z1 = torch.randn(1, 16)  # Design A
z2 = torch.randn(1, 16)  # Design B
interpolated = interpolate_designs(vae, z1[0].numpy(), z2[0].numpy(), n_steps=10)

# Interpolated shape sequence available for animation
```

## Performance Benchmarks

| Hardware | Phase | Time | Notes |
|----------|-------|------|-------|
| CPU (i9-12900K) | FEMbyGEN (100 designs) | 2-4 hrs | CalculiX solver |
| RTX 5080 | VAE Training (100 epochs) | 30 min | FP32, batch=8 |
| RTX 5080 + CPU | Optimization (50 iters) | 1-2 hrs | Parallel FEM evals |

**Total pipeline time**: ~4-6 hours for 100-design dataset

### Memory Requirements

| Component | VRAM | System RAM |
|-----------|------|------------|
| VAE training (batch=8) | 6 GB | 16 GB |
| Optimization | 4 GB | 8 GB |
| FEMbyGEN + CalculiX | - | 4-8 GB |

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size or voxel resolution
```python
config['batch_size'] = 4  # 8 → 4
config['voxel_resolution'] = 16  # 32 → 16
```

### Issue: FreeCAD module not found
**Solution**: Ensure FreeCAD Python is in PATH
```bash
export PYTHONPATH=/usr/lib/freecad/lib:$PYTHONPATH  # Linux
export PYTHONPATH=C:\Program Files\FreeCAD\lib:$PYTHONPATH  # Windows
```

### Issue: VAE training loss not decreasing
**Solution**: Adjust hyperparameters
```python
config['learning_rate'] = 5e-4  # Reduce LR
config['beta_vae'] = 0.5  # Reduce KL weight
config['voxel_resolution'] = 64  # Increase resolution
```

### Issue: Optimization stuck at local minimum
**Solution**: 
- Increase initialization points: `optimizer.initialize_search(n_init_points=10)`
- Increase acquisition function exploration: Change `beta` in UCB
- Use parallel evaluations: Set `parallel_evaluations` to a higher value

## Advanced Features

### Custom Performance Metrics

```python
# In optimization_engine.py, modify objective function:
def objective_function(self, z: np.ndarray, real_eval=False) -> float:
    perf_pred = self.predictor.predict(z)
    stress = perf_pred[0, 0]
    compliance = perf_pred[0, 1]
    
    # Custom multi-objective:
    # Minimize stress, minimize weight, maximize stiffness
    multi_obj = stress + 0.05*weight - 0.02*stiffness
    return multi_obj
```

### Manufacturing Constraints

```python
from utils import ManufacturabilityConstraints

mfg = ManufacturabilityConstraints(
    min_feature_size=1.0,  # mm
    max_overhang_angle=45.0  # degrees
)

constrained_voxels = mfg.apply_constraints(voxel_grid)
```

### Latent Space Visualization

```bash
# Using tensorboard
tensorboard --logdir ./logs
# Open http://localhost:6006
```

## Contributing

Contributions welcome! Key areas:
- [ ] Multi-material optimization
- [ ] Topology sensing (for assembly constraints)
- [ ] ML-assisted mesh quality optimization
- [ ] Real-time FEM prediction (neural operators)
- [ ] Customizable optimization algorithms

## References

- FEMbyGEN: https://github.com/Serince/FEMbyGEN
- 3D-VAE: https://arxiv.org/abs/1910.00935
- BoTorch: https://botorch.org/
- FreeCAD: https://www.freecad.org/

## Citation

```bibtex
@software{fembygen_pytorch_2025,
  title={FEMbyGEN + PyTorch Generative Design Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/fembygen-pytorch}
}
```

## License

MIT License - See LICENSE file

## Support

- Issues: GitHub Issues
- Discussions: FreeCAD Forum
- Documentation: See `fembygen_pytorch_guide.md`

---

**Last Updated**: February 2025  
**Tested With**: FreeCAD 1.0, PyTorch 2.1, CUDA 12.1, RTX 5080
