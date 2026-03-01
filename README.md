# GenPipeline

Generative design pipeline combining FreeCAD FEM simulation, 3D SIMP topology optimisation, a 3D VAE, and Bayesian optimisation to discover mechanically optimal geometries. Optimised for RTX 50 series card (Blackwell/CUDA 12.8).

---

## Key Features

- **Hybrid Physics Engine** — High-fidelity FreeCAD (CalculiX) simulations or high-speed 3D SIMP topology optimisation (CPU & PyTorch GPU).
- **One-Command Bootstrap** — Generate 1000s of physics-based training samples without FreeCAD via the integrated topology data generator.
- **Topology Refinement** — Multi-objective BO loop can refine VAE-decoded candidates using 20 SIMP iterations before final FEM validation.
- **Joint Preservation** — Define non-design domains (locked solid regions) directly in FreeCAD to ensure structural integrity for bolt holes, flanges, and multi-part assemblies.
- **Multi-Geometry Support** — Out-of-the-box support for Cantilever, L-Bracket, Tapered Beam, and Ribbed Plate families.
- **Blackwell Optimised** — Custom 3D VAE and CUDA kernels (voxelisation up to 832× faster) tuned for RTX 50 series card precision and memory constraints.
    
---

## MANDATE: No Non-Physical Data
All training data must originate from physics-based simulations (FreeCAD/CalculiX or SIMP). **Non-physical 'synthetic' data** (random noise, pure geometric heuristics) **is strictly forbidden.** The model must exclusively learn from physically plausible mechanics.

---

## Architecture

- [topology/simp_solver_gpu.py](file:///home/genpipeline/topology/simp_solver_gpu.py): PyTorch-based 3D SIMP.
- [fem/](file:///home/genpipeline/fem/): Unified physics/FEM package.
  - [data_pipeline.py](file:///home/genpipeline/fem/data_pipeline.py): Consolidates JSON/Parquet results.
  - [voxel_fem.py](file:///home/genpipeline/fem/voxel_fem.py): Voxel-based FEM solver.
  - [data/](file:///home/genpipeline/fem/data/): Main training results.
- [vae_design_model.py](file:///home/genpipeline/vae_design_model.py): 3D VAE with performance prediction.
- [optimisation_engine.py](file:///home/genpipeline/optimisation_engine.py): Multi-objective BO.
- [quickstart.py](file:///home/genpipeline/quickstart.py): Integrated CLI/API for the pipeline.

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

FreeCAD 1.0 is required for parametric variants (Windows host). SIMP-based features run natively in Linux/WSL2.

---

## Usage

### 1. Bootstrap (Generate Data)
Generate physics-grounded topology data without requiring FreeCAD:
```bash
python quickstart.py --topo-data --n-samples 200
```

### 2. Train VAE
```bash
python quickstart.py --step 3 --epochs 300 --batch-size 32
```

### 3. Optimise Designs
Run Bayesian optimisation with topology refinement enabled:
```bash
python quickstart.py --step 4 --n-iter 50 --topo-refine
```

### 4. Export Best Result
```bash
python quickstart.py --step 5
```

---

## Geometry Families

| Geometry | Fixed face | Load face | Physics Backend |
|----------|-----------|-----------|-----------------|
| `cantilever` | x-min | x-max | FreeCAD / SIMP |
| `lbracket` | z-min | x-max | FreeCAD / SIMP |
| `tapered` | x-min | x-max | FreeCAD / SIMP |
| `ribbed` | x-min | x-max | FreeCAD / SIMP |

---

## Verification

Run the comprehensive unit test suite:
```bash
python -m pytest tests/test_simp_solver.py
```

---

## Hardware Notes (Blackwell)

The RTX 50 series requires `torch` from the `cu128` index for sm_120 support. Due to a cuBLAS driver bug in CUDA 12.8 (batched GEMM crash), BoTorch GP models are automatically routed to CPU via `blackwell_compat.py`. VAE training and SIMP GPU solving utilise full CUDA acceleration.

---

## License

Research and industrial optimisation use. Refer to [FreeCAD](https://www.freecad.org/) and [BoTorch](https://botorch.org/) for dependency licenses.
