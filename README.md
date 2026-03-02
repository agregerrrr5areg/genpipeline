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

## Current Status (2026-03-03)

| Stage | Status | Notes |
|-------|--------|-------|
| FEM data generation | ✅ Complete | 10 variants, 1.4 s each via FreeCAD WSL2 bridge |
| Dataset | ✅ Built | 32³ (2.8 MB) and 64³ (53 MB) `.pt` files |
| VAE training | ✅ 300 epochs | `checkpoints/vae_best.pth`, train loss 0.103 |
| Bayesian optimisation | ✅ 20+ iters | Best objective −0.1058, 16.2% occupancy |
| Integration test | ✅ Added | `tests/test_integration_decode_fem.py` |
| SIMP data augmentation | ⏳ Pending | `topo_data_gen.py` exists, not yet run at scale |
| Geometry conditioning | ⏳ Pending | Single shared latent space for all 4 geometry families |

See `PROGRESS.md` for full dated log.

---

## MANDATE: No Non-Physical Data
All training data must originate from physics-based simulations (FreeCAD/CalculiX or SIMP). **Non-physical 'synthetic' data** (random noise, pure geometric heuristics) **is strictly forbidden.** The model must exclusively learn from physically plausible mechanics.

---

## Architecture

- `genpipeline/topology/simp_solver_gpu.py` — PyTorch-based 3D SIMP.
- `genpipeline/fem/` — Unified physics/FEM package.
  - `data_pipeline.py` — Consolidates JSON/Parquet results.
  - `voxel_fem.py` — Direct CalculiX voxel FEM solver.
  - `data/` — Training voxel grids, FEM metrics, dataset `.pt` files.
- `genpipeline/` — Core pipeline logic package.
  - [vae_design_model.py](file:///home/genpipeline/genpipeline/vae_design_model.py): 3D VAE with performance prediction.
  - [optimization_engine.py](file:///home/genpipeline/genpipeline/optimization_engine.py): Multi-objective parallel BO.
  - [schema.py](file:///home/genpipeline/genpipeline/schema.py): Type-safe Pydantic data models.
- [quickstart.py](file:///home/genpipeline/quickstart.py): Integrated CLI/API orchestrator.

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

## Testing & Verification

The project includes a comprehensive suite of tests to ensure data integrity and structural accuracy.

### 1. Run All Tests
To run the full suite using `pytest`:
```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/
```

### 2. Verify Schema Validation
The pipeline uses **Pydantic** for rigorous input validation. You can verify the latest validators (e.g., ensuring `voxel_resolution` is a multiple of 16) by running:
```bash
PYTHONPATH=. ./venv/bin/python tests/test_schema_validation.py
```

### 3. Verify Hardware Logging
To check if hardware telemetry is working on your RTX 50 series card:
```bash
# This will run a small forward pass and log VRAM/Temp
PYTHONPATH=. ./venv/bin/python vae_design_model.py
```

### 4. Regression Testing for SIMP
Ensure the topology optimisation engine is mathematically correct:
```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_simp_solver.py
```

---

## Hardware Notes (Blackwell)

The RTX 50 series requires `torch` from the `cu128` index for sm_120 support. Due to a cuBLAS driver bug in CUDA 12.8 (batched GEMM crash), BoTorch GP models are automatically routed to CPU via `blackwell_compat.py`. VAE training and SIMP GPU solving utilise full CUDA acceleration.

---

## License

Proprietary. All rights reserved. You are granted permission to use this software for evaluation and research, but redistribution or resale is strictly prohibited. See the [LICENSE](LICENSE) file for the full legal text.
