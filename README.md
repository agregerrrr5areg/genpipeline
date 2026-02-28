# GenPipeline — Generative Design via FEM + VAE + Bayesian Optimisation

A pipeline that combines FreeCAD FEM simulation with a 3D VAE and Bayesian
optimisation to search for structurally optimal geometries.

---

## Current State

All the components exist and run individually. The loop is not yet closed.

| Component | Status | Notes |
|-----------|--------|-------|
| FreeCAD FEM bridge | Working | WSL → Windows, ~1.4 s/variant |
| FEM data generation | Partial | 10 cantilever variants generated |
| Dataset | Too small | 4 voxel samples — VAE cannot learn from this |
| 3D VAE | Trained but unreliable | 500 epochs on 4 samples; weights are essentially noise |
| Bayesian optimisation | Runs | 28 iters completed; surrogate trained on 4 points is meaningless |
| SIMP topology solver | Working | Produces STL; not integrated with FEM validation |
| Active learning loop | Not built | BO → FEM → update dataset → retrain cycle doesn't exist yet |

**Root problem:** The VAE and BO are only as good as the training data.
With 4 samples they produce garbage. Everything else is scaffolding
waiting for a real dataset.

---

## Architecture

```
FreeCAD FEM (freecad_bridge.py)
    ↓  parametric variants → STL + stress/compliance/mass JSON
fem_data_pipeline.py
    ↓  voxelise 32³, build PyTorch dataset
vae_design_model.py
    ↓  train encoder/decoder + performance predictor head
optimization_engine.py  ←─── BoTorch GP + UCB acquisition
    ↓  propose z → decode → FEM validate → update GP
Export best design as STL
```

---

## Repository Structure

```
genpipeline/
├── freecad_bridge.py          WSL→Windows FEM runner
├── freecad_scripts/
│   ├── run_fem_variant.py     Runs inside FreeCAD; builds cantilever, runs CalculiX
│   └── extract_fem.py         Extracts results from existing .FCStd files
├── fem_data_pipeline.py       Voxelisation + PyTorch dataset builder
├── vae_design_model.py        3D convolutional VAE + performance predictor
├── optimization_engine.py     BoTorch GP + UCB Bayesian optimisation
├── sim_config.py              Material presets, load, weights — saved to sim_config.json
├── topology/
│   ├── simp_solver.py         3D SIMP density optimiser
│   ├── mesh_export.py         Density field → STL via marching cubes
│   └── solver.py              TopologySolver facade
├── cuda_kernels/              Custom CUDA voxelisation + marching cubes (RTX 5080)
├── blackwell_compat.py        RTX 5080 workaround — keeps BoTorch on CPU
├── utils.py                   Geometry utilities, manufacturability checks
├── quickstart.py              Orchestrates all pipeline steps
├── pipeline_config.json       Hyperparameters
├── sim_config.json            Simulation settings (auto-generated)
├── tests/                     13 unit tests
└── fem_variants/              Generated FEM results + STL meshes
```

---

## Setup

### Requirements

- FreeCAD 1.0 (Windows) — headless via `freecad.exe --console`
- Python 3.10+
- CUDA 12.8 + PyTorch (RTX 5080 / Blackwell)

### Install

```bash
python -m venv venv
source venv/bin/activate

# RTX 5080 / Blackwell — must use cu128
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install numpy scipy scikit-image scikit-learn trimesh pyvista
pip install botorch gpytorch tensorboard
pip install numpy-stl
```

### FreeCAD path

Set `FREECAD_PATH` in `freecad_bridge.py` to match your install:

```
/mnt/c/Users/<you>/AppData/Local/Programs/FreeCAD 1.0
```

### Run FEM variants

```bash
# Generate parametric cantilever variants (h_mm, r_mm sweeps)
python freecad_bridge.py generate --output-dir fem_variants --n 50

# Or single variant
python freecad_bridge.py run --h-mm 12 --r-mm 2 --output-dir fem_variants
```

### Run pipeline

```bash
python quickstart.py --step 2   # Build dataset from fem_variants/
python quickstart.py --step 3 --epochs 200   # Train VAE
python quickstart.py --step 4 --n-iter 50    # Bayesian optimisation
python quickstart.py --step 5                # Export best STL
```

---

## Known Hardware Notes (RTX 5080 / Blackwell)

`cublasDgemmStridedBatched` is broken for batch ≥ 2 in CUDA 12.8 +
PyTorch 2.10. BoTorch GP models are kept on CPU via `blackwell_compat.py`.
VAE training runs on GPU normally. Monitor pytorch/pytorch for a fix.

---

## Next Steps

The project needs these things in order before it produces useful results.

### 1 — Expand the FEM dataset (blocker for everything else)

The VAE needs at minimum ~200 diverse samples to learn a useful latent
space. Currently there are 4.

```bash
python freecad_bridge.py generate --n 200 \
    --h-range 5 25 --r-range 0 10 --output-dir fem_variants
python quickstart.py --step 2
```

Also consider adding geometry variation beyond h/r — ribs, fillets,
cutouts — so the VAE learns a richer design space.

### 2 — Retrain VAE on real data

Once the dataset has 200+ samples, retrain from scratch with longer
training and monitor reconstruction loss on validation set.

```bash
python quickstart.py --step 3 --epochs 500 --latent-dim 32
```

A working VAE should reconstruct held-out voxel grids with low binary
cross-entropy and produce smooth interpolations in latent space.

### 3 — Close the BO–FEM loop

Right now `optimize_step()` uses the VAE performance predictor as its
objective. That predictor is trained on 4 points and is meaningless.

The loop needs to be:
1. BO proposes latent vector `z`
2. Decode `z` → voxel → extract geometry parameters
3. Run real FreeCAD FEM on the geometry
4. Feed actual stress/compliance back to update the GP

This requires wiring `real_eval=True` into the BO loop and making sure
`fem_evaluator` is populated. The infrastructure exists in
`optimization_engine.py`; it just isn't connected.

### 4 — Active learning cycle

Once step 3 works, add a cycle that:
- Runs BO to propose the most uncertain geometry
- Validates it with FEM
- Adds result to the training set
- Periodically fine-tunes the VAE on the growing dataset

This is the core loop that makes the system self-improving.

### 5 — Improve mesh and stress accuracy

Current FEM uses C3D4 linear tetrahedra. These give ~50–70% of the
theoretical stress value for a cantilever. Switching to C3D10 (quadratic)
or refining the mesh in high-stress regions would give trustworthy numbers.

```python
# In run_fem_variant.py
mesh_obj.ElementOrder = "2nd"   # second-order tets
mesh_obj.CharacteristicLengthMax = "3 mm"
```

### 6 — Multi-geometry support

The current FEM script only handles a rectangular cantilever beam.
A useful generative design system needs a wider shape vocabulary:
L-brackets, plates with holes, rib structures. Each new geometry type
needs its own FreeCAD script and voxelisation logic.

### 7 — Topology → parametric export

The SIMP solver produces density fields but these can't be manufactured
directly. The missing step is converting the SIMP output into a clean
parametric FreeCAD model (e.g. via skeleton extraction or feature
recognition), which can then be FEM-validated and exported to STEP.

---

## Tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

13 tests covering sim_config, SIMP solver, mesh export, and topology facade.

---

## References

- [FreeCAD](https://www.freecad.org/)
- [BoTorch](https://botorch.org/)
- [SIMP topology optimisation](https://doi.org/10.1007/s001580050176)
- [β-VAE](https://openreview.net/forum?id=Sy2fchgcW)
