# Getting Started

This guide gets you from zero to a running pipeline on **WSL2 + Windows FreeCAD + RTX 5080**.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Windows 11 | any | Host OS |
| WSL2 | any | Ubuntu recommended |
| Python | 3.10+ | Inside WSL2 |
| CUDA | 12.8 | RTX 5080 (Blackwell) |
| FreeCAD | 1.0+ | **Windows only** — do not install in WSL |
| FEMbyGEN | latest | FreeCAD addon |

---

## Part 1 — WSL2 Python Environment

### 1. Clone the repo

```bash
git clone https://github.com/agregerrrr5areg/genpipeline.git
cd genpipeline
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch for Blackwell (RTX 5080 = sm_120, needs cu128)

> **Important:** Do NOT use the cu121 index — it will silently run on CPU.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install remaining dependencies

```bash
pip install numpy scipy scikit-image scikit-learn \
            trimesh pyvista \
            botorch gpytorch \
            tensorboard tensorboardX \
            matplotlib seaborn tqdm pyyaml
```

### 5. Verify GPU

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
# Expected: 2.10.0+cu128  NVIDIA GeForce RTX 5080
```

### 6. Run synthetic test (no FreeCAD needed)

Verifies the full pipeline — VAE training on GPU + Bayesian optimisation — without any real FEM data.

```bash
python synthetic_test.py
```

Expected output (takes ~30 seconds):
```
=== Synthetic pipeline test | device=cuda | botorch_device=cpu ===
...
Training done in 4.5s  |  best val loss: 0.09
...
=== Pipeline test PASSED ===
```

> **Note on Blackwell:** BoTorch's Gaussian Process runs on CPU (`botorch_device=cpu`).
> This is intentional — a known cuBLAS bug in CUDA 12.8 crashes batched GEMM on sm_120.
> The VAE and all Conv3D ops run on GPU normally. See `blackwell_compat.py` for details.

---

## Part 2 — Windows FreeCAD Setup

Do this on the **Windows side** (not in WSL).

### 1. Install FreeCAD

Download and install FreeCAD 1.0 from https://www.freecad.org/

Default install path: `C:\Program Files\FreeCAD 1.0\`

### 2. Install FEMbyGEN addon

Inside FreeCAD:
```
Tools → Addon Manager → search "FEMbyGEN" → Install → restart FreeCAD
```

### 3. Create a parametric model

In FreeCAD:

1. **Part Design workbench** → new sketch → add geometry (e.g. a bracket, plate, beam)
2. Constrain dimensions with named parameters (e.g. `thickness`, `radius`, `height`)
3. Add pad/pocket/fillet features referencing those parameters
4. Save as `master_design.FCStd`

### 4. Set up FEM analysis

In FreeCAD **FEM workbench**:

```
Model menu → Analysis container
  ├── Constraints → Fixed (select face)
  ├── Constraints → Force (e.g. 1000 N on opposite face)
  ├── Material → Steel or Aluminium
  ├── Mesh → Gmsh (element size 2–5 mm)
  └── Solver → CalculiX
```

Run one simulation manually to confirm it works (Solve → Run).

### 5. Generate design variations with FEMbyGEN

In FreeCAD **FEMbyGEN workbench**:

```
1. Click "Initialize" → creates Parameters spreadsheet in the model

2. Set parameter ranges in the spreadsheet, e.g.:
     thickness  →  min=1.5, max=3.5  (mm)
     radius     →  min=3.0, max=7.0  (mm)

3. Click "Generate" → creates 50–100 design variant files in a folder

4. Click "FEA" → runs CalculiX on all variants (takes 1–4 hours)
```

The generated `.FCStd` files (one per variant) are what the pipeline ingests.

---

## Part 3 — Connect FreeCAD to WSL2

Back in **WSL2**:

```bash
source venv/bin/activate

python freecad_bridge.py --designs-dir /mnt/c/Users/YOU/path/to/variants
```

This will:
1. Auto-detect `FreeCADCmd.exe` in common install paths
2. Copy the extraction script to `C:\Windows\Temp\`
3. Call FreeCAD headlessly on each `.FCStd` file to extract stress, compliance, mass, and export STL
4. Voxelise each STL to a 32³ grid
5. Save `fem_data/fem_dataset.pt` ready for training

### If FreeCAD is in a non-standard location

```bash
python freecad_bridge.py \
  --designs-dir  /mnt/c/Users/YOU/designs \
  --output-dir   ./fem_data \
  --freecad-path "/mnt/c/Program Files/FreeCAD 1.0"
```

---

## Part 4 — Train and Optimise

### Train the VAE

```bash
python quickstart.py --step 3 --epochs 100
```

Monitor training in TensorBoard (open a second terminal):
```bash
source venv/bin/activate
tensorboard --logdir ./logs
# Open http://localhost:6006
```

Watch for:
- `train/recon_loss` decreasing steadily
- `train/kl_loss` stabilising (not exploding)
- Val loss tracking train loss (no large gap = not overfitting)

### Run Bayesian optimisation

```bash
python quickstart.py --step 4 --n-iter 50
```

Each iteration decodes a latent vector → runs FEM → updates the GP surrogate.
Results are saved to `optimization_results/optimization_history.json`.

### Export best design

```bash
python quickstart.py --step 5
```

Outputs an STL/STEP of the best design found to `optimization_results/exported_designs/`.

---

## Configuration

All hyperparameters are in `pipeline_config.json`.  Key ones:

| Key | Default | When to change |
|-----|---------|----------------|
| `voxel_resolution` | 32 | Raise to 64 for finer geometry (8× more VRAM) |
| `latent_dim` | 16 | Raise to 32 if VAE underfits complex shapes |
| `batch_size` | 8 | Lower to 4 if VRAM runs out |
| `epochs` | 100 | Raise to 200 if val loss still dropping at 100 |
| `beta_vae` | 1.0 | Lower to 0.5 for more diverse samples |
| `n_optimization_iterations` | 50 | Raise for better optima (slower) |

---

## Troubleshooting

### `torch.cuda.is_available()` returns False
You installed the wrong PyTorch build. Reinstall:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### `FreeCADCmd.exe not found`
FreeCAD isn't installed on Windows, or it's in a non-standard path. Use:
```bash
python freecad_bridge.py --designs-dir ... --freecad-path "/mnt/c/Program Files/FreeCAD 1.0"
```

### FreeCAD extraction returns no STL
The `.FCStd` file has no solid body (only a sketch or FEM mesh).
In FreeCAD, make sure your model has a solid Part Design feature (Pad/Pocket) before running FEMbyGEN.

### VAE loss not decreasing
```json
"learning_rate": 0.0005,
"beta_vae": 0.5,
"voxel_resolution": 64
```

### BoTorch `cublasSgemmStridedBatched` crash
You're running the GP on GPU on a Blackwell card. The bridge already sets `botorch_device=cpu` automatically via `blackwell_compat.py`. If you see this crash, confirm `from blackwell_compat import botorch_device` is being used everywhere BoTorch models are created.

---

## File Reference

```
genpipeline/
├── synthetic_test.py        — smoke test (no FreeCAD needed)
├── freecad_bridge.py        — WSL2 ↔ Windows FreeCAD bridge
├── freecad_scripts/
│   └── extract_fem.py       — runs inside FreeCADCmd.exe
├── fem_data_pipeline.py     — voxelisation + dataset builder
├── vae_design_model.py      — 3D VAE + performance predictor
├── optimization_engine.py   — Bayesian optimisation loop
├── utils.py                 — geometry utilities + constraints
├── quickstart.py            — end-to-end pipeline runner
├── blackwell_compat.py      — RTX 5080 cuBLAS workaround
└── pipeline_config.json     — all hyperparameters
```

---

**Tested:** FreeCAD 1.0 · PyTorch 2.10.0+cu128 · CUDA 12.8 · RTX 5080 · WSL2 Ubuntu
