# GenPipeline — Generative Design via FEM + VAE + Bayesian Optimisation

A high-performance pipeline that combines **FreeCAD FEM simulation**, **3D Variational Autoencoders (VAE)**, and **Bayesian Optimisation** to autonomously discover structurally optimal and organic geometries.

---

## 🚀 Current State: Fully Operational

The pipeline has been upgraded from a scaffolding state to a high-precision discovery engine. The "Active Learning Loop" is closed, and the system is specifically tuned for **RTX 5080 (Blackwell)** hardware.

| Component | Status | Notes |
|-----------|--------|-------|
| **Closed-Loop BO** | ✅ Active | BO now triggers real FreeCAD simulations via the WSL2 bridge. |
| **Multi-Geometry BO** | ✅ Active | Cantilever, L-bracket, tapered beam, ribbed plate — each with correct BCs and per-geometry BO bounds. |
| **Voxel FEM Path** | ✅ Active | Direct CalculiX C3D8 hex mesh from decoded voxels — bypasses FreeCAD entirely (`--voxel-fem`). |
| **Organic Filtering**| ✅ Active | VAE decoding uses a Gaussian-Heaviside density filter for smooth, bone-like forms. |
| **Scale Preservation**| ✅ Active | 10x50mm in FreeCAD remains exactly 10x50mm in the exported STL. |
| **Hardware Speed** | ✅ Optimized | Batch size 128 + FP8 (uint8) storage + Pinned memory for RTX 5080. |
| **Physicality Guard** | ✅ Enforced | Strict connectivity and volume fraction invariants prevent "void" designs. |
| **Material Physics** | ✅ JSON-Driven | `materials.json` defines E, Poisson, Density, and Thermal Expansion. |

---

## 🧠 Core Architecture

```
1. DATA GENERATION (freecad_bridge.py)
   ↑ Parallel (ThreadPool) extraction of Parametric Variants (C3D10 Quadratic elements).
   ↑ Supports 4 geometry types: cantilever, lbracket, tapered, ribbed.

2. REPRESENTATION LEARNING (vae_design_model.py)
   ↑ 3D Convolutional VAE learns latent "DNA" of structures.
   ↑ Predictor heads learn to map latent vectors to Performance (Stress/Mass) and Parameters (h/r).

3. ORGANIC DISCOVERY (optimization_engine.py)
   ↑ BoTorch Gaussian Process explores the design space.
   ↑ Stage 1: GPU surrogate sweep in 16-D latent space (VAE predictor).
   ↑ Stage 2a (default): Physical BO in per-geometry (h, r) space via FreeCAD bridge.
   ↑ Stage 2b (--voxel-fem): Latent-space BO with direct CalculiX hex mesh (no FreeCAD).

4. ACCURATE EXPORT (utils.py)
   ↑ Scale-preserved Marching Cubes using original FreeCAD BoundBox metadata.
```

---

## 🛠 Setup & Usage

### Precision Installation (RTX 5080)
```bash
python -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 1. High-Precision Generation (300 variants)
```bash
python freecad_bridge.py generate --n-variants 300 --n-workers 16 --geometry-types cantilever lbracket tapered ribbed
```

### 2. High-Speed VAE Training (1000 epochs)
```bash
# Uses Batch 128 and Pinned Memory for maximum RTX 5080 saturation
python quickstart.py --step 3 --epochs 1000
```

### 3. Organic Bayesian Optimization
```bash
# Anchors search using existing valid designs; enforces connectivity invariants
python quickstart.py --step 4 --n-iter 100
```

### 4. Scale-Preserved Export
```bash
python quickstart.py --step 5
```

---

## 💎 Features

### 🌿 Organic Discovery
The decoder implements an **Organic Density Filter**. Instead of sharp pixelated voxels, the optimizer explores a continuous density field, leading to structures that resemble biological growth or high-end topology optimization.

### 🔷 Multi-Geometry Bayesian Optimisation
Four geometry types are fully wired end-to-end with correct FEM boundary conditions and dedicated BO parameter bounds:

| Geometry | Fixed BC | Load BC | BO params |
|----------|----------|---------|-----------|
| `cantilever` | left x-face | right x-face | h ∈ [5,20] mm, r ∈ [0,5] mm |
| `lbracket` | bottom z-face (vertical arm) | x-max tip (horizontal arm) | arm_h ∈ [8,25] mm, thickness ∈ [5,20] mm |
| `tapered` | left x-face | right x-face | h_start ∈ [8,25] mm, taper ∈ [2,7] |
| `ribbed` | left x-face | right x-face | rib_h ∈ [6,20] mm, plate_frac ∈ [2,6] |

Switch geometry via `"geometry_type"` in `sim_config.json`.

### ⚡ Direct CalculiX Voxel FEM (`--voxel-fem`)
`voxel_fem.py` converts VAE-decoded voxel grids directly into CalculiX C3D8 hex meshes and runs `ccx` without FreeCAD. This enables true non-parametric topology optimisation: the full 32³ voxel structure is evaluated, not just an (h, r) parameter pair.

```bash
# Unit test (10×10×10 solid cube, stress + displacement verified):
python voxel_fem.py --test

# Run full optimisation with voxel FEM in Stage 2:
python optimization_engine.py --model-checkpoint checkpoints/vae_best.pth --voxel-fem
```

### 📐 Scale Preservation
Every design sample carries its **BoundBox** metadata. The exported STL is shifted and scaled back to its real-world origin and dimensions, ready for 3D printing or CAD assembly.

### 🛡️ Physicality Guardrails
Before a simulation is attempted the system checks for **Structural Connectivity**. Fragmented or too-thin designs are penalised and skipped, saving CPU cycles.

---

## 🧊 Hardware & Precision (Blackwell/RTX 5080)

- **FP8 Simulation**: Voxel grids are stored as `uint8` in memory, reducing VRAM usage by 4x during massive training runs.
- **Stability Workaround**: BoTorch `SingleTaskGP` models are automatically offloaded to the **CPU** via `blackwell_compat.py` to avoid the CUDA 12.8 batch GEMM bug, while VAE inference remains on the GPU for speed.

---

## 📂 Configuration

- **`materials.json`**: Define Young's Modulus, Yield Strength, Density, and Thermal constants.
- **`pipeline_config.json`**: Fine-tune VAE dimensions, learning rates, and BO acquisition strategies.
- **`sim_config.json`**: Set optimization weights (Stiffness vs. Mass) and safety factors.

---

## 📄 License
This project is for generative design research and industrial optimization. Refer to [FreeCAD](https://www.freecad.org/) and [BoTorch](https://botorch.org/) for underlying dependency licenses.
