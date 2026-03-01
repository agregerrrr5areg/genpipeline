# GenPipeline — Generative Design via FEM + VAE + Bayesian Optimisation

A high-performance pipeline that combines **FreeCAD FEM simulation**, **3D Variational Autoencoders (VAE)**, and **Bayesian Optimisation** to autonomously discover structurally optimal and organic geometries.

---

## 🚀 Current State: Fully Operational

The pipeline has been upgraded from a scaffolding state to a high-precision discovery engine. The "Active Learning Loop" is closed, and the system is specifically tuned for **RTX 5080 (Blackwell)** hardware.

| Component | Status | Notes |
|-----------|--------|-------|
| **Closed-Loop BO** | ✅ Active | BO now triggers real FreeCAD simulations via the WSL2 bridge. |
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
   
2. REPRESENTATION LEARNING (vae_design_model.py)
   ↑ 3D Convolutional VAE learns latent "DNA" of structures. 
   ↑ Predictor heads learn to map latent vectors to Performance (Stress/Mass) and Parameters (h/r).

3. ORGANIC DISCOVERY (optimization_engine.py)
   ↑ BoTorch Gaussian Process explores the VAE latent space.
   ↑ PROPOSE z → ORGANIC FILTER → PHYSICALITY CHECK → BRIDGE SIMULATION → UPDATE GP.

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
The decoder now implements an **Organic Density Filter**. Instead of sharp pixelated voxels, the optimizer explores a continuous density field, leading to structures that resemble biological growth or high-end topology optimization.

### 📐 Scale Preservation
We solved the "unit voxel" problem. Every design sample now carries its **BoundBox** metadata. The exported STL is shifted and scaled back to its real-world origin and dimensions, making it ready for immediate 3D printing or CAD assembly.

### 🛡️ Physicality Guardrails
The optimizer is barred from proposing "void" designs. Before a simulation is even attempted, the system checks for **Structural Connectivity**. If a design is fragmented or too thin, it is penalized and rejected, saving CPU cycles.

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
