# GenPipeline Development Guide: VAE + Bayesian Optimisation

This document outlines the mathematical and architectural foundations of the GenPipeline project, specifically optimized for RTX 50 series (Blackwell) GPUs.

## 1. Variational Autoencoder (VAE) Architecture

The core of the pipeline is a 3D Variational Autoencoder that learns a compressed, continuous representation (latent space) of structural designs.

### Latent Space ($z$-vector)
- **Dimension**: Default is 32.
- **Representation**: Each point in the latent space corresponds to a unique topology.
- **Parameter Head**: The VAE includes a secondary regression head that predicts physical design parameters ($h_{mm}$, $r_{mm}$) directly from the $z$-vector. This ensures that generated designs are not just voxels, but correspond to real CAD dimensions.

### Performance Predictor
- The encoder also branches into a **Performance Predictor** that estimates:
  - Max Von Mises Stress (MPa)
  - Structural Compliance
  - Mass (kg)
- This allows for "Virtual FEM" evaluation during the early stages of Bayesian Optimisation.

## 2. Bayesian Optimisation (BO) Loop

We use Multi-Objective Bayesian Optimisation (MOBO) to discover the Pareto Front of stress vs. mass.

- **Acquisition Function**: Upper Confidence Bound (UCB) or Expected Improvement (EI).
- **Surrogate Model**: Gaussian Processes (GP) implemented via `BoTorch` and `GPyTorch`.
- **Constraint Handling**: We strictly enforce physical safety factors (e.g., Yield stress / 1.5).

## 3. GPU Optimisation (RTX 50 Series / Blackwell)

### Mixed Precision
- We utilise `torch.bfloat16` for VAE training to take advantage of Blackwell's improved BF16 throughput.
- **cuBLAS Workaround**: Due to a known bug in early Blackwell builds affecting batched GEMM, GP fitting is currently restricted to the CPU while VAE inference runs on the GPU.

### Coordinate Systems
- **FreeCAD**: Standard ISO coordinates.
- **Voxel Grid**: 64³ resolution, normalized to a `[0, 1]` unit cube.
- **Orientation**: The "Bridge" geometry is aligned with the X-axis as the primary span.

## 4. Development Workflow

1. **Config**: Always modify `pipeline_config.json` rather than hardcoding values.
2. **Type Safety**: Use Pydantic models in `schema.py` for all data passing.
3. **Validation**: Run `pytest tests/` before committing.
4. **Mandate**: NEVER commit synthetic or non-physics-grounded data to the repository.
