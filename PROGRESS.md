# Pipeline Progress

## 2026-03-07 — GPU Optimisation + Full BO Run

- **Status**: Complete
- **Result**: 400 evaluations over 50 rounds, 10 Pareto designs. Best: stress=0.015 MPa proxy, mass=0.242 kg.
- **Notes**:
  - Replaced `qExpectedHypervolumeImprovement` (multi-output GP + MC, 20+ min/round on CPU in 32D) with `qUpperConfidenceBound` + scalarized objective (seconds/round, ~80x speedup).
  - Fixed DenseGPUSolver thread-safety: replaced numpy dense solve with cuSPARSE PCG (`torch.mv` on CSR, avoids cuBLAS entirely).
  - Implemented real GPU FEM evaluator (`gpu_fem_solver.py`) using SIMPSolverGPU infrastructure with penal=1.0. Stress proxy = compliance/n_solid * 10.
  - Vectorised `_build_filter` and `_build_edof_mapping` in `simp_solver_gpu.py` (neighbour-offset broadcast, no Python loops).
  - GPU FEM throughput: ~8 evaluations in ~10s per batch vs 8s each with CalculiX.
  - VAE: 300 epochs, checkpoint at `checkpoints/vae_best.pth`.
  - Dataset: 461 samples at 64^3 from `fem_data_all/`.
  - BO output saved to `optimization_results/bo_checkpoint.json` (400 samples, `x_history` + `y_history`).

## 2026-03-01 — Post-Refactor Validation

- **Status**: Complete
- **Result**: 58 tests passing, VAE val IoU=0.8813
- **Notes**: Pipeline refactored for resolution-independence, sentinel-based FEM failure handling, CCX/FreeCAD auto-discovery.
