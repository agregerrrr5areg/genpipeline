# Pipeline Progress Log

Dated record of what has actually been run and what the results were.
Update this file whenever a pipeline stage completes.

---

### FEM Data Generation
- **Status**: Complete (237-450 variants)
- **Tool**: FreeCAD 1.0 headless via WSL2 bridge with Windows FreeCAD
- **Variants**: cantilever (h: 5-18 mm, r: 0-5 mm), ribbed plate, tapered beam
- **Stress range**: 68-640 MPa (C3D4 linear tets, ~50-70% of analytical -- expected)
- **Duration**: ~1.4 s/variant
- **Files**: `genpipeline/fem/data/*_fem_results.json`, `*_mesh.stl`

### Dataset
- **32³ dataset**: `genpipeline/fem/data/fem_dataset.pt` (2.8 MB, faster training)
- **64³ dataset**: `genpipeline/fem/data/fem_dataset_res64.pt` (53 MB, full resolution)
- **Samples**: ~237-450 (marginal for 37.7M parameter model -- SIMP augmentation pending)
- **Format**: `{'train_loader': DataLoader, 'val_loader': DataLoader}`, batches contain `geometry (B,1,64,64,64)`, `performance (B,3)`, `parameters (B,2)`

### VAE Training
- **Status**: 300 epochs complete, best checkpoint saved
- **Checkpoint**: `checkpoints/vae_best.pth` (144 MB)
- **Epoch snapshots**: `checkpoints/vae_epoch_*.pth` (every 10 epochs, 0-300)
- **Final train loss**: ~0.103
- **Architecture**: DesignVAE, latent_dim=32, input_shape=(64,64,64)
- **Config used**: `pipeline_config.json` (beta_vae=1.0, pos_weight=30.0, batch_size=128)
- **Hardware**: RTX 5080 (Blackwell sm_120), BF16 mixed precision, CUDA 12.8

### Bayesian Optimisation
- **Status**: 20+ iterations completed
- **Best objective**: -0.1058 (stress × mass proxy)
- **Best occupancy**: 16.2%
- **Geometry**: bridge/cantilever family
- **Results**: `optimization_results/bridge_run/`, `optimization_results/bridge_final/`
- **Note**: `real_run.json` shows `best_voxel_shape: [32, 32, 32]` -- this is a legacy result from before the 64³ migration.

### Integration Tests
- **Status**: Complete
- **File**: `tests/test_integration_decode_fem.py`
- **Result**: 6/6 passed (4 decode tests + 2 FEM tests with ccx)

### Known Blockers
- **ccx on WSL2**: VoxelFEMEvaluator discovers ccx via glob of Windows FreeCAD installs at `/mnt/c/Users/*/AppData/Local/Programs/FreeCAD*/bin/ccx.exe`. If that path changes, evaluations silently return sentinel (1e6).
- **Blackwell cuBLAS**: `cublasDgemmStridedBatched` broken for batch≥2 on CUDA 12.8. BoTorch GP models must stay on CPU via `blackwell_compat.py`.
- **Data scarcity**: 237-450 samples is marginal for 37.7M parameters. SIMP augmentation (`genpipeline/topology/topo_data_gen.py`) exists but has not been used to augment the FreeCAD data yet.
- **beta_vae mismatch**: Config has `beta_vae=1.0`; design doc recommends `0.05` for better reconstruction at 64³ with limited data. Ablation pending.