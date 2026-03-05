# SIMP Augmentation Design

**Date**: 2026-03-03
**Status**: Approved, pending implementation

## Problem

The VAE has 37.7M parameters but only 237–450 training samples — marginal for 64³ voxel grids. The SIMP topology generator (`topo_data_gen.py`) exists to produce physics-grounded data at scale, but has never been run to augment the FreeCAD dataset. The existing stress labels in SIMP outputs are also faked (`compliance × 0.1`), which would corrupt the performance predictor.

## Goal

Generate 1000 SIMP samples with real ccx stress labels and merge them into `fem_dataset.pt`, bringing the training set from ~450 to ~1450 samples in a single command (~17 minutes at 4 parallel workers).

## Approach

**New standalone script**: `scripts/generate_simp_data.py`

No changes to existing modules. Produces STL + JSON pairs that `rebuild_dataset.py` already knows how to consume.

## Architecture

```
scripts/generate_simp_data.py --n-samples 1000 --workers 4
│
├── PHASE 1: SIMP geometry generation (ThreadPoolExecutor, N workers)
│   FOR each sample:
│     pick random geometry: cantilever / tapered / ribbed / lbracket
│     randomise: grid dims, volfrac ∈ [0.2, 0.5], force_mag ∈ [500, 2000] N
│     SIMPSolverGPU.run() → density field [nx, ny, nz]
│     export STL via marching cubes
│     SKIP if {stem}_fem_results.json already exists  ← resume support
│
├── PHASE 2: Real FEM stress via ccx (same workers)
│   FOR each density field from Phase 1:
│     downsample to 32³
│     VoxelFEMEvaluator.evaluate(voxels) → {stress, compliance, mass}
│     IF FEM failed → write JSON with stress_max=null, fem_failed=true
│     ELSE → write JSON with real stress, compliance, mass
│     include "source": "simp" tag in every JSON
│
└── PHASE 3: Rebuild dataset
    inline rebuild_dataset logic (or subprocess call)
    merge new SIMP+FEM samples with existing FreeCAD samples
    write genpipeline/fem/data/fem_dataset.pt
    print: "Dataset: N_before → N_after samples"
```

## File Outputs

| File | Location |
|------|----------|
| Per-sample STL | `genpipeline/fem/data/{stem}_mesh.stl` |
| Per-sample JSON | `genpipeline/fem/data/{stem}_fem_results.json` |
| Updated dataset | `genpipeline/fem/data/fem_dataset.pt` |

## JSON Schema (per sample)

```json
{
  "stress_max": 142300.0,
  "compliance": 1423000.0,
  "mass": 0.047,
  "source": "simp",
  "fem_failed": false,
  "parameters": {
    "geometry": "cantilever",
    "volfrac": 0.35,
    "force_n": 1200.0,
    "nx": 32, "ny": 8, "nz": 8
  },
  "bbox": {"x": [0, 100], "y": [0, 25], "z": [0, 25]}
}
```

## CLI Interface

```bash
# Generate 1000 samples (default)
python scripts/generate_simp_data.py

# Custom run
python scripts/generate_simp_data.py --n-samples 500 --workers 4 --geometry cantilever

# Resume interrupted run (skips existing JSON files)
python scripts/generate_simp_data.py --n-samples 1000
```

## Config Options

| Flag | Default | Effect |
|------|---------|--------|
| `--n-samples` | 1000 | Total samples to generate |
| `--workers` | 4 | Parallel workers |
| `--output-dir` | `genpipeline/fem/data` | Where to write STL + JSON |
| `--geometry` | all | Filter to one geometry type |
| `--no-rebuild` | off | Skip dataset rebuild step |

## Testing

`tests/test_simp_augmentation.py` — smoke test with 3 samples on a tiny 8×4×4 grid:
- STL file created and non-empty
- JSON has `stress_max` (real, not null) and `source: "simp"`
- `fem_dataset.pt` sample count increases after rebuild

## Success Criteria

1. `python scripts/generate_simp_data.py --n-samples 1000` runs to completion
2. 1000 JSON files written with real stress values (not sentinel)
3. `fem_dataset.pt` contains 1000+ new samples merged with existing FreeCAD data
4. Smoke test passes in under 60 seconds
