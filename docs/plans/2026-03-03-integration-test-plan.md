# Integration Test + Informative File Updates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write `tests/test_integration_decode_fem.py` with a decode-only class (always runs) and a FEM class (skips if ccx absent), then update all informative files to accurately describe the current pipeline state.

**Architecture:** Two test classes in a single file share a module-level VAE loader helper. Sentinel detection uses the existing `FEM_SENTINEL = 1e6` constant from `genpipeline.pipeline_utils`. Informative files are updated in their own tasks with no code changes.

**Tech Stack:** pytest, torch, numpy, `genpipeline.vae_design_model.DesignVAE`, `genpipeline.fem.voxel_fem.VoxelFEMEvaluator`, `genpipeline.pipeline_utils.FEM_SENTINEL`

---

### Task 1: Write the decode-only tests (failing first)

**Files:**
- Create: `tests/test_integration_decode_fem.py`

**Step 1: Write the failing test file**

```python
"""
Integration tests for the core decode→FEM path.

TestDecodeAlwaysRuns — no external dependencies, always runs.
TestFEMWithCCX       — skipped unless ccx is discoverable.
"""
import pytest
import torch
import numpy as np
from pathlib import Path

# ── helpers ───────────────────────────────────────────────────────────────────

CHECKPOINT = Path(__file__).parent.parent / "checkpoints" / "vae_best.pth"
LATENT_DIM = 32
INPUT_SHAPE = (64, 64, 64)


def _load_vae():
    """Load DesignVAE from the best checkpoint, set eval mode."""
    from genpipeline.vae_design_model import DesignVAE
    vae = DesignVAE(input_shape=INPUT_SHAPE, latent_dim=LATENT_DIM)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()
    return vae


def _four_latent_vectors():
    """Reproducible set of 4 latent vectors (seed 42)."""
    torch.manual_seed(42)
    return torch.randn(4, LATENT_DIM)


def _ccx_available() -> bool:
    from genpipeline.fem.voxel_fem import find_ccx
    return find_ccx() is not None


# ── TestDecodeAlwaysRuns ──────────────────────────────────────────────────────

class TestDecodeAlwaysRuns:

    def test_checkpoint_exists(self):
        assert CHECKPOINT.exists(), f"Checkpoint not found: {CHECKPOINT}"

    def test_checkpoint_keys(self):
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert ckpt.get("latent_dim") == LATENT_DIM
        assert ckpt.get("input_shape") == INPUT_SHAPE

    def test_decode_returns_correct_shape(self):
        vae = _load_vae()
        zs = _four_latent_vectors()
        with torch.no_grad():
            out = vae.decode(zs)
        assert out.shape == (4, 1, 64, 64, 64), f"Unexpected shape: {out.shape}"

    def test_voxels_are_non_trivial(self):
        """Each decoded voxel must have 5%–95% occupancy (not all-zero, not saturated)."""
        vae = _load_vae()
        zs = _four_latent_vectors()
        with torch.no_grad():
            probs = torch.sigmoid(vae.decode(zs))  # (4, 1, 64, 64, 64)
        for i in range(4):
            occ = probs[i].mean().item()
            assert 0.05 <= occ <= 0.95, (
                f"z[{i}] occupancy {occ:.3f} out of expected 5%–95% range"
            )


# ── TestFEMWithCCX ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _ccx_available(), reason="CalculiX (ccx) not found on this system")
class TestFEMWithCCX:

    def test_evaluate_batch_returns_four_results(self):
        from genpipeline.fem.voxel_fem import VoxelFEMEvaluator
        vae = _load_vae()
        evaluator = VoxelFEMEvaluator(vae_model=vae)
        zs = _four_latent_vectors().numpy()

        results = [evaluator.evaluate(z, vae) for z in zs]

        assert len(results) == 4
        for r in results:
            assert "stress" in r
            assert "compliance" in r
            assert "mass" in r

    def test_at_least_one_non_sentinel(self):
        from genpipeline.fem.voxel_fem import VoxelFEMEvaluator
        from genpipeline.pipeline_utils import FEM_SENTINEL
        vae = _load_vae()
        evaluator = VoxelFEMEvaluator(vae_model=vae)
        zs = _four_latent_vectors().numpy()

        results = [evaluator.evaluate(z, vae) for z in zs]

        non_sentinel = [r for r in results if r["stress"] < FEM_SENTINEL]
        assert len(non_sentinel) >= 1, (
            f"All 4 FEM evaluations returned sentinel (1e6). "
            f"Failure reasons: {[r.get('failure_reason') for r in results]}"
        )
```

**Step 2: Run to confirm tests fail (checkpoint not yet verified)**

```bash
cd /home/genpipeline
PYTHONPATH=. venv/bin/python -m pytest tests/test_integration_decode_fem.py::TestDecodeAlwaysRuns -v
```

Expected: Tests either PASS (checkpoint exists) or fail with clear `AssertionError` / `FileNotFoundError`. If `ModuleNotFoundError`, check `PYTHONPATH=.` is set.

**Step 3: Verify no import errors**

```bash
PYTHONPATH=. venv/bin/python -c "from tests.test_integration_decode_fem import _load_vae; print('imports ok')"
```

Expected: `imports ok`

**Step 4: Run decode tests**

```bash
PYTHONPATH=. venv/bin/python -m pytest tests/test_integration_decode_fem.py::TestDecodeAlwaysRuns -v
```

Expected: 4 tests PASSED. If `test_voxels_are_non_trivial` fails with occupancy outside 5–95%, check that `vae.decode()` returns logits (pre-sigmoid) — the test applies `torch.sigmoid` explicitly.

**Step 5: Run FEM tests (will skip if no ccx)**

```bash
PYTHONPATH=. venv/bin/python -m pytest tests/test_integration_decode_fem.py::TestFEMWithCCX -v
```

Expected: Either 2 PASSED or 2 SKIPPED (`ccx not found`). Never FAILED unless ccx is present but broken.

**Step 6: Commit**

```bash
cd /home/genpipeline
git add tests/test_integration_decode_fem.py
git commit -m "test: add integration test for decode→FEM path (Task 1)"
```

---

### Task 2: Create PROGRESS.md

**Files:**
- Create: `PROGRESS.md`

**Step 1: Write the file**

```markdown
# Pipeline Progress Log

Dated record of what has actually been run and what the results were.
Update this file whenever a pipeline stage completes.

---

## 2026-03-03 — Current State

### FEM Data Generation
- **Status**: Complete (10 variants)
- **Tool**: FreeCAD 1.0 headless → CalculiX via WSL2 bridge
- **Variants**: cantilever (h: 5–18 mm, r: 0–5 mm), ribbed plate, tapered beam
- **Stress range**: 68–640 MPa (C3D4 linear tets, ~50–70% of analytical — expected)
- **Duration**: ~1.4 s/variant
- **Files**: `genpipeline/fem/data/*_fem_results.json`, `*_mesh.stl`

### Dataset
- **32³ dataset**: `genpipeline/fem/data/fem_dataset.pt` (2.8 MB, faster training)
- **64³ dataset**: `genpipeline/fem/data/fem_dataset_res64.pt` (53 MB, full resolution)
- **Samples**: ~237–450 (marginal for 37.7M parameter model — SIMP augmentation pending)
- **Format**: `{'train_loader': DataLoader, 'val_loader': DataLoader}`, batches contain `geometry (B,1,64,64,64)`, `performance (B,3)`, `parameters (B,2)`

### VAE Training
- **Status**: 300 epochs complete, best checkpoint saved
- **Checkpoint**: `checkpoints/vae_best.pth` (144 MB)
- **Epoch snapshots**: `checkpoints/vae_epoch_*.pth` (every 10 epochs, 0–300)
- **Final train loss**: ~0.103
- **Architecture**: DesignVAE, latent_dim=32, input_shape=(64,64,64)
- **Config used**: `pipeline_config.json` (beta_vae=1.0, pos_weight=30.0, batch_size=128)
- **Hardware**: RTX 5080 (Blackwell sm_120), BF16 mixed precision, CUDA 12.8

### Bayesian Optimisation
- **Status**: 20+ iterations completed
- **Best objective**: −0.1058 (stress × mass proxy)
- **Best occupancy**: 16.2%
- **Geometry**: bridge/cantilever family
- **Results**: `optimization_results/bridge_run/`, `optimization_results/bridge_final/`
- **Note**: `real_run.json` shows `best_voxel_shape: [32, 32, 32]` — this is a legacy result from before the 64³ migration.

### Known Blockers
- **ccx on WSL2**: VoxelFEMEvaluator discovers ccx via glob of Windows FreeCAD installs at `/mnt/c/Users/*/AppData/Local/Programs/FreeCAD*/bin/ccx.exe`. If that path changes, evaluations silently return sentinel (1e6).
- **Blackwell cuBLAS**: `cublasDgemmStridedBatched` broken for batch≥2 on CUDA 12.8. BoTorch GP models must stay on CPU via `blackwell_compat.py`.
- **Data scarcity**: 237–450 samples is marginal for 37.7M parameters. SIMP augmentation (`genpipeline/topology/topo_data_gen.py`) exists but has not been used to augment the FreeCAD data yet.
- **beta_vae mismatch**: Config has `beta_vae=1.0`; design doc recommends `0.05` for better reconstruction at 64³ with limited data. Ablation pending.
```

**Step 2: Verify file created**

```bash
head -5 /home/genpipeline/PROGRESS.md
```

**Step 3: Commit**

```bash
git add PROGRESS.md
git commit -m "docs: add PROGRESS.md with dated pipeline state log"
```

---

### Task 3: Update README.md status section

**Files:**
- Modify: `README.md`

**Step 1: Read current README**

The README currently has these sections: Key Features, MANDATE, Architecture, Setup, Usage, Geometry Families, Testing & Verification, Hardware Notes, License.

It has no "Current Status" section. Add one after "Key Features" and before "MANDATE".

**Step 2: Add status section**

After the `---` following the Key Features block, insert:

```markdown
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
```

**Step 3: Also fix stale references in README**

The README refers to `fem/data_pipeline.py` and `fem/voxel_fem.py` as root-level paths. They now live in `genpipeline/fem/`. Update the Architecture link table:

Change:
```markdown
- [topology/simp_solver_gpu.py](file:///home/genpipeline/topology/simp_solver_gpu.py): PyTorch-based 3D SIMP.
- [fem/](file:///home/genpipeline/fem/): Unified physics/FEM package.
  - [data_pipeline.py](file:///home/genpipeline/fem/data_pipeline.py): Consolidates JSON/Parquet results.
  - [voxel_fem.py](file:///home/genpipeline/fem/voxel_fem.py): Direct CalculiX voxel FEM solver.
  - [data/](file:///home/genpipeline/fem/data/): Main training results.
- [genpipeline/](file:///home/genpipeline/genpipeline/): Core pipeline logic package.
```

To:
```markdown
- `genpipeline/topology/simp_solver_gpu.py` — PyTorch-based 3D SIMP.
- `genpipeline/fem/` — Unified physics/FEM package.
  - `data_pipeline.py` — Consolidates JSON/Parquet results.
  - `voxel_fem.py` — Direct CalculiX voxel FEM solver.
  - `data/` — Training voxel grids, FEM metrics, dataset `.pt` files.
- `genpipeline/` — Core pipeline logic package.
```

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add current status table and fix stale paths in README"
```

---

### Task 4: Update CLAUDE.md — add PROGRESS.md rule

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add PROGRESS.md update rule**

At the end of the `## Pipeline Commands` section in CLAUDE.md, add:

```markdown

## Keeping PROGRESS.md Current

After completing any pipeline stage, append a dated entry to `PROGRESS.md`:

```markdown
## YYYY-MM-DD — <stage name>
- **Status**: Complete / Partial / Failed
- **Result**: <key metric or outcome>
- **Notes**: <anything unexpected>
```

This is the single source of truth for what has actually run. README.md summarises it; PROGRESS.md is the full record.
```

**Step 2: Fix the stale config table in CLAUDE.md**

The table currently shows `voxel_resolution: 32` and `latent_dim: 16`. The actual `pipeline_config.json` has `voxel_resolution: 64` and `latent_dim: 32`. Update:

Change:
```markdown
| `voxel_resolution` | 32 | 32³ = fast; 64³ = accurate but ~8× more VRAM |
| `latent_dim` | 16 | Dimensionality of design space |
```

To:
```markdown
| `voxel_resolution` | 64 | 32³ = fast; 64³ = accurate but ~8× more VRAM |
| `latent_dim` | 32 | Dimensionality of design space |
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add PROGRESS.md update rule and fix stale config defaults in CLAUDE.md"
```

---

### Task 5: Update DEVELOPMENT.md — add known issues section

**Files:**
- Modify: `DEVELOPMENT.md`

**Step 1: Add known issues section at the top (after the title)**

Insert after `# GenPipeline Development Guide: VAE + Bayesian Optimisation`:

```markdown

## ⚠️ Known Issues (as of 2026-03-03)

| Issue | Symptom | Workaround |
|-------|---------|------------|
| Blackwell cuBLAS batched GEMM crash | `CUBLAS_STATUS_INVALID_VALUE` on any `matmul` with `dim>2, batch≥2` on GPU | BoTorch GP stays on CPU via `blackwell_compat.py` |
| ccx discovery via glob | If Windows FreeCAD install path changes, `VoxelFEMEvaluator` silently returns sentinel `1e6` | Set `CCX_PATH` env var explicitly |
| beta_vae config mismatch | `pipeline_config.json` has `beta_vae=1.0`; design doc recommends `0.05` for 64³ with limited data | Ablation pending; current checkpoint trained at `1.0` |
| Legacy BO result | `optimization_results/real_run.json` shows `best_voxel_shape: [32, 32, 32]` | This predates 64³ migration — treat as historical only |

```

**Step 2: Commit**

```bash
git add DEVELOPMENT.md
git commit -m "docs: add known issues table to DEVELOPMENT.md"
```

---

### Task 6: Final check — run full test suite

**Step 1: Run all tests**

```bash
cd /home/genpipeline
PYTHONPATH=. venv/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: All existing tests still pass. New integration tests either PASS or SKIP (never FAIL unless ccx is present and broken).

**Step 2: Confirm integration tests specifically**

```bash
PYTHONPATH=. venv/bin/python -m pytest tests/test_integration_decode_fem.py -v
```

Expected output (no ccx):
```
PASSED tests/test_integration_decode_fem.py::TestDecodeAlwaysRuns::test_checkpoint_exists
PASSED tests/test_integration_decode_fem.py::TestDecodeAlwaysRuns::test_checkpoint_keys
PASSED tests/test_integration_decode_fem.py::TestDecodeAlwaysRuns::test_decode_returns_correct_shape
PASSED tests/test_integration_decode_fem.py::TestDecodeAlwaysRuns::test_voxels_are_non_trivial
SKIPPED tests/test_integration_decode_fem.py::TestFEMWithCCX::test_evaluate_batch_returns_four_results
SKIPPED tests/test_integration_decode_fem.py::TestFEMWithCCX::test_at_least_one_non_sentinel
```

**Step 3: Final commit if anything was missed**

```bash
git status
# Stage anything untracked and commit with message:
# "docs: finalize informative file updates for 2026-03-03 pipeline state"
```
