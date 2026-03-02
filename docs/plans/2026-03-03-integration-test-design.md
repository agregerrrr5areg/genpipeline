# Integration Test: Decode → FEM Path

**Date**: 2026-03-03
**Status**: Approved, pending implementation

## Problem

There is no automated way to verify the core decode→FEM path without running the full BO loop manually. A single silent failure in `VoxelFEMEvaluator` returns sentinel values (`1e6`) that look like valid data to the BO loop, causing it to silently degrade.

## Goal

A single `pytest` run that gives a green/red signal in under 2 minutes: loads the real checkpoint, decodes 4 latent vectors, optionally runs FEM, asserts correctness.

## Design

### New file: `tests/test_integration_decode_fem.py`

**Helpers**
- `ccx_available() → bool`: probes `PATH` for `ccx` binary, then globs FreeCAD Windows installs at known WSL path. Returns `True` if any ccx found.
- `load_vae_checkpoint() → DesignVAE`: constructs `DesignVAE(latent_dim=32, input_shape=(64,64,64))`, loads `checkpoints/vae_best.pth`, sets `eval()` mode.

**Class `TestDecodeAlwaysRuns`** (no external dependencies, always runs)
- `test_checkpoint_loads`: asserts `model_state_dict`, `latent_dim`, `input_shape` keys present; `latent_dim == 32`.
- `test_decode_returns_correct_shape`: 4 × `torch.randn(32)` with seed 42 → VAE decode → shape `(4, 1, 64, 64, 64)`.
- `test_voxels_are_non_trivial`: each decoded voxel sigmoid output has occupancy between 5% and 95% (not all-zero, not saturated).

**Class `TestFEMWithCCX`** (`@pytest.mark.skipif(not ccx_available(), reason="ccx not found")`)
- `test_evaluate_batch_runs`: same 4 latent vectors → `VoxelFEMEvaluator.evaluate_batch()` → list of 4 result dicts with keys `stress`, `compliance`, `mass`.
- `test_at_least_one_non_sentinel`: `assert any(r["stress"] < 1e6 for r in results)`.

Both test classes use `torch.manual_seed(42)` for the same 4 latent vectors so results are reproducible and comparable across runs.

## Informative File Updates

In the same implementation pass, update the following files to reflect actual current state:

| File | Changes |
|------|---------|
| `README.md` | Update "Status" section: 64³ resolution, 300-epoch checkpoint, BO 20+ iterations |
| `CLAUDE.md` | Add rule: after any pipeline stage completes, update `PROGRESS.md` with date and outcome |
| `DEVELOPMENT.md` | Add Blackwell/ccx known issues section clearly at top |
| `PROGRESS.md` (new) | Dated log: FEM variants, VAE training, BO runs, known blockers |

## Success Criteria

1. `pytest tests/test_integration_decode_fem.py::TestDecodeAlwaysRuns` passes with no external dependencies.
2. On a machine with ccx installed, `TestFEMWithCCX` passes with at least one non-sentinel FEM result.
3. On a machine without ccx, `TestFEMWithCCX` is skipped cleanly (not failed).
4. All informative files accurately describe the current pipeline state as of 2026-03-03.
