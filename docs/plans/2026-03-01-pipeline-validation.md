# Pipeline Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate every critical component of the generative design pipeline with a proper test suite, then run an end-to-end optimisation to produce a verified Pareto front of mechanical designs.

**Architecture:** Four layers of validation — unit tests (pure functions, no FreeCAD/GPU), integration tests (VAE forward pass on GPU), end-to-end smoke test (BO with 20 evaluations), and visual inspection (STL exports + latent PCA). Broken stale tests are removed first. All new tests live in `tests/` and run with `pytest`.

**Tech Stack:** pytest, numpy, torch, trimesh (mesh export), voxel_fem.VoxelHexMesher, freecad_bridge utilities, vae_design_model.DesignVAE, optimization_engine.DesignOptimizer

---

## Prerequisites

Before starting: confirm background tasks have completed.

```bash
# Check retrain finished
grep "RETRAIN COMPLETE\|Epoch 499" /tmp/retrain.log | tail -1

# Check variant generation finished
grep "All variant generation complete" /tmp/gen_variants.log

# Confirm checkpoint exists and is recent
ls -lh checkpoints/vae_best.pth
```

Expected: retrain log shows epoch 499 done, `vae_best.pth` modified after the rebuild.

If retrain hasn't finished yet, start Task 1 (stale test cleanup) while waiting.

---

### Task 1: Remove Stale Tests

The existing tests import modules/symbols that no longer exist (`topology.mesh_export`, `sim_config.load_config`, `SIMP`, etc.). They block `pytest` collection.

**Files:**
- Delete: `tests/test_sim_config.py`
- Delete: `tests/test_simp_solver.py`
- Delete: `tests/test_topology_solver.py`
- Delete: `tests/test_mesh_export.py`
- Keep: `tests/__init__.py`

**Step 1: Delete stale test files**

```bash
rm tests/test_sim_config.py tests/test_simp_solver.py \
   tests/test_topology_solver.py tests/test_mesh_export.py
```

**Step 2: Verify pytest collects zero tests (no errors)**

```bash
source venv/bin/activate
python -m pytest tests/ --collect-only -q 2>&1
```

Expected output:
```
no tests ran
```

**Step 3: Commit**

```bash
git add -u tests/
git commit -m "test: remove stale tests (import modules no longer exist)"
```

---

### Task 2: Unit Tests — `voxel_fem.py` Pure Functions

These tests cover `_wsl_to_win()` and `VoxelHexMesher.voxels_to_inp()` — no FreeCAD, no GPU, no ccx required.

**Files:**
- Create: `tests/test_voxel_fem.py`

**Step 1: Write the tests**

```python
# tests/test_voxel_fem.py
import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from voxel_fem import _wsl_to_win, VoxelHexMesher


class TestWslToWin:
    def test_mnt_c_path(self):
        assert _wsl_to_win("/mnt/c/Windows/Temp/foo.inp") == "C:\\Windows\\Temp\\foo.inp"

    def test_mnt_d_path(self):
        assert _wsl_to_win("/mnt/d/data/file.txt") == "D:\\data\\file.txt"

    def test_non_mnt_path_unchanged(self):
        assert _wsl_to_win("/home/user/file.inp") == "/home/user/file.inp"

    def test_drive_root_only(self):
        assert _wsl_to_win("/mnt/c") == "C:\\"


class TestVoxelHexMesher:
    def _solid_cube(self, n=4):
        """Return a fully solid n³ binary voxel grid."""
        return np.ones((n, n, n), dtype=np.float32)

    def test_empty_voxels_raises(self, tmp_path):
        vox = np.zeros((4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="void"):
            VoxelHexMesher.voxels_to_inp(vox, output_path=str(tmp_path / "out.inp"))

    def test_inp_file_created(self, tmp_path):
        vox = self._solid_cube(4)
        out = str(tmp_path / "cube.inp")
        VoxelHexMesher.voxels_to_inp(vox, output_path=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 500

    def test_inp_contains_node_section(self, tmp_path):
        vox = self._solid_cube(4)
        out = str(tmp_path / "cube.inp")
        VoxelHexMesher.voxels_to_inp(vox, output_path=out)
        content = Path(out).read_text()
        assert "*NODE" in content
        assert "*ELEMENT" in content

    def test_inp_contains_material(self, tmp_path):
        vox = self._solid_cube(4)
        out = str(tmp_path / "cube.inp")
        VoxelHexMesher.voxels_to_inp(vox, E_mpa=200000.0, poisson=0.28, output_path=out)
        content = Path(out).read_text()
        assert "200000" in content
        assert "0.28" in content

    def test_boundary_conditions_present(self, tmp_path):
        vox = self._solid_cube(6)
        out = str(tmp_path / "bc.inp")
        VoxelHexMesher.voxels_to_inp(
            vox, fixed_face="x_min", load_face="x_max",
            force_n=500.0, output_path=out
        )
        content = Path(out).read_text()
        assert "*BOUNDARY" in content
        assert "*CLOAD" in content

    def test_node_count_matches_solid_voxels(self, tmp_path):
        # A 2×2×2 solid cube has 3³=27 unique corner nodes
        vox = self._solid_cube(2)
        out = str(tmp_path / "small.inp")
        VoxelHexMesher.voxels_to_inp(vox, output_path=out)
        content = Path(out).read_text()
        node_lines = [l for l in content.splitlines()
                      if l and l[0].isdigit() and "*" not in l]
        # Count lines in *NODE section (stop at *ELEMENT)
        in_node = False
        node_count = 0
        for line in content.splitlines():
            if "*NODE" in line:
                in_node = True; continue
            if in_node and line.startswith("*"):
                break
            if in_node and line.strip():
                node_count += 1
        assert node_count == 27  # (2+1)³

    def test_partial_solid_smaller_node_count(self, tmp_path):
        vox = np.zeros((4, 4, 4), dtype=np.float32)
        vox[0, 0, 0] = 1.0  # single solid voxel → 8 nodes
        out = str(tmp_path / "single.inp")
        VoxelHexMesher.voxels_to_inp(vox, output_path=out)
        content = Path(out).read_text()
        in_node = False
        node_count = 0
        for line in content.splitlines():
            if "*NODE" in line:
                in_node = True; continue
            if in_node and line.startswith("*"):
                break
            if in_node and line.strip():
                node_count += 1
        assert node_count == 8
```

**Step 2: Run tests — expect all to pass (no GPU/FreeCAD needed)**

```bash
source venv/bin/activate
python -m pytest tests/test_voxel_fem.py -v
```

Expected:
```
tests/test_voxel_fem.py::TestWslToWin::test_mnt_c_path PASSED
tests/test_voxel_fem.py::TestWslToWin::test_mnt_d_path PASSED
...
tests/test_voxel_fem.py::TestVoxelHexMesher::test_partial_solid_smaller_node_count PASSED
9 passed in 0.3s
```

**Step 3: Commit**

```bash
git add tests/test_voxel_fem.py
git commit -m "test: add VoxelHexMesher unit tests (no FreeCAD/GPU)"
```

---

### Task 3: Unit Tests — `freecad_workbench/export_pipeline.py` Pure Functions

Test the config export logic without needing a live FreeCAD document — mock the document objects.

**Files:**
- Create: `tests/test_export_pipeline.py`

**Step 1: Write the tests**

```python
# tests/test_export_pipeline.py
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "freecad_workbench"))

# export_pipeline imports FreeCAD at module level — mock it before import
import unittest.mock as mock
sys.modules["FreeCAD"] = mock.MagicMock()

from export_pipeline import (
    collect_constraints,
    collect_loads,
    find_seed_part,
    _derive_bc_from_constraints,
)


def _make_constraint(ctype="fixed", label="Fixed_1", refs=None):
    obj = SimpleNamespace(
        Name="Constraint_fixed",
        ConstraintType=ctype,
        Label=label,
        References=refs or [],
    )
    return obj


def _make_load(ltype="force", magnitude=1000.0, direction=(0, 0, -1), label="Load_1", refs=None):
    d = SimpleNamespace(x=direction[0], y=direction[1], z=direction[2])
    obj = SimpleNamespace(
        Name="Load_force",
        LoadType=ltype,
        Magnitude=magnitude,
        Direction=d,
        Label=label,
        References=refs or [],
    )
    return obj


def _make_doc(*objects):
    return SimpleNamespace(Objects=list(objects), FileName="")


class TestCollectConstraints:
    def test_empty_doc(self):
        assert collect_constraints(_make_doc()) == []

    def test_single_fixed_no_refs(self):
        obj = _make_constraint("fixed", "MyFixed")
        result = collect_constraints(_make_doc(obj))
        assert len(result) == 1
        assert result[0]["type"] == "fixed"
        assert result[0]["label"] == "MyFixed"
        assert result[0]["faces"] == []

    def test_ignores_non_constraint_objects(self):
        body = SimpleNamespace(Name="Body", Label="Body")
        obj = _make_constraint()
        result = collect_constraints(_make_doc(body, obj))
        assert len(result) == 1

    def test_multiple_constraints(self):
        c1 = _make_constraint("fixed",    "C1")
        c2 = _make_constraint("symmetry", "C2")
        result = collect_constraints(_make_doc(c1, c2))
        assert len(result) == 2
        types = {r["type"] for r in result}
        assert types == {"fixed", "symmetry"}


class TestCollectLoads:
    def test_empty_doc(self):
        assert collect_loads(_make_doc()) == []

    def test_single_force(self):
        obj = _make_load("force", 500.0, (0, 0, -1), "MyForce")
        result = collect_loads(_make_doc(obj))
        assert len(result) == 1
        assert result[0]["magnitude"] == 500.0
        assert result[0]["direction"] == [0, 0, -1]

    def test_ignores_non_load_objects(self):
        body = SimpleNamespace(Name="Body", Label="Body")
        obj = _make_load()
        result = collect_loads(_make_doc(body, obj))
        assert len(result) == 1


class TestFindSeedPart:
    def test_no_seed_returns_none(self):
        assert find_seed_part(_make_doc()) is None

    def test_finds_seed_part(self):
        seed = SimpleNamespace(Name="SeedPart_cantilever", GeometryType="cantilever")
        result = find_seed_part(_make_doc(seed))
        assert result is seed

    def test_ignores_non_seed(self):
        body = SimpleNamespace(Name="Body")
        assert find_seed_part(_make_doc(body)) is None


class TestDeriveBCFromConstraints:
    def test_force_direction_extracted(self):
        constraints = [{"type": "fixed", "faces": []}]
        loads = [{"type": "force", "direction": [1, 0, 0], "magnitude": 2000.0}]
        _, _, force_dir, force_n = _derive_bc_from_constraints(constraints, loads)
        assert force_dir == [1, 0, 0]
        assert force_n == 2000.0

    def test_defaults_when_no_loads(self):
        _, _, force_dir, force_n = _derive_bc_from_constraints([], [])
        assert force_dir == [0, 0, -1]
        assert force_n == 1000.0
```

**Step 2: Run the tests**

```bash
source venv/bin/activate
python -m pytest tests/test_export_pipeline.py -v
```

Expected: 12 tests, all PASSED.

**Step 3: Commit**

```bash
git add tests/test_export_pipeline.py
git commit -m "test: add export_pipeline unit tests with mocked FreeCAD doc"
```

---

### Task 4: Unit Tests — `rebuild_dataset.py` Pair Scanner

Test `load_pairs()` with a synthetic temp directory of JSON+STL fixtures.

**Files:**
- Create: `tests/test_rebuild_dataset.py`

**Step 1: Write the tests**

```python
# tests/test_rebuild_dataset.py
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from rebuild_dataset import load_pairs


def _write_pair(d: Path, stem: str, stress: float = 100.0):
    (d / f"{stem}_mesh.stl").write_text("solid empty\nendsolid\n")
    (d / f"{stem}_fem_results.json").write_text(json.dumps({
        "stress_max": stress,
        "stress_mean": stress * 0.3,
        "compliance": 10.0,
        "mass": 0.5,
        "parameters": {"h_mm": 10.0, "r_mm": 2.0},
    }))


class TestLoadPairs:
    def test_empty_dir(self, tmp_path):
        assert load_pairs(tmp_path) == []

    def test_single_valid_pair(self, tmp_path):
        _write_pair(tmp_path, "cant_h10p0_r2p0")
        pairs = load_pairs(tmp_path)
        assert len(pairs) == 1
        stl, d, stem = pairs[0]
        assert stl.name == "cant_h10p0_r2p0_mesh.stl"
        assert d["stress_max"] == 100.0

    def test_skips_json_without_stl(self, tmp_path):
        (tmp_path / "cant_h10p0_r2p0_fem_results.json").write_text(
            json.dumps({"stress_max": 50.0, "compliance": 5.0, "mass": 0.3,
                        "parameters": {}})
        )
        assert load_pairs(tmp_path) == []

    def test_skips_zero_stress_zero_compliance(self, tmp_path):
        _write_pair(tmp_path, "bad_run", stress=0.0)
        # Overwrite compliance to 0 too
        (tmp_path / "bad_run_fem_results.json").write_text(
            json.dumps({"stress_max": 0.0, "compliance": 0.0, "mass": 0.1,
                        "parameters": {}})
        )
        assert load_pairs(tmp_path) == []

    def test_multiple_geometries(self, tmp_path):
        _write_pair(tmp_path, "cant_h10p0_r2p0")
        _write_pair(tmp_path, "lbra_h15p0_r5p0")
        _write_pair(tmp_path, "tape_h12p0_r3p0")
        pairs = load_pairs(tmp_path)
        assert len(pairs) == 3
        stems = {p[2] for p in pairs}
        assert "cant_h10p0_r2p0" in stems
        assert "lbra_h15p0_r5p0" in stems

    def test_geometry_count_reported(self, tmp_path, capsys):
        _write_pair(tmp_path, "cant_h10p0_r2p0")
        _write_pair(tmp_path, "cant_h12p0_r1p0")
        _write_pair(tmp_path, "lbra_h15p0_r5p0")
        load_pairs(tmp_path)
        captured = capsys.readouterr()
        assert "cant=2" in captured.out or "cant=2" in captured.err
```

**Step 2: Run the tests**

```bash
source venv/bin/activate
python -m pytest tests/test_rebuild_dataset.py -v
```

Expected: 6 tests, all PASSED.

**Step 3: Commit**

```bash
git add tests/test_rebuild_dataset.py
git commit -m "test: add rebuild_dataset load_pairs unit tests"
```

---

### Task 5: Integration Tests — VAE Forward Pass (GPU)

Verify the model architecture, dtype handling, and encode/decode roundtrip on GPU. Requires CUDA but no dataset or FreeCAD.

**Files:**
- Create: `tests/test_vae_model.py`

**Step 1: Write the tests**

```python
# tests/test_vae_model.py
import sys
from pathlib import Path
import pytest
import torch
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


@pytest.fixture(scope="module")
def vae():
    from vae_design_model import DesignVAE
    model = DesignVAE(input_shape=(64, 64, 64), latent_dim=32).cuda()
    model.eval()
    return model


class TestDesignVAEShapes:
    def test_param_count(self, vae):
        n = sum(p.numel() for p in vae.parameters())
        assert 30_000_000 < n < 60_000_000, f"Unexpected param count: {n}"

    def test_forward_output_shapes(self, vae):
        x = torch.randn(2, 1, 64, 64, 64, device="cuda")
        with autocast("cuda", dtype=torch.bfloat16):
            x_rec, mu, logvar, perf, params = vae(x)
        assert x_rec.shape  == (2, 1, 64, 64, 64)
        assert mu.shape     == (2, 32)
        assert logvar.shape == (2, 32)
        assert perf.shape   == (2, 3)
        assert params.shape == (2, 2)

    def test_encode_returns_cpu_float32(self, vae):
        x = torch.randn(4, 1, 64, 64, 64, device="cuda")
        with torch.no_grad():
            with autocast("cuda", dtype=torch.bfloat16):
                mu, logvar = vae.encode(x)
        # FC layers run in FP32 and stay on GPU
        assert mu.dtype == torch.float32
        assert mu.device.type == "cuda"

    def test_decode_output_range(self, vae):
        z = torch.randn(2, 32, device="cuda")
        with torch.no_grad():
            voxels = vae.decode(z)
        assert voxels.shape == (2, 1, 64, 64, 64)
        assert voxels.min() >= 0.0
        assert voxels.max() <= 1.0

    def test_backward_does_not_crash(self, vae):
        """Regression: BF16 GEMM backward was crashing on Blackwell."""
        vae.train()
        x = torch.randn(2, 1, 64, 64, 64, device="cuda")
        with autocast("cuda", dtype=torch.bfloat16):
            x_rec, mu, logvar, _, _ = vae(x)
            loss = x_rec.mean() + mu.mean()
        loss.backward()
        vae.eval()

    def test_predict_parameters_shape(self, vae):
        z = torch.randn(8, 32, device="cuda")
        with torch.no_grad():
            p = vae.predict_parameters(z)
        assert p.shape == (8, 2)


class TestDesignVAECheckpoint:
    def test_checkpoint_loads(self, tmp_path):
        from vae_design_model import DesignVAE
        model = DesignVAE(input_shape=(64, 64, 64), latent_dim=32).cuda()
        ckpt_path = tmp_path / "test.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "latent_dim": 32,
            "input_shape": (64, 64, 64),
        }, ckpt_path)
        loaded = DesignVAE(input_shape=(64, 64, 64), latent_dim=32).cuda()
        state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        loaded.load_state_dict(state["model_state_dict"])
        assert state["latent_dim"] == 32

    def test_production_checkpoint_loads(self):
        """Load the real vae_best.pth — fails fast if architecture drifted."""
        ckpt_path = Path("checkpoints/vae_best.pth")
        if not ckpt_path.exists():
            pytest.skip("checkpoints/vae_best.pth not found")
        from vae_design_model import DesignVAE
        state = torch.load(str(ckpt_path), map_location="cuda", weights_only=False)
        model = DesignVAE(
            input_shape=state.get("input_shape", (64, 64, 64)),
            latent_dim=state.get("latent_dim", 32),
        ).cuda()
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        z = torch.randn(1, 32, device="cuda")
        with torch.no_grad():
            vox = model.decode(z)
        assert vox.shape == (1, 1, 64, 64, 64)
```

**Step 2: Run the tests**

```bash
source venv/bin/activate
python -m pytest tests/test_vae_model.py -v
```

Expected: 8 tests, all PASSED. The `test_backward_does_not_crash` is the key regression test for the Blackwell BF16 fix.

**Step 3: Commit**

```bash
git add tests/test_vae_model.py
git commit -m "test: add VAE GPU integration tests incl Blackwell BF16 regression"
```

---

### Task 6: Run Full Test Suite

Confirm all tasks 1-5 pass together cleanly.

**Step 1: Run all tests**

```bash
source venv/bin/activate
python -m pytest tests/ -v --tb=short 2>&1
```

Expected:
```
tests/test_voxel_fem.py          9 passed
tests/test_export_pipeline.py   12 passed
tests/test_rebuild_dataset.py    6 passed
tests/test_vae_model.py          8 passed
======= 35 passed in Xs =======
```

**Step 2: Commit if not already clean**

```bash
git add tests/
git commit -m "test: full suite passes (35 tests, 0 failures)"
```

---

### Task 7: Post-Retrain Evaluation

Run `eval_vae.py` on the production checkpoint to get reconstruction metrics and sample STLs.

**Prerequisite:** retrain must be complete (`grep "RETRAIN COMPLETE" /tmp/retrain.log`).

**Step 1: Run evaluation**

```bash
source venv/bin/activate
python eval_vae.py \
  --checkpoint checkpoints/vae_best.pth \
  --fem-data ./fem_data \
  --output-dir eval_results \
  --n-samples 8 2>&1 | tee /tmp/eval_vae.log
```

Expected output (approximately):
```
Loaded checkpoint: latent_dim=32  input=(64, 64, 64)
Evaluating reconstruction quality ...
  Val IoU:  0.72      ← target: > 0.65
  Val BCE:  0.042     ← target: < 0.10
Collecting latent vectors ...
  Latent shape: (N, 32)  mean_norm: ~4.0
PCA plot saved to eval_results/latent_pca.png
Exported sample_00.stl  (N faces)
...
Report saved to eval_results/eval_report.json
```

**Step 2: Inspect the STL samples**

The 8 random samples in `eval_results/samples/` should show geometrically plausible shapes. Copy to Windows to view:

```bash
# WSL path to Windows explorer path
ls eval_results/samples/*.stl
# Open \\wsl.localhost\Ubuntu\home\genpipeline\eval_results\samples\ in Explorer
```

**Step 3: Fail criteria** — if IoU < 0.5, the model has not learned geometry well. Re-examine training loss curves in TensorBoard (`tensorboard --logdir ./logs`) before proceeding to BO.

**Step 4: Commit the report**

```bash
git add eval_results/eval_report.json
git commit -m "eval: post-retrain VAE evaluation (IoU=X.XX, BCE=X.XXX)"
```

---

### Task 8: BO Smoke Test — 20 Evaluations, Cantilever

Verify the full optimisation loop runs without error and produces a non-trivial Pareto front.

**Step 1: Run the smoke test**

```bash
source venv/bin/activate
python optimization_engine.py \
  --model-checkpoint checkpoints/vae_best.pth \
  --n-iter 5 \
  --q 4 \
  --output-dir ./optimization_results/smoke_test \
  2>&1 | tee /tmp/bo_smoke.log
```

Expected log lines:
```
Starting Multi-Objective Parallel Discovery (q=4)...
Round 1/5 complete. Pareto Designs Found: N
Round 2/5 complete. ...
MOBO Results Saved. N Pareto-optimal designs identified.
```

**Step 2: Inspect results**

```bash
python3 -c "
import json
with open('optimization_results/smoke_test/optimization_history.json') as f:
    h = json.load(f)
pf = h['pareto_front']
print(f'Pareto designs: {len(pf)}')
for d in pf[:3]:
    print(f'  stress={d[\"stress\"]:.1f} MPa  mass={d[\"mass\"]:.3f} kg')
"
```

Expected: at least 2 Pareto designs, stress values 50–500 MPa range, mass 0.05–1.0 kg range. If stress=1e6 for all entries, FEM evaluations failed — check FreeCAD is reachable (`freecad_bridge.find_freecad_cmd()`).

**Step 3: Commit results summary**

```bash
git add optimization_results/smoke_test/optimization_history.json
git commit -m "eval: BO smoke test (5 rounds, N Pareto designs)"
```

---

### Task 9: Voxel FEM Integration Test

Run a short BO loop using `--voxel-fem` to verify the direct CalculiX path works end-to-end with a real decoded voxel.

**Step 1: Verify ccx is accessible**

```bash
source venv/bin/activate
python3 -c "from voxel_fem import find_ccx; print(find_ccx())"
```

Expected: prints path to `ccx.exe`. If `None`, FreeCAD is not installed at the expected path — check `CCX_SEARCH_PATHS` in `voxel_fem.py`.

**Step 2: Single voxel FEM evaluation**

```bash
python3 -c "
import torch, sys
sys.path.insert(0, '.')
from vae_design_model import DesignVAE
from voxel_fem import VoxelFEMEvaluator

vae = DesignVAE(input_shape=(64,64,64), latent_dim=32).cuda()
import torch as t
state = t.load('checkpoints/vae_best.pth', map_location='cuda', weights_only=False)
vae.load_state_dict(state['model_state_dict'])
vae.eval()

evaluator = VoxelFEMEvaluator()
z = torch.zeros(32, device='cuda')  # mean latent
result = evaluator.evaluate(z, vae, bbox=None)
print(result)
"
```

Expected: `{'stress': float > 0, 'compliance': float > 0, 'mass': float > 0}`. Values should be physically plausible (stress 50–2000 MPa for typical geometry under 1000 N load).

**Step 3: BO smoke test with --voxel-fem**

```bash
python optimization_engine.py \
  --model-checkpoint checkpoints/vae_best.pth \
  --n-iter 3 \
  --q 2 \
  --voxel-fem \
  --output-dir ./optimization_results/voxel_fem_test \
  2>&1 | tee /tmp/voxel_fem_bo.log
```

Expected: 6 evaluations complete, results saved. Stress values should be non-zero.

**Step 4: Commit**

```bash
git commit --allow-empty -m "eval: voxel FEM BO smoke test (3 rounds, direct CalculiX)"
```

---

### Task 10: Push and Final Status Check

**Step 1: Push all commits**

```bash
git push origin master
```

**Step 2: Final test run to confirm nothing regressed**

```bash
source venv/bin/activate
python -m pytest tests/ -q 2>&1
```

Expected: `35 passed`.

**Step 3: Report system state**

```bash
python3 -c "
from pathlib import Path
import json, torch

# Checkpoint
ckpt = torch.load('checkpoints/vae_best.pth', map_location='cpu', weights_only=False)
print(f'Checkpoint: latent_dim={ckpt[\"latent_dim\"]}  input={ckpt.get(\"input_shape\")}')

# Dataset
ds = torch.load('fem_data/fem_dataset.pt', map_location='cpu', weights_only=False)
print(f'Dataset: {len(ds[\"samples\"])} samples')

# Eval report
if Path('eval_results/eval_report.json').exists():
    r = json.load(open('eval_results/eval_report.json'))
    print(f'VAE IoU: {r[\"val_iou\"]:.3f}  BCE: {r[\"val_bce\"]:.3f}')

# BO results
if Path('optimization_results/smoke_test/optimization_history.json').exists():
    h = json.load(open('optimization_results/smoke_test/optimization_history.json'))
    print(f'Pareto designs: {len(h[\"pareto_front\"])}')
"
```

---

## Monitoring Background Tasks

```bash
# Variant generation (50 tapered + 50 ribbed remaining)
tail -f /tmp/gen_variants.log

# Retrain (fires automatically after generation finishes)
tail -f /tmp/retrain.log

# Current variant counts
python3 -c "
from pathlib import Path
d = Path('fem_data')
for g in ['cant','lbra','tape','ribb']:
    print(g, len(list(d.glob(f'{g}_*_fem_results.json'))))
"
```
