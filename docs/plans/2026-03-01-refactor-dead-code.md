# Dead Code Removal & utils.py Merge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Delete 6 dead scripts, merge the live parts of `utils.py` into `pipeline_utils.py`, update 3 import sites, and delete `utils.py`.

**Architecture:** `pipeline_utils.py` becomes the single shared utility module — it already holds `NumpyEncoder`, `smooth_voxels`, and FEM sentinel constants. We append `VoxelConverter` and `ManufacturabilityConstraints` from `utils.py`, drop the dead classes (`FreeCADInterface`, `PerformanceNormalizer`, `GeometryMetrics`, `VoxelConverter.smooth_voxel_grid`), then update `eval_vae.py`, `optimization_engine.py`, and `quickstart.py` to import from `pipeline_utils`.

**Tech Stack:** Python, numpy, scipy, scikit-image (marching cubes), trimesh

---

### Task 1: Delete dead scripts

These files are not imported by any core module and are superseded by the proper test suite or by other files.

**Files:**
- Delete: `synthetic_test.py`
- Delete: `sim_config.py`
- Delete: `visualize_pareto.py`
- Delete: `live_viewer.py`
- Delete: `view_stresses.py`
- Delete: `refine_design.py`

**Step 1: Delete the files**

```bash
rm synthetic_test.py sim_config.py visualize_pareto.py live_viewer.py view_stresses.py refine_design.py
```

**Step 2: Verify no test imports them**

```bash
source venv/bin/activate
python -m pytest tests/ -q --tb=short 2>&1 | tail -5
```

Expected: `58 passed`

**Step 3: Commit**

```bash
git add -u
git commit -m "chore: delete dead scripts (synthetic_test, sim_config, visualize_pareto, live_viewer, view_stresses, refine_design)"
```

---

### Task 2: Append VoxelConverter to pipeline_utils.py

Copy the live methods from `utils.VoxelConverter` into `pipeline_utils.py`. Drop `smooth_voxel_grid` (already covered by `smooth_voxels()`).

**Files:**
- Modify: `pipeline_utils.py`

**Step 1: Append VoxelConverter class**

Add the following block to the bottom of `pipeline_utils.py`:

```python
# ── Mesh conversion ────────────────────────────────────────────────────────────

class VoxelConverter:
    """Convert between voxel grids and triangle meshes."""

    @staticmethod
    def voxel_to_mesh(voxel_grid: np.ndarray, voxel_size: float = 1.0, bbox: dict = None) -> dict:
        from skimage import measure
        import logging
        _log = logging.getLogger(__name__)

        spacing = (voxel_size, voxel_size, voxel_size)
        origin  = np.array([0.0, 0.0, 0.0])

        if bbox:
            res = voxel_grid.shape[0]
            dx = (bbox['xmax'] - bbox['xmin']) / res
            dy = (bbox['ymax'] - bbox['ymin']) / res
            dz = (bbox['zmax'] - bbox['zmin']) / res
            spacing = (dx, dy, dz)
            origin  = np.array([bbox['xmin'], bbox['ymin'], bbox['zmin']])

        try:
            verts, faces, normals, _ = measure.marching_cubes(
                voxel_grid.astype(np.float32), level=0.5, spacing=spacing
            )
            verts += origin
            return {'vertices': verts, 'faces': faces, 'normals': normals}
        except Exception as e:
            _log.error(f"Marching cubes failed: {e}")
            return None

    @staticmethod
    def mesh_to_voxel(vertices: np.ndarray, faces: np.ndarray, resolution: int = 32) -> np.ndarray:
        import logging
        _log = logging.getLogger(__name__)
        try:
            import trimesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            voxelized = mesh.voxelized(pitch=mesh.extents.max() / resolution)
            return voxelized.matrix.astype(np.float32)
        except Exception as e:
            _log.error(f"Mesh to voxel conversion failed: {e}")
            return np.zeros((resolution, resolution, resolution), dtype=np.float32)

    @staticmethod
    def threshold_voxel_grid(voxel_grid: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (voxel_grid > threshold).astype(np.float32)

    @staticmethod
    def fill_holes(voxel_grid: np.ndarray, structure=None) -> np.ndarray:
        from scipy.ndimage import binary_fill_holes
        if structure is None:
            structure = np.ones((3, 3, 3))
        return binary_fill_holes(voxel_grid > 0.5, structure=structure).astype(np.float32)

    @staticmethod
    def remove_small_components(voxel_grid: np.ndarray, min_size: int = 10) -> np.ndarray:
        from scipy.ndimage import label, sum as ndi_sum
        labeled, n = label(voxel_grid > 0.5)
        sizes = ndi_sum(voxel_grid > 0.5, labeled, range(n + 1))
        return (sizes[labeled] >= min_size).astype(np.float32)
```

**Step 2: Verify import works**

```bash
source venv/bin/activate
python -c "from pipeline_utils import VoxelConverter; print('ok')"
```

Expected: `ok`

---

### Task 3: Append ManufacturabilityConstraints to pipeline_utils.py

**Files:**
- Modify: `pipeline_utils.py`

**Step 1: Append ManufacturabilityConstraints**

Add this block to the bottom of `pipeline_utils.py` (after VoxelConverter):

```python
# ── Manufacturability constraints ──────────────────────────────────────────────

class ManufacturabilityConstraints:
    """Check and enforce printability / manufacturability rules on voxel grids."""

    def __init__(self, min_feature_size: float = 1.0, max_overhang_angle: float = 45.0, config: dict = None):
        self.config = config or {}
        mfg_cfg = self.config.get("manufacturing_constraints", {})
        self.min_feature_size   = mfg_cfg.get("min_feature_size_mm",   min_feature_size)
        self.max_overhang_angle = mfg_cfg.get("max_overhang_angle_deg", max_overhang_angle)

    def check_min_feature_size(self, voxel_grid: np.ndarray, voxel_size: float = 1.0) -> bool:
        from scipy.ndimage import binary_erosion, binary_dilation
        eroded = binary_erosion(voxel_grid > 0.5)
        if eroded.sum() == 0:
            return False
        restored = binary_dilation(eroded)
        return (restored.sum() / (voxel_grid > 0.5).sum()) > 0.7

    def check_overhang_constraint(self, voxel_grid: np.ndarray) -> bool:
        solid = voxel_grid > 0.5
        for z in range(1, solid.shape[2]):
            layer      = solid[:, :, z]
            layer_below = solid[:, :, z - 1]
            unsupported = layer & ~layer_below
            if layer.sum() > 0 and unsupported.sum() > 0.3 * layer.sum():
                return False
        return True

    def apply_constraints(self, voxel_grid: np.ndarray, voxel_size: float = 1.0) -> np.ndarray:
        import logging
        _log = logging.getLogger(__name__)
        constrained = voxel_grid.copy()
        radius_mm  = self.min_feature_size / 2.0
        min_voxels = int((4.0 / 3.0) * np.pi * (radius_mm ** 3) / (voxel_size ** 3))

        if not self.check_min_feature_size(constrained, voxel_size=voxel_size):
            _log.warning(f"Design violates minimum feature size ({self.min_feature_size}mm)")
            constrained = VoxelConverter.remove_small_components(constrained, min_size=max(min_voxels, 1))

        if not self.check_overhang_constraint(constrained):
            _log.warning(f"Design violates overhang constraint ({self.max_overhang_angle} deg)")
            constrained = self._fix_overhangs(constrained)

        return constrained

    @staticmethod
    def _fix_overhangs(voxel_grid: np.ndarray) -> np.ndarray:
        fixed = voxel_grid.copy()
        for z in range(fixed.shape[2] - 2, 0, -1):
            unsupported = fixed[:, :, z] & ~fixed[:, :, z - 1]
            fixed[:, :, z] = fixed[:, :, z] & ~unsupported
        return fixed
```

**Step 2: Verify import works**

```bash
python -c "from pipeline_utils import VoxelConverter, ManufacturabilityConstraints; print('ok')"
```

Expected: `ok`

---

### Task 4: Update import sites

Three files import from `utils`. Change them all to import from `pipeline_utils`.

**Files:**
- Modify: `eval_vae.py:87`
- Modify: `optimization_engine.py:411`
- Modify: `quickstart.py:194`

**Step 1: eval_vae.py**

Find:
```python
from utils import VoxelConverter
```
Replace with:
```python
from pipeline_utils import VoxelConverter
```

**Step 2: optimization_engine.py**

Find:
```python
from utils import VoxelConverter, ManufacturabilityConstraints
```
Replace with:
```python
from pipeline_utils import VoxelConverter, ManufacturabilityConstraints
```

**Step 3: quickstart.py**

Find:
```python
from utils import VoxelConverter, ManufacturabilityConstraints
```
Replace with:
```python
from pipeline_utils import VoxelConverter, ManufacturabilityConstraints
```

**Step 4: Verify no remaining utils imports**

```bash
grep -rn "from utils import\|import utils" *.py tests/*.py 2>/dev/null
```

Expected: no output.

---

### Task 5: Delete utils.py and run full test suite

**Step 1: Delete utils.py**

```bash
rm utils.py
```

**Step 2: Run full test suite**

```bash
source venv/bin/activate
python -m pytest tests/ -v --tb=short 2>&1 | tail -15
```

Expected: `58 passed`

**Step 3: Smoke-test the imports that previously used utils**

```bash
python -c "
from pipeline_utils import NumpyEncoder, smooth_voxels, FEM_SENTINEL, VoxelConverter, ManufacturabilityConstraints
import numpy as np
v = np.random.rand(8, 8, 8).astype(np.float32)
mesh = VoxelConverter.voxel_to_mesh(v)
print('VoxelConverter ok:', mesh is not None)
mfg = ManufacturabilityConstraints(config={})
out = mfg.apply_constraints(v)
print('ManufacturabilityConstraints ok:', out.shape == v.shape)
"
```

Expected:
```
VoxelConverter ok: True
ManufacturabilityConstraints ok: True
```

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: merge utils.py into pipeline_utils, delete dead scripts and utils.py"
```

---

## Summary of changes

| Action | Files |
|--------|-------|
| Deleted | `synthetic_test.py`, `sim_config.py`, `visualize_pareto.py`, `live_viewer.py`, `view_stresses.py`, `refine_design.py`, `utils.py` |
| Grown | `pipeline_utils.py` (+VoxelConverter, +ManufacturabilityConstraints) |
| Updated imports | `eval_vae.py`, `optimization_engine.py`, `quickstart.py` |
| Dead code dropped | `FreeCADInterface`, `PerformanceNormalizer`, `GeometryMetrics`, `VoxelConverter.smooth_voxel_grid`, `convert_windows_path_to_wsl` |
