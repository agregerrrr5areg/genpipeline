"""
pipeline_utils.py — Shared utilities for the generative design pipeline.

Provides:
  NumpyEncoder     — JSON encoder for numpy types
  smooth_voxels()  — post-decode density filter (gaussian + logistic sharpening)
  FEM_SENTINEL     — sentinel value returned by failed FEM evaluations (1e6)
  FEM_VALID_THRESHOLD — upper bound for a valid (non-failed) FEM result (1e5)
  is_valid_fem_result() — True when result looks like a real FEM output
  VoxelConverter   — voxel/mesh conversion (voxel_to_mesh, mesh_to_voxel,
                     threshold_voxel_grid, fill_holes, remove_small_components)
  ManufacturabilityConstraints — check/enforce min feature size and overhang rules
"""

import json
import numpy as np

# ── FEM sentinel constants ─────────────────────────────────────────────────────

FEM_SENTINEL: float = 1e6
FEM_VALID_THRESHOLD: float = 1e5


def is_valid_fem_result(result: dict) -> bool:
    """Return True if result contains a plausible (non-sentinel) FEM answer."""
    stress = result.get("stress", result.get("stress_max", FEM_SENTINEL))
    return 0.0 < float(stress) < FEM_VALID_THRESHOLD


# ── JSON encoder ───────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


# ── Voxel post-processing ──────────────────────────────────────────────────────

def smooth_voxels(voxels: np.ndarray, sigma: float = 0.5, sharpness: float = 15.0) -> np.ndarray:
    """
    Apply organic density filter to raw VAE decoder output.

    1. Gaussian blur with *sigma* to smooth jagged boundaries.
    2. Logistic sharpening with *sharpness* to push values back toward 0/1.

    Parameters
    ----------
    voxels : (D, H, W) float array — raw sigmoid output from the decoder.
    sigma  : Gaussian blur radius in voxels (default 0.5).
    sharpness : logistic steepness (default 15.0).

    Returns
    -------
    Smoothed (D, H, W) float array in [0, 1].
    """
    from scipy.ndimage import gaussian_filter

    voxels = gaussian_filter(voxels, sigma=sigma)
    voxels = 1.0 / (1.0 + np.exp(-sharpness * (voxels - 0.5)))
    return voxels


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
    def fill_holes(voxel_grid: np.ndarray) -> np.ndarray:
        from scipy.ndimage import binary_fill_holes
        return binary_fill_holes(voxel_grid > 0.5).astype(np.float32)

    @staticmethod
    def remove_small_components(voxel_grid: np.ndarray, min_size: int = 10) -> np.ndarray:
        from scipy.ndimage import label, sum as ndi_sum
        labeled, n = label(voxel_grid > 0.5)
        sizes = ndi_sum(voxel_grid > 0.5, labeled, range(n + 1))
        return (sizes[labeled] >= min_size).astype(np.float32)


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
            layer = fixed[:, :, z].astype(bool)
            layer_below = fixed[:, :, z - 1].astype(bool)
            unsupported = layer & ~layer_below
            fixed[:, :, z] = (layer & ~unsupported).astype(voxel_grid.dtype)
        return fixed
