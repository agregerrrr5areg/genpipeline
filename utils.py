import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def convert_windows_path_to_wsl(windows_path: str) -> str:
    """
    Convert a Windows path (e.g., 'C:\\path\\to\\file.FCStd') to a WSL2 path (e.g., '/mnt/c/path/to/file.FCStd').
    """
    if not windows_path:
        return ""
    # Replace backslashes with forward slashes and convert drive letter
    wsl_path = windows_path.replace("\\", "/")
    if wsl_path.startswith("file://"):
        wsl_path = wsl_path[7:]
    if len(wsl_path) > 1 and wsl_path[1] == ":":
        drive_letter = wsl_path[0].upper()
        wsl_path = "/mnt/" + drive_letter + wsl_path[2:]
    return wsl_path


class VoxelConverter:
    @staticmethod
    def voxel_to_mesh(voxel_grid: np.ndarray, voxel_size: float = 1.0, bbox: dict = None) -> dict:
        from skimage import measure

        # Scale Preservation: use real-world spacing if bbox is available
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
            # Ensure voxel_grid is float for marching cubes
            v_float = voxel_grid.astype(np.float32)
            
            verts, faces, normals, _ = measure.marching_cubes(
                v_float,
                level=0.5,
                spacing=spacing
            )
            
            # Shift back to real-world origin
            verts += origin

            return {
                'vertices': verts,
                'faces': faces,
                'normals': normals
            }
        except Exception as e:
            logger.error(f"Marching cubes failed: {e}")
            return None

    @staticmethod
    def mesh_to_voxel(vertices: np.ndarray, faces: np.ndarray,
                      resolution: int = 32) -> np.ndarray:
        try:
            import trimesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            voxelized = mesh.voxelized(pitch=mesh.extents.max() / resolution)
            return voxelized.matrix.astype(np.float32)
        except Exception as e:
            logger.error(f"Mesh to voxel conversion failed: {e}")
            return np.zeros((resolution, resolution, resolution), dtype=np.float32)

    @staticmethod
    def smooth_voxel_grid(voxel_grid: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(voxel_grid, sigma=sigma)

    @staticmethod
    def threshold_voxel_grid(voxel_grid: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (voxel_grid > threshold).astype(np.float32)

    @staticmethod
    def fill_holes(voxel_grid: np.ndarray, structure=None) -> np.ndarray:
        from scipy.ndimage import binary_fill_holes
        if structure is None:
            structure = np.ones((3, 3, 3))
        filled = binary_fill_holes(voxel_grid > 0.5, structure=structure)
        return filled.astype(np.float32)

    @staticmethod
    def remove_small_components(voxel_grid: np.ndarray, min_size: int = 10) -> np.ndarray:
        from scipy.ndimage import label, sum as ndi_sum
        labeled, num_features = label(voxel_grid > 0.5)
        component_sizes = ndi_sum(voxel_grid > 0.5, labeled, range(num_features + 1))

        mask = component_sizes >= min_size
        filtered = mask[labeled]
        return filtered.astype(np.float32)


class FreeCADInterface:
    @staticmethod
    def export_voxel_to_freecad(voxel_grid: np.ndarray, output_path: str, resolution: int = 32):
        try:
            import FreeCAD
            import Part
        except ImportError:
            logger.error("FreeCAD not available")
            return False

        # Convert Windows path to WSL path if needed
        output_path = convert_windows_path_to_wsl(output_path)

        mesh_data = VoxelConverter.voxel_to_mesh(voxel_grid, voxel_size=1.0 / resolution)
        if mesh_data is None:
            return False

        try:
            doc = FreeCAD.newDocument()
            mesh_obj = doc.addObject("Mesh::Feature", "ImportedMesh")
            mesh = FreeCAD.Mesh.Mesh(
                mesh_data['vertices'].tolist(),
                mesh_data['faces'].tolist()
            )
            mesh_obj.Mesh = mesh

            doc.saveAs(output_path)
            doc.close()
            logger.info(f"Exported voxel grid to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to FreeCAD: {e}")
            return False

    @staticmethod
    def create_parametric_template(
        param_names: list,
        param_ranges: dict,
        template_path: str = "parametric_template.FCStd"
    ):
        try:
            import FreeCAD
            import Sketcher
        except ImportError:
            logger.error("FreeCAD not available")
            return False

        # Convert Windows path to WSL path if needed
        template_path = convert_windows_path_to_wsl(template_path)

        try:
            doc = FreeCAD.newDocument()

            spreadsheet = doc.addObject("Spreadsheet::Sheet", "Parameters")

            for i, param_name in enumerate(param_names, start=1):
                min_val, max_val = param_ranges[param_name]
                spreadsheet.setAlias(f'A{i}', param_name)
                spreadsheet.set(f'A{i}', f'{min_val}')

            doc.saveAs(template_path)
            doc.close()

            logger.info(f"Created parametric template: {template_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            return False

    @staticmethod
    def run_fem_simulation(freecad_doc_path: str) -> dict:
        try:
            import FreeCAD
            import FEM
        except ImportError:
            logger.error("FreeCAD/FEM not available")
            return {}

        # Convert Windows path to WSL path if needed
        freecad_doc_path = convert_windows_path_to_wsl(freecad_doc_path)

        try:
            doc = FreeCAD.open(freecad_doc_path)

            for analysis in doc.Objects:
                if analysis.TypeId == "Fem::FemAnalysis":
                    solver = None
                    for obj in analysis.Group:
                        if "Solver" in obj.TypeId:
                            solver = obj
                            break

                    if solver:
                        from FEM import FemSolver
                        FemSolver.runSolver(analysis, solver)

            results = FreeCADInterface._extract_fem_results(doc)
            doc.close()
            return results

        except Exception as e:
            logger.error(f"FEM simulation failed: {e}")
            return {}

    @staticmethod
    def _extract_fem_results(doc) -> dict:
        results = {
            "stress_max": 0.0,
            "stress_mean": 0.0,
            "compliance": 0.0,
            "displacement_max": 0.0
        }

        for obj in doc.Objects:
            if hasattr(obj, 'StressValues'):
                stress_vals = obj.StressValues
                if stress_vals:
                    results["stress_max"] = float(max(stress_vals))
                    results["stress_mean"] = float(np.mean(stress_vals))

            if hasattr(obj, 'DisplacementLengths'):
                displacements = obj.DisplacementLengths
                if displacements:
                    results["compliance"] = float(np.sum(displacements))
                    results["displacement_max"] = float(max(displacements))

        return results


class PerformanceNormalizer:
    def __init__(self):
        self.stress_mean = None
        self.stress_std = None
        self.compliance_mean = None
        self.compliance_std = None

    def fit(self, stresses: np.ndarray, compliances: np.ndarray):
        self.stress_mean = stresses.mean()
        self.stress_std = stresses.std() + 1e-6
        self.compliance_mean = compliances.mean()
        self.compliance_std = compliances.std() + 1e-6

    def normalize(self, stress: float, compliance: float) -> Tuple[float, float]:
        stress_norm = (stress - self.stress_mean) / self.stress_std
        compliance_norm = (compliance - self.compliance_mean) / self.compliance_std
        return stress_norm, compliance_norm

    def denormalize(self, stress_norm: float, compliance_norm: float) -> Tuple[float, float]:
        stress = stress_norm * self.stress_std + self.stress_mean
        compliance = compliance_norm * self.compliance_std + self.compliance_mean
        return stress, compliance


class ManufacturabilityConstraints:
    def __init__(self, min_feature_size: float = 1.0, max_overhang_angle: float = 45.0, config: dict = None):
        self.config = config or {}
        # If config provides specific material/manufacturing constraints, override defaults
        mfg_cfg = self.config.get("manufacturing_constraints", {})
        self.min_feature_size = mfg_cfg.get("min_feature_size_mm", min_feature_size)
        self.max_overhang_angle = mfg_cfg.get("max_overhang_angle_deg", max_overhang_angle)

    def check_min_feature_size(self, voxel_grid: np.ndarray, voxel_size: float = 1.0) -> bool:
        min_physical_size = self.min_feature_size
        min_voxel_size = min_physical_size / voxel_size

        from scipy.ndimage import binary_erosion, binary_dilation
        eroded = binary_erosion(voxel_grid > 0.5)

        if eroded.sum() == 0:
            return False

        restored = binary_dilation(eroded)
        recovery_ratio = restored.sum() / (voxel_grid > 0.5).sum()

        return recovery_ratio > 0.7

    def check_overhang_constraint(self, voxel_grid: np.ndarray) -> bool:
        voxel_grid = voxel_grid > 0.5

        for z in range(1, voxel_grid.shape[2]):
            layer = voxel_grid[:, :, z]
            layer_below = voxel_grid[:, :, z-1]

            unsupported = layer & ~layer_below

            if unsupported.sum() > 0.3 * layer.sum():
                return False

        return True

    def apply_constraints(self, voxel_grid: np.ndarray, voxel_size: float = 1.0) -> np.ndarray:
        constrained = voxel_grid.copy()

        # Conservative min volume: sphere with diameter = min_feature_size
        # V = (4/3) * pi * (r^3)
        radius_mm = self.min_feature_size / 2.0
        min_vol_mm3 = (4.0/3.0) * np.pi * (radius_mm ** 3)
        min_voxels = int(min_vol_mm3 / (voxel_size ** 3))

        if not self.check_min_feature_size(constrained, voxel_size=voxel_size):
            logger.warning(f"Design violates minimum feature size ({self.min_feature_size}mm)")
            # Filter out components smaller than the minimal physical volume
            constrained = VoxelConverter.remove_small_components(constrained, min_size=max(min_voxels, 1))

        if not self.check_overhang_constraint(constrained):
            logger.warning(f"Design violates overhang constraint ({self.max_overhang_angle} deg)")
            constrained = self._fix_overhangs(constrained)

        return constrained

    @staticmethod
    def _fix_overhangs(voxel_grid: np.ndarray, max_angle: float = 45.0) -> np.ndarray:
        from scipy.ndimage import binary_erosion
        fixed = voxel_grid.copy()

        for z in range(fixed.shape[2] - 2, 0, -1):
            layer = fixed[:, :, z]
            layer_below = fixed[:, :, z-1]

            unsupported = layer & ~layer_below
            fixed[:, :, z] = fixed[:, :, z] & ~unsupported

        return fixed


class GeometryMetrics:
    @staticmethod
    def compute_volume(voxel_grid: np.ndarray, voxel_size: float = 1.0) -> float:
        occupied_voxels = (voxel_grid > 0.5).sum()
        volume = occupied_voxels * (voxel_size ** 3)
        return volume

    @staticmethod
    def compute_surface_area(voxel_grid: np.ndarray, voxel_size: float = 1.0) -> float:
        from scipy.ndimage import sobel
        
        sx = sobel(voxel_grid.astype(float), axis=0)
        sy = sobel(voxel_grid.astype(float), axis=1)
        sz = sobel(voxel_grid.astype(float), axis=2)

        surface_area = np.sqrt(sx**2 + sy**2 + sz**2).sum() * (voxel_size ** 2)
        return surface_area

    @staticmethod
    def compute_smoothness(voxel_grid: np.ndarray) -> float:
        """Measures geometric smoothness (0.0 to 1.0)."""
        from scipy.ndimage import gaussian_filter
        if np.max(voxel_grid) < 0.1: return 0.0
        
        v = (voxel_grid > 0.5).astype(float)
        v_smooth = gaussian_filter(v, sigma=1.0)
        diff = np.abs(v - v_smooth).mean()
        return 1.0 - (diff * 2.0)

    @staticmethod
    def compute_connectivity(voxel_grid: np.ndarray) -> int:
        """Returns the number of connected components."""
        from scipy.ndimage import label
        _, num_features = label(voxel_grid > 0.5)
        return num_features

    @staticmethod
    def compute_centroid(voxel_grid: np.ndarray) -> np.ndarray:
        coords = np.where(voxel_grid > 0.5)
        if len(coords[0]) == 0:
            return np.array([0.0, 0.0, 0.0])
        centroid = np.array([coords[i].mean() for i in range(3)])
        return centroid

    @staticmethod
    def compute_moments_of_inertia(voxel_grid: np.ndarray, voxel_size: float = 1.0) -> dict:
        coords = np.where(voxel_grid > 0.5)
        if len(coords[0]) == 0:
            return {"Ixx": 0.0, "Iyy": 0.0, "Izz": 0.0}

        positions = np.array(coords).T * voxel_size
        centroid = positions.mean(axis=0)
        centered = positions - centroid

        Ixx = (centered[:, 1]**2 + centered[:, 2]**2).sum() * (voxel_size**3)
        Iyy = (centered[:, 0]**2 + centered[:, 2]**2).sum() * (voxel_size**3)
        Izz = (centered[:, 0]**2 + centered[:, 1]**2).sum() * (voxel_size**3)

        return {
            "Ixx": float(Ixx),
            "Iyy": float(Iyy),
            "Izz": float(Izz)
        }


if __name__ == "__main__":
    voxel_test = np.random.rand(32, 32, 32) > 0.7

    print("Geometry Metrics:")
    print(f"Volume: {GeometryMetrics.compute_volume(voxel_test):.2f}")
    print(f"Surface Area: {GeometryMetrics.compute_surface_area(voxel_test):.2f}")
    print(f"Centroid: {GeometryMetrics.compute_centroid(voxel_test)}")

    moi = GeometryMetrics.compute_moments_of_inertia(voxel_test)
    print(f"Moments of Inertia: {moi}")

    mfg = ManufacturabilityConstraints()
    print(f"\nMin feature size OK: {mfg.check_min_feature_size(voxel_test)}")
    print(f"Overhang constraint OK: {mfg.check_overhang_constraint(voxel_test)}")
