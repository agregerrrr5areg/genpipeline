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
        vox = np.zeros((1, 1, 1), dtype=np.float32)
        vox[0, 0, 0] = 1.0  # single solid voxel in 1³ grid → 8 corner nodes
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
