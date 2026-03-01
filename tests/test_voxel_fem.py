# tests/test_voxel_fem.py
import numpy as np
import pytest
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from voxel_fem import _wsl_to_win, VoxelHexMesher
from pipeline_utils import FEM_SENTINEL


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


# ── .frd parser tests ─────────────────────────────────────────────────────────

def _fmt_val(v: float) -> str:
    """Format a float to exactly 12 chars (CalculiX fixed-width slot)."""
    return f"{v:12.6E}"


def _frd_record(node_id: int, *values) -> str:
    """Build a CalculiX -1 data record with correct fixed-width layout."""
    node_str = f"{node_id:10d}"   # 10-char node ID
    vals_str  = "".join(_fmt_val(v) for v in values)
    return f" -1{node_str}{vals_str}"


# Minimal synthetic .frd: 2 displacement nodes + 1 stress node.
# Layout matches _parse_frd() exactly: ' -1' prefix, 10-char node ID, 12-char slots.
_MINIMAL_FRD = "\n".join([
    "     2C  minimal_test",
    " -4  DISP        4    1",
    " -5  D1          1    1    0    0",
    " -5  D2          1    2    0    0",
    " -5  D3          1    3    0    0",
    " -5  ALL         1    0    0    0",
    _frd_record(1, 1.0e-2, 2.0e-2, 3.0e-2),   # |disp| = sqrt(14)*1e-2
    _frd_record(2, 4.0e-2, 5.0e-2, 6.0e-2),   # |disp| = sqrt(77)*1e-2
    " -3",
    " -4  STRESS      6    1",
    " -5  SXX         1    4    1    1",
    " -5  SYY         1    4    1    2",
    " -5  SZZ         1    4    1    3",
    " -5  SXY         1    4    1    4",
    " -5  SYZ         1    4    1    5",
    " -5  SZX         1    4    1    6",
    _frd_record(1, 1.0e2, 5.0e1, 2.5e1, 1.0e1, 5.0e0, 0.0),  # vm ≈ 68.9 MPa
    " -3",
    "  9999",
])


class TestParseFrd:
    def test_parse_frd_valid_output(self, tmp_path):
        frd_file = tmp_path / "test.frd"
        frd_file.write_text(_MINIMAL_FRD)
        result = VoxelHexMesher._parse_frd(str(frd_file))
        assert result["stress_max"] > 0.0, "Expected non-zero stress"
        assert result["displacement_max"] > 0.0, "Expected non-zero displacement"
        assert result.get("failure_reason") is None

    def test_parse_frd_malformed_lines(self, tmp_path):
        """Corrupt displacement and stress lines — should return zeros, not crash."""
        bad_frd = textwrap.dedent("""\
             2C  corrupt_test
            -4  DISP        4    1
            -5  D1          1    1    0    0
            -1         1 NOT_A_NUMBER BAD BAD
            -3
            -4  STRESS      6    1
            -5  SXX         1    4    1    1
            -1         1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            -3
             9999
        """)
        frd_file = tmp_path / "bad.frd"
        frd_file.write_text(bad_frd)
        result = VoxelHexMesher._parse_frd(str(frd_file))
        # Should not raise — returns zeros for malformed content
        assert isinstance(result, dict)
        assert result["stress_max"] == 0.0
        assert result["displacement_max"] == 0.0

    def test_parse_frd_missing_file_returns_sentinel(self, tmp_path):
        result = VoxelHexMesher._parse_frd(str(tmp_path / "nonexistent.frd"))
        assert result["stress_max"] == FEM_SENTINEL


class TestRunCcxFailureModes:
    def test_run_ccx_timeout_returns_sentinel(self, tmp_path):
        """When ccx times out, run_ccx should return sentinel with failure_reason=timeout."""
        vox = np.ones((4, 4, 4), dtype=np.float32)
        inp = str(tmp_path / "cube.inp")
        VoxelHexMesher.voxels_to_inp(vox, output_path=inp)

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ccx", timeout=1)):
            result = VoxelHexMesher.run_ccx(inp, ccx_cmd="ccx", timeout=1)

        assert result["stress_max"] == FEM_SENTINEL
        assert result["compliance"] == FEM_SENTINEL
        assert result["failure_reason"] == "timeout"

    def test_run_ccx_bad_exit_code_returns_sentinel(self, tmp_path):
        """When ccx exits non-zero, run_ccx should return sentinel with failure_reason=ccx_error."""
        vox = np.ones((4, 4, 4), dtype=np.float32)
        inp = str(tmp_path / "cube.inp")
        VoxelHexMesher.voxels_to_inp(vox, output_path=inp)

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "error"
        with patch("subprocess.run", return_value=mock_proc):
            result = VoxelHexMesher.run_ccx(inp, ccx_cmd="ccx", timeout=30)

        assert result["stress_max"] == FEM_SENTINEL
        assert result["failure_reason"] == "ccx_error"

    def test_run_ccx_no_frd_returns_sentinel(self, tmp_path):
        """When ccx exits 0 but produces no .frd, return sentinel with failure_reason=no_frd."""
        vox = np.ones((4, 4, 4), dtype=np.float32)
        inp = str(tmp_path / "cube.inp")
        VoxelHexMesher.voxels_to_inp(vox, output_path=inp)

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        with patch("subprocess.run", return_value=mock_proc):
            result = VoxelHexMesher.run_ccx(inp, ccx_cmd="ccx", timeout=30)

        assert result["stress_max"] == FEM_SENTINEL
        assert result["failure_reason"] == "no_frd"
