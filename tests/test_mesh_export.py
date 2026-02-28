import numpy as np
from pathlib import Path
import pytest
from topology_solver.mesh_export import density_to_stl

def test_density_to_stl_creates_file(tmp_path):
    vox = np.zeros((16, 8, 8))
    vox[4:12, 2:6, 2:6] = 1.0
    out = str(tmp_path / "test.stl")
    density_to_stl(vox, out, threshold=0.5)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 100

def test_density_to_stl_empty_raises(tmp_path):
    vox = np.zeros((8, 8, 8))
    out = str(tmp_path / "empty.stl")
    with pytest.raises(ValueError, match="no surface"):
        density_to_stl(vox, out, threshold=0.5)
