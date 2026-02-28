import numpy as np
import pytest
from dashboard_utils import load_stl_for_plotly, voxel_to_plotly_isosurface

def test_load_stl_for_plotly(tmp_path):
    # Write a minimal ASCII STL (one triangle)
    stl_content = """solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid test
"""
    f = tmp_path / "test.stl"
    f.write_text(stl_content)
    x, y, z, i, j, k = load_stl_for_plotly(str(f))
    assert len(x) == 3   # 3 unique vertices
    assert len(i) == 1   # 1 triangle

def test_voxel_to_isosurface():
    vox = np.zeros((32, 32, 32))
    vox[10:20, 10:20, 10:20] = 1.0   # solid cube in the middle
    data = voxel_to_plotly_isosurface(vox)
    assert "x" in data and "y" in data and "z" in data and "value" in data
    assert len(data["x"]) == 32*32*32
