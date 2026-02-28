"""dashboard_utils.py — helper functions for the Streamlit dashboard."""
from __future__ import annotations
import numpy as np


def load_stl_for_plotly(path: str):
    """
    Load an STL file and return (x, y, z, i, j, k) arrays for Plotly mesh3d.
    Deduplicates vertices so the mesh renders cleanly.
    """
    from stl import mesh as stlmesh
    m = stlmesh.Mesh.from_file(path)
    # m.vectors shape: (n_triangles, 3, 3) — last dim is xyz
    verts = m.vectors.reshape(-1, 3)          # (n_tri*3, 3)
    # Deduplicate
    unique_verts, inv = np.unique(verts, axis=0, return_inverse=True)
    tris = inv.reshape(-1, 3)
    x, y, z = unique_verts[:, 0], unique_verts[:, 1], unique_verts[:, 2]
    i, j, k = tris[:, 0], tris[:, 1], tris[:, 2]
    return x, y, z, i, j, k


def voxel_to_plotly_isosurface(voxel: np.ndarray) -> dict:
    """
    Convert a (D,H,W) voxel grid to dict of flat arrays for Plotly isosurface.
    Returns {"x", "y", "z", "value"} — all shape (D*H*W,).
    """
    D, H, W = voxel.shape
    xi = np.arange(W)
    yi = np.arange(H)
    zi = np.arange(D)
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
    return {
        "x": xx.ravel().astype(float),
        "y": yy.ravel().astype(float),
        "z": zz.ravel().astype(float),
        "value": voxel.ravel().astype(float),
    }
