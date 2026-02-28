"""mesh_export.py — convert voxel density field to STL via marching cubes."""
from __future__ import annotations
import numpy as np


def density_to_stl(density: np.ndarray, output_path: str,
                   threshold: float = 0.5,
                   voxel_size_mm: tuple = (100/32, 20/8, 20/8)) -> str:
    """
    Convert (nx, ny, nz) density field to STL.

    Parameters
    ----------
    density       : float array [0,1]
    output_path   : where to write .stl
    threshold     : marching cubes iso-level (default 0.5)
    voxel_size_mm : physical size of each voxel in mm

    Raises
    ------
    ValueError if no surface found at threshold.
    """
    from skimage.measure import marching_cubes
    from stl import mesh as stlmesh

    if threshold < density.min() or threshold > density.max():
        raise ValueError("density_to_stl: no surface found at threshold — "
                         "threshold is outside the density field range")

    try:
        verts, faces, _, _ = marching_cubes(density, level=threshold)
    except ValueError:
        raise ValueError("density_to_stl: no surface found at threshold — "
                         "check density field values")

    if len(faces) == 0:
        raise ValueError("density_to_stl: no surface found at threshold — "
                         "check density field values")

    sx, sy, sz = voxel_size_mm
    verts_mm = verts * np.array([sx, sy, sz])

    m = stlmesh.Mesh(np.zeros(len(faces), dtype=stlmesh.Mesh.dtype))
    for idx, f in enumerate(faces):
        for j in range(3):
            m.vectors[idx][j] = verts_mm[f[j]]
    m.save(output_path)
    return output_path
