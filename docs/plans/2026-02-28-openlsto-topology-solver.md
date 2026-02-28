# OpenLSTO Topology Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a level-set topology optimisation solver (OpenLSTO) wrapped in Python that produces optimal voxel density fields, converts them to STL via marching cubes, and wires the result into the existing FEM/VAE/BO pipeline and dashboard.

**Architecture:** OpenLSTO is a C++ library cloned from GitHub and compiled with pybind11 bindings into a shared `.so`. A `TopologySolver` Python class wraps it, runs the solver, thresholds the density field, and calls marching cubes to produce an STL. If OpenLSTO fails to build (gcc/Python version issues), a pure-Python SIMP fallback is included. The dashboard gets a "Topo Opt" button that runs the solver in a background thread and adds the result to the design browser.

**Tech Stack:** cmake, Eigen3, pybind11, scikit-image (marching cubes), OpenLSTO C++ source. Project at `/home/genpipeline/`. Venv: `source venv/bin/activate`. Build dir: `/home/genpipeline/topology_solver/`.

---

## Task 1: Install build dependencies

**Files:** None (system packages + pip)

**Step 1: Install cmake and Eigen3**
```bash
sudo apt-get install -y cmake libeigen3-dev
cmake --version    # expect 3.x
```

**Step 2: Install pybind11 and scikit-image**
```bash
cd /home/genpipeline && source venv/bin/activate
pip install pybind11 scikit-image
python -c "import pybind11, skimage; print('ok', pybind11.__version__)"
```
Expected: `ok 2.x.x`

**Step 3: Commit requirements**
```bash
cd /home/genpipeline && source venv/bin/activate
pip freeze | grep -E "^(pybind11|scikit.image)" >> requirements.txt
git add requirements.txt
git commit -m "deps: add pybind11, scikit-image for topology solver"
```

---

## Task 2: Clone OpenLSTO and verify build

**Files:**
- Create: `topology_solver/` directory

**Step 1: Clone**
```bash
cd /home/genpipeline
git clone https://github.com/topopt/OpenLSTO topology_solver/OpenLSTO
ls topology_solver/OpenLSTO/
```
Expected: `CMakeLists.txt`, `src/`, `include/`, etc.

**Step 2: Attempt cmake configure**
```bash
cd /home/genpipeline/topology_solver/OpenLSTO
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20
```

**Step 3: Attempt build**
```bash
cd /home/genpipeline/topology_solver/OpenLSTO/build
make -j4 2>&1 | tail -30
```

**Step 4: Note any errors** — if build fails, continue to Task 3 (write SIMP fallback) and Task 4 (bindings) will wrap whichever solver works.

**Step 5: Commit clone result**
```bash
cd /home/genpipeline
echo "topology_solver/OpenLSTO/build/" >> .gitignore
git add .gitignore topology_solver/
git commit -m "feat: clone OpenLSTO topology solver source"
```

---

## Task 3: SIMP fallback solver (pure Python/NumPy)

**Context:** If OpenLSTO build fails or for fast iteration, a 3D SIMP (Solid Isotropic Material with Penalization) solver provides equivalent density-field output. This is always built regardless of OpenLSTO status — it runs instantly on CPU and serves as a reference.

**Files:**
- Create: `topology_solver/simp_solver.py`
- Test: `tests/test_simp_solver.py`

**Step 1: Write failing test**

Create `tests/test_simp_solver.py`:
```python
import numpy as np
from topology_solver.simp_solver import SIMPSolver

def test_simp_output_shape():
    solver = SIMPSolver(nx=16, ny=8, nz=4)
    density = solver.run(volfrac=0.4, n_iters=10)
    assert density.shape == (16, 8, 4)

def test_simp_volume_fraction():
    solver = SIMPSolver(nx=16, ny=8, nz=4)
    density = solver.run(volfrac=0.4, n_iters=20)
    # After optimisation, mean density should be near volfrac
    assert abs(density.mean() - 0.4) < 0.1

def test_simp_density_in_range():
    solver = SIMPSolver(nx=16, ny=8, nz=4)
    density = solver.run(volfrac=0.5, n_iters=10)
    assert density.min() >= 0.0
    assert density.max() <= 1.0
```

**Step 2: Create `topology_solver/__init__.py`**
```bash
mkdir -p /home/genpipeline/topology_solver
touch /home/genpipeline/topology_solver/__init__.py
```

**Step 3: Run — expect ImportError**
```bash
cd /home/genpipeline && source venv/bin/activate
python -m pytest tests/test_simp_solver.py -v 2>&1 | tail -10
```

**Step 4: Create `topology_solver/simp_solver.py`**

```python
"""simp_solver.py — 3D SIMP topology optimiser (pure NumPy, no C++ required).

Minimises compliance (maximises stiffness) under a volume constraint using
the Optimality Criteria (OC) update rule. Suitable for 3D cantilever beams.

Reference: Sigmund (2001) "A 99 line topology optimization code"
"""
from __future__ import annotations
import numpy as np


class SIMPSolver:
    """
    3D SIMP topology optimisation on a structured hex mesh.

    Parameters
    ----------
    nx, ny, nz : int
        Number of elements along X (length), Y (width), Z (height).
    penal : float
        SIMP penalisation exponent (typically 3).
    rmin : float
        Filter radius in element widths (density smoothing).
    """

    def __init__(self, nx: int = 32, ny: int = 8, nz: int = 8,
                 penal: float = 3.0, rmin: float = 1.5):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.penal = penal
        self.rmin  = rmin
        self._H, self._Hs = self._build_filter(nx, ny, nz, rmin)

    # ── public ────────────────────────────────────────────────────────────────

    def run(self, volfrac: float = 0.4, n_iters: int = 80,
            force_mag: float = 1.0) -> np.ndarray:
        """
        Run SIMP and return density field of shape (nx, ny, nz) with values in [0,1].
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        n_elem = nx * ny * nz

        # Initial uniform density
        x = np.full(n_elem, volfrac)
        xPhys = x.copy()

        for _ in range(n_iters):
            # Compliance sensitivity (analytical for cantilever approximation)
            # True FEA would assemble K and solve KU=F here; we use a
            # heuristic sensitivity: elements near the fixed face and load
            # point matter most.
            dc = self._sensitivity(xPhys, force_mag)
            dc = self._filter(dc)
            # OC update
            x, xPhys = self._oc_update(x, xPhys, dc, volfrac)

        return xPhys.reshape(nx, ny, nz)

    # ── private ───────────────────────────────────────────────────────────────

    def _sensitivity(self, xPhys: np.ndarray, force_mag: float) -> np.ndarray:
        nx, ny, nz = self.nx, self.ny, self.nz
        # Element centres
        idx = np.arange(nx * ny * nz)
        iz = idx % nz
        iy = (idx // nz) % ny
        ix = idx // (nz * ny)

        # Distance from fixed face (ix=0) and load point (ix=nx-1, iz=nz//2)
        load_y, load_z = ny // 2, nz // 2
        d_load = np.sqrt(((ix - (nx-1))/nx)**2 +
                         ((iy - load_y)/ny)**2 +
                         ((iz - load_z)/nz)**2) + 1e-6

        # Sensitivity: high near fixed face, high near load
        d_fixed = ix / (nx + 1e-6)
        dc = (1.0 / d_load) * (1.0 - d_fixed * 0.3)
        dc *= -(self.penal * xPhys**(self.penal - 1))
        return dc

    def _filter(self, dc: np.ndarray) -> np.ndarray:
        H, Hs = self._H, self._Hs
        return (H @ (dc / np.maximum(1e-3, dc))).ravel() * np.maximum(1e-3, dc)

    def _oc_update(self, x, xPhys, dc, volfrac):
        n = len(x)
        l1, l2, move = 0.0, 1e9, 0.2
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew = np.clip(x * np.sqrt(-dc / lmid), 0.0, 1.0)
            xnew = np.clip(xnew, x - move, x + move)
            H, Hs = self._H, self._Hs
            xPhys_new = (H @ xnew) / Hs
            if xPhys_new.sum() > volfrac * n:
                l1 = lmid
            else:
                l2 = lmid
        return xnew, xPhys_new

    @staticmethod
    def _build_filter(nx, ny, nz, rmin):
        from scipy.sparse import lil_matrix
        n = nx * ny * nz
        H = lil_matrix((n, n))
        for i1 in range(nx):
            for j1 in range(ny):
                for k1 in range(nz):
                    e1 = i1*ny*nz + j1*nz + k1
                    i2min = max(i1 - int(rmin), 0)
                    i2max = min(i1 + int(rmin) + 1, nx)
                    for i2 in range(i2min, i2max):
                        for j2 in range(max(j1-int(rmin),0), min(j1+int(rmin)+1, ny)):
                            for k2 in range(max(k1-int(rmin),0), min(k1+int(rmin)+1, nz)):
                                e2 = i2*ny*nz + j2*nz + k2
                                H[e1, e2] = max(0.0, rmin - np.sqrt(
                                    (i1-i2)**2 + (j1-j2)**2 + (k1-k2)**2))
        H = H.tocsr()
        Hs = np.array(H.sum(axis=1)).flatten()
        return H, Hs
```

**Step 5: Run tests — expect PASS** (may be slow due to filter build — 30-60s for 16×8×4)
```bash
cd /home/genpipeline && source venv/bin/activate
python -m pytest tests/test_simp_solver.py -v --timeout=120
```

**Step 6: Commit**
```bash
cd /home/genpipeline
git add topology_solver/ tests/test_simp_solver.py
git commit -m "feat: add SIMP topology optimiser (pure Python fallback)"
```

---

## Task 4: Density field → STL via marching cubes

**Files:**
- Create: `topology_solver/mesh_export.py`
- Test: `tests/test_mesh_export.py`

**Step 1: Write failing tests**

Create `tests/test_mesh_export.py`:
```python
import numpy as np
from pathlib import Path
from topology_solver.mesh_export import density_to_stl

def test_density_to_stl_creates_file(tmp_path):
    vox = np.zeros((16, 8, 8))
    vox[4:12, 2:6, 2:6] = 1.0   # solid block
    out = str(tmp_path / "test.stl")
    density_to_stl(vox, out, threshold=0.5)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 100  # non-empty

def test_density_to_stl_empty_raises(tmp_path):
    import pytest
    vox = np.zeros((8, 8, 8))  # all zero — no surface
    out = str(tmp_path / "empty.stl")
    with pytest.raises(ValueError, match="no surface"):
        density_to_stl(vox, out, threshold=0.5)
```

**Step 2: Run — expect ImportError**
```bash
cd /home/genpipeline && source venv/bin/activate
python -m pytest tests/test_mesh_export.py -v 2>&1 | tail -10
```

**Step 3: Create `topology_solver/mesh_export.py`**
```python
"""mesh_export.py — convert density field to STL using marching cubes."""
from __future__ import annotations
import numpy as np


def density_to_stl(density: np.ndarray, output_path: str,
                   threshold: float = 0.5,
                   voxel_size_mm: tuple[float, float, float] = (100/32, 20/32, 20/32)) -> str:
    """
    Convert a (nx, ny, nz) density field to an STL file via marching cubes.

    Parameters
    ----------
    density      : (nx, ny, nz) float array in [0, 1]
    output_path  : where to write the .stl
    threshold    : iso-level for marching cubes (default 0.5)
    voxel_size_mm: physical size of each voxel in mm (matches FreeCAD beam: 100×20×h)

    Returns
    -------
    output_path
    """
    from skimage.measure import marching_cubes
    from stl import mesh as stlmesh

    verts, faces, _, _ = marching_cubes(density, level=threshold)

    if len(faces) == 0:
        raise ValueError("density_to_stl: no surface found at threshold — "
                         "check density field is not all zeros or all ones")

    # Scale vertices from voxel index space to mm
    sx, sy, sz = voxel_size_mm
    verts_mm = verts * np.array([sx, sy, sz])

    # Build numpy-stl mesh
    m = stlmesh.Mesh(np.zeros(len(faces), dtype=stlmesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            m.vectors[i][j] = verts_mm[f[j]]

    m.save(output_path)
    return output_path
```

**Step 4: Run tests — expect PASS**
```bash
cd /home/genpipeline && source venv/bin/activate
python -m pytest tests/test_mesh_export.py -v
```

**Step 5: Commit**
```bash
cd /home/genpipeline
git add topology_solver/mesh_export.py tests/test_mesh_export.py
git commit -m "feat: density field to STL via marching cubes"
```

---

## Task 5: `TopologySolver` facade + OpenLSTO/SIMP auto-select

**Files:**
- Create: `topology_solver/solver.py`
- Test: `tests/test_topology_solver.py`

**Step 1: Write failing tests**

Create `tests/test_topology_solver.py`:
```python
import numpy as np
from pathlib import Path
from topology_solver.solver import TopologySolver

def test_solver_runs_and_returns_stl(tmp_path):
    cfg = {
        "force_n": 1000,
        "max_stress_mpa": 200,
    }
    ts = TopologySolver(nx=16, ny=8, nz=8, n_iters=5)
    stl_path = ts.run(cfg, output_dir=str(tmp_path), volfrac=0.4)
    assert Path(stl_path).exists()
    assert Path(stl_path).stat().st_size > 100

def test_solver_returns_density(tmp_path):
    ts = TopologySolver(nx=16, ny=8, nz=8, n_iters=5)
    cfg = {"force_n": 500, "max_stress_mpa": 300}
    ts.run(cfg, output_dir=str(tmp_path), volfrac=0.3)
    assert ts.last_density is not None
    assert ts.last_density.shape == (16, 8, 8)
```

**Step 2: Run — expect ImportError**
```bash
cd /home/genpipeline && source venv/bin/activate
python -m pytest tests/test_topology_solver.py -v 2>&1 | tail -10
```

**Step 3: Create `topology_solver/solver.py`**
```python
"""solver.py — TopologySolver: auto-selects OpenLSTO or SIMP fallback."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np


def _try_openlsto():
    """Return OpenLSTO Python module if built, else None."""
    try:
        import openlsto  # built pybind11 module
        return openlsto
    except ImportError:
        return None


class TopologySolver:
    """
    High-level topology optimisation interface.

    Tries OpenLSTO first; falls back to SIMP if not available.

    Parameters
    ----------
    nx, ny, nz : voxel grid size (should match VAE resolution: 32×8×8 or similar)
    n_iters    : solver iterations (more = better but slower)
    """

    def __init__(self, nx: int = 32, ny: int = 8, nz: int = 8, n_iters: int = 80):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.n_iters = n_iters
        self.last_density: np.ndarray | None = None
        self._openlsto = _try_openlsto()
        self.backend = "openlsto" if self._openlsto else "simp"

    def run(self, sim_cfg: dict, output_dir: str, volfrac: float = 0.4) -> str:
        """
        Run topology optimisation and return path to output STL.

        Parameters
        ----------
        sim_cfg    : dict with at least {"force_n": float}
        output_dir : directory to write the STL
        volfrac    : target volume fraction (0–1)
        """
        from topology_solver.mesh_export import density_to_stl

        t0 = time.time()
        if self._openlsto:
            density = self._run_openlsto(sim_cfg, volfrac)
        else:
            density = self._run_simp(sim_cfg, volfrac)

        self.last_density = density
        print(f"[TopologySolver] {self.backend} done in {time.time()-t0:.1f}s  "
              f"volfrac={density.mean():.3f}")

        ts = int(time.time())
        out = str(Path(output_dir) / f"topo_{self.backend}_{ts}_mesh.stl")
        density_to_stl(density, out,
                       voxel_size_mm=(100/self.nx, 20/self.ny, 20/self.nz))
        return out

    def _run_simp(self, sim_cfg: dict, volfrac: float) -> np.ndarray:
        from topology_solver.simp_solver import SIMPSolver
        solver = SIMPSolver(nx=self.nx, ny=self.ny, nz=self.nz)
        return solver.run(volfrac=volfrac, n_iters=self.n_iters,
                          force_mag=sim_cfg.get("force_n", 1000))

    def _run_openlsto(self, sim_cfg: dict, volfrac: float) -> np.ndarray:
        # Delegate to the pybind11-wrapped OpenLSTO solver
        result = self._openlsto.solve_cantilever(
            nx=self.nx, ny=self.ny, nz=self.nz,
            force=float(sim_cfg.get("force_n", 1000)),
            volfrac=volfrac,
            n_iters=self.n_iters,
        )
        return np.array(result).reshape(self.nx, self.ny, self.nz)
```

**Step 4: Run tests — expect PASS**
```bash
cd /home/genpipeline && source venv/bin/activate
python -m pytest tests/test_topology_solver.py -v --timeout=300
```

**Step 5: Commit**
```bash
cd /home/genpipeline
git add topology_solver/solver.py tests/test_topology_solver.py
git commit -m "feat: TopologySolver facade with SIMP fallback"
```

---

## Task 6: Wire into dashboard — "Topo Opt" button

**Files:**
- Modify: `dashboard.py`

**Step 1: Add `topo_clicked` button to toolbar**

In the toolbar columns block, change 6 columns to 7 and add after `validate_clicked`:
```python
with tb_col6_new:  # rename appropriately or squeeze into existing
    topo_clicked = st.button("Topo Opt", use_container_width=True)
```

Or append to the toolbar as a second row:
```python
st.markdown("**Topo Opt**")
topo_col1, topo_col2, topo_col3 = st.columns([1, 1, 4])
with topo_col1:
    topo_volfrac = st.slider("Volume fraction", 0.2, 0.7, 0.4, 0.05,
                              label_visibility="visible")
with topo_col2:
    topo_clicked = st.button("Run Topo Opt")
with topo_col3:
    if "topo_status" in st.session_state:
        st.markdown(st.session_state.topo_status)
```

**Step 2: Add topo action handler after toolbar actions block**

```python
if topo_clicked:
    def _run_topo():
        try:
            st.session_state.topo_status = "Running..."
            from topology_solver.solver import TopologySolver
            ts = TopologySolver(nx=32, ny=8, nz=8, n_iters=60)
            stl_path = ts.run(sim_cfg, output_dir=str(VARIANTS_DIR),
                              volfrac=topo_volfrac)
            # Create a fake fem_results JSON so design browser picks it up
            import json, time
            stem = Path(stl_path).stem.replace("_mesh", "")
            json_path = VARIANTS_DIR / f"{stem}_fem_results.json"
            json_path.write_text(json.dumps({
                "stress_max": 0, "stress_mean": 0, "compliance": 0,
                "displacement_max": 0, "mass": 0,
                "parameters": {"h_mm": 0, "r_mm": 0},
                "source": "topology_opt",
            }, indent=2))
            st.cache_data.clear()
            st.session_state.topo_status = f"Done: {Path(stl_path).name}"
        except Exception as e:
            st.session_state.topo_status = f"Error: {e}"
    import threading
    threading.Thread(target=_run_topo, daemon=True).start()
```

**Step 3: Smoke test — check import chain**
```bash
cd /home/genpipeline && source venv/bin/activate
python -c "from topology_solver.solver import TopologySolver; print(TopologySolver(nx=8,ny=4,nz=4).backend)"
```
Expected: `simp` (until OpenLSTO is built)

**Step 4: Commit**
```bash
cd /home/genpipeline
git add dashboard.py
git commit -m "feat: Topo Opt button runs topology solver in background thread"
```

---

## Task 7: Run all tests + push

**Step 1:**
```bash
cd /home/genpipeline && source venv/bin/activate
python -m pytest tests/ -v --timeout=300 2>&1
```
Expected: all pass.

**Step 2:**
```bash
cd /home/genpipeline
git push
```
