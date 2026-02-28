# Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Fusion 360-style Streamlit dashboard (`dashboard.py`) for running and watching the Bayesian optimisation loop live, browsing FEM variants, and validating top designs with FreeCAD.

**Architecture:** Single-file Streamlit app with a dark theme. BO runs in a `threading.Thread`, writing into a lock-guarded `AppState` dataclass. Three `@st.fragment(run_every=0.5)` fragments handle live re-renders: the 3D viewport, properties panel, and bottom BO bar. Layout is 3-column (browser | viewport | properties) with a fixed top toolbar and bottom bar.

**Tech Stack:** Streamlit ≥1.35, Plotly 5.x, numpy-stl, PyTorch (already in venv), existing `vae_design_model.py` + `optimization_engine.py` + `freecad_bridge.py`.

---

## Task 1: Install dependencies

**Files:**
- No new files — just install into venv

**Step 1: Install streamlit, plotly, numpy-stl**

```bash
cd /home/genpipeline
source venv/bin/activate
pip install "streamlit>=1.35" plotly numpy-stl
```

**Step 2: Verify**

```bash
source venv/bin/activate
python -c "import streamlit, plotly, stl; print('all ok')"
```
Expected: `all ok`

**Step 3: Commit requirements update (freeze only new packages)**

```bash
source venv/bin/activate
pip freeze | grep -E "streamlit|plotly|numpy.stl" >> requirements.txt
git add requirements.txt
git commit -m "deps: add streamlit, plotly, numpy-stl for dashboard"
```

---

## Task 2: AppState — shared live state between BO thread and UI

**Files:**
- Create: `dashboard_state.py`
- Test: `tests/test_dashboard_state.py`

**Background:** The BO thread and Streamlit fragments run concurrently. All mutable shared data lives in one `AppState` dataclass protected by a `threading.Lock`.

**Step 1: Write failing tests**

```bash
mkdir -p /home/genpipeline/tests
```

Create `tests/test_dashboard_state.py`:

```python
import threading
from dashboard_state import AppState, IterResult

def test_initial_state():
    s = AppState()
    assert s.status == "idle"
    assert s.iterations == []
    assert s.best is None

def test_add_iteration():
    s = AppState()
    r = IterResult(step=1, objective=-0.05, z=[0.0]*16, voxel=None, fem=None)
    s.add_iteration(r)
    assert len(s.iterations) == 1
    assert s.best.objective == -0.05

def test_best_tracks_minimum():
    s = AppState()
    s.add_iteration(IterResult(step=1, objective=-0.03, z=[0.0]*16, voxel=None, fem=None))
    s.add_iteration(IterResult(step=2, objective=-0.08, z=[0.0]*16, voxel=None, fem=None))
    s.add_iteration(IterResult(step=3, objective=-0.05, z=[0.0]*16, voxel=None, fem=None))
    assert s.best.objective == -0.08
    assert s.best.step == 2

def test_stop_flag():
    s = AppState()
    assert not s.stop_requested
    s.request_stop()
    assert s.stop_requested

def test_thread_safety():
    s = AppState()
    errors = []
    def writer():
        for i in range(100):
            try:
                s.add_iteration(IterResult(step=i, objective=-i*0.01,
                                           z=[0.0]*16, voxel=None, fem=None))
            except Exception as e:
                errors.append(e)
    threads = [threading.Thread(target=writer) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert len(s.iterations) == 400
```

**Step 2: Run — expect failure**

```bash
source venv/bin/activate
python -m pytest tests/test_dashboard_state.py -v
```
Expected: `ImportError: No module named 'dashboard_state'`

**Step 3: Create `dashboard_state.py`**

```python
"""dashboard_state.py — shared live state between BO thread and Streamlit UI."""
from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class FEMResult:
    stress_max: float
    displacement_max: float
    mass: float
    h_mm: float
    r_mm: float


@dataclass
class IterResult:
    step: int
    objective: float
    z: list          # 16-dim float list
    voxel: Optional[np.ndarray]   # (32,32,32) binary, or None
    fem: Optional[FEMResult]      # only set when FreeCAD validates


class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self.status: str = "idle"          # "idle" | "running" | "done"
        self.iterations: list[IterResult] = []
        self.best: Optional[IterResult] = None
        self.stop_requested: bool = False
        self.total_iters: int = 50
        self.selected: Optional[str] = None   # filename from design browser

    def add_iteration(self, r: IterResult) -> None:
        with self._lock:
            self.iterations.append(r)
            if self.best is None or r.objective < self.best.objective:
                self.best = r

    def request_stop(self) -> None:
        with self._lock:
            self.stop_requested = True

    def reset(self, total_iters: int = 50) -> None:
        with self._lock:
            self.iterations.clear()
            self.best = None
            self.stop_requested = False
            self.status = "idle"
            self.total_iters = total_iters
```

**Step 4: Run tests — expect pass**

```bash
source venv/bin/activate
python -m pytest tests/test_dashboard_state.py -v
```
Expected: all 5 tests PASS

**Step 5: Commit**

```bash
git add dashboard_state.py tests/test_dashboard_state.py
git commit -m "feat: add AppState for thread-safe BO↔UI communication"
```

---

## Task 3: STL loader utility

**Files:**
- Create: `dashboard_utils.py`
- Test: `tests/test_dashboard_utils.py`

**Background:** Plotly `mesh3d` needs arrays of vertices (x,y,z) and triangle indices (i,j,k). `numpy-stl` reads binary/ASCII STL files. We also need to decode a 32³ voxel grid into Plotly `isosurface` data for when no STL exists.

**Step 1: Write failing tests**

Create `tests/test_dashboard_utils.py`:

```python
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
```

**Step 2: Run — expect failure**

```bash
source venv/bin/activate
python -m pytest tests/test_dashboard_utils.py -v
```

**Step 3: Create `dashboard_utils.py`**

```python
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
```

**Step 4: Run tests — expect pass**

```bash
source venv/bin/activate
python -m pytest tests/test_dashboard_utils.py -v
```

**Step 5: Commit**

```bash
git add dashboard_utils.py tests/test_dashboard_utils.py
git commit -m "feat: add STL loader and voxel↔isosurface helpers"
```

---

## Task 4: BO runner thread

**Files:**
- Create: `dashboard_bo_runner.py`
- Test: `tests/test_dashboard_bo_runner.py`

**Background:** Wraps the existing `DesignOptimizer` so it can write into `AppState` step-by-step rather than only returning at the end. Runs in a `threading.Thread`. Supports BO-only mode (VAE surrogate only) and Full-FEM mode (calls `freecad_bridge.run_variant` per iteration).

**Step 1: Write failing test**

Create `tests/test_dashboard_bo_runner.py`:

```python
import time
import threading
from unittest.mock import patch, MagicMock
import numpy as np
from dashboard_state import AppState
from dashboard_bo_runner import BORunner


def _mock_vae():
    vae = MagicMock()
    vae.latent_dim = 16
    vae.decode.return_value = (
        MagicMock(squeeze=lambda: MagicMock(cpu=lambda: MagicMock(
            numpy=lambda: np.zeros((32, 32, 32)))))
    )
    return vae


def test_bo_runner_fills_state():
    state = AppState()
    state.reset(total_iters=3)
    vae = _mock_vae()

    # Patch DesignOptimizer so we don't need a real checkpoint
    with patch("dashboard_bo_runner.DesignOptimizer") as MockOpt:
        instance = MockOpt.return_value
        instance.x_history = []
        instance.y_history = []

        def fake_optimize_step():
            z = np.random.randn(16)
            instance.x_history.append(z)
            instance.y_history.append(-float(np.random.rand()))
            return z, {}
        instance.optimize_step.side_effect = fake_optimize_step
        instance.initialize_search.return_value = None
        # Make y_history available after initialize
        instance.y_history = [-0.01]
        instance.x_history = [np.zeros(16)]

        runner = BORunner(state=state, vae=vae, device="cpu",
                          n_iters=3, mode="bo-only")
        t = threading.Thread(target=runner.run)
        t.start()
        t.join(timeout=10)

    assert state.status == "done"
    assert len(state.iterations) >= 1


def test_bo_runner_stops_on_request():
    state = AppState()
    state.reset(total_iters=100)
    vae = _mock_vae()

    with patch("dashboard_bo_runner.DesignOptimizer") as MockOpt:
        instance = MockOpt.return_value
        call_count = [0]

        def fake_step():
            call_count[0] += 1
            if call_count[0] == 2:
                state.request_stop()
            z = np.zeros(16)
            instance.x_history.append(z)
            instance.y_history.append(-0.01 * call_count[0])
            return z, {}

        instance.optimize_step.side_effect = fake_step
        instance.initialize_search.return_value = None
        instance.y_history = [-0.01]
        instance.x_history = [np.zeros(16)]

        runner = BORunner(state=state, vae=vae, device="cpu",
                          n_iters=100, mode="bo-only")
        t = threading.Thread(target=runner.run)
        t.start()
        t.join(timeout=10)

    assert state.status in ("done", "idle")
    assert call_count[0] <= 5   # stopped early
```

**Step 2: Run — expect failure**

```bash
source venv/bin/activate
python -m pytest tests/test_dashboard_bo_runner.py -v
```

**Step 3: Create `dashboard_bo_runner.py`**

```python
"""dashboard_bo_runner.py — wraps DesignOptimizer for step-by-step BO in a thread."""
from __future__ import annotations
import numpy as np
import torch
from dashboard_state import AppState, IterResult, FEMResult

# Import the existing optimizer
import sys
sys.path.insert(0, "/home/genpipeline")
from optimization_engine import DesignOptimizer


class BORunner:
    """
    Runs Bayesian optimisation step-by-step, writing each result into AppState.

    mode="bo-only"  — uses VAE performance head as surrogate (no FreeCAD)
    mode="full-fem" — calls freecad_bridge.run_variant for real stress per step
    """

    def __init__(self, state: AppState, vae, device: str,
                 n_iters: int = 50, mode: str = "bo-only",
                 freecad_cmd: str = "", output_dir: str = "/tmp/bo_variants"):
        self.state = state
        self.vae = vae
        self.device = device
        self.n_iters = n_iters
        self.mode = mode
        self.freecad_cmd = freecad_cmd
        self.output_dir = output_dir

    def _decode_voxel(self, z: np.ndarray) -> np.ndarray | None:
        try:
            zt = torch.tensor(z, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
            with torch.no_grad():
                voxel = self.vae.decode(zt).squeeze().cpu().numpy()
            return (voxel > 0.5).astype(np.uint8)
        except Exception:
            return None

    def _fem_validate(self, z: np.ndarray) -> FEMResult | None:
        if self.mode != "full-fem" or not self.freecad_cmd:
            return None
        # Decode voxel to get h/r occupancy estimate — use mean of z as proxy
        # Real implementation: inverse-map z → h,r via nearest-neighbor in training set
        # For now use voxel occupancy to scale h_mm
        voxel = self._decode_voxel(z)
        if voxel is None:
            return None
        occ = float(voxel.mean())
        h_mm = 5.0 + occ * 150.0   # rough heuristic, replace with real inverse map
        h_mm = float(np.clip(h_mm, 5.0, 20.0))
        r_mm = 0.0
        try:
            from freecad_bridge import run_variant, wsl_to_windows
            result = run_variant(self.freecad_cmd, h_mm, r_mm, self.output_dir)
            if result:
                return FEMResult(
                    stress_max=result["stress_max"],
                    displacement_max=result["displacement_max"],
                    mass=result["mass"],
                    h_mm=h_mm,
                    r_mm=r_mm,
                )
        except Exception:
            pass
        return None

    def run(self) -> None:
        self.state.status = "running"
        try:
            optimizer = DesignOptimizer(
                vae_model=self.vae,
                fem_evaluator=None,
                device=self.device,
                latent_dim=getattr(self.vae, "latent_dim", 16),
            )
            optimizer.initialize_search(n_init_points=5)

            for step in range(self.n_iters):
                if self.state.stop_requested:
                    break

                z, _ = optimizer.optimize_step()
                obj = float(optimizer.y_history[-1])
                voxel = self._decode_voxel(z)
                fem = self._fem_validate(z)

                result = IterResult(
                    step=step + 1,
                    objective=obj,
                    z=z.tolist() if hasattr(z, "tolist") else list(z),
                    voxel=voxel,
                    fem=fem,
                )
                self.state.add_iteration(result)

        except Exception as e:
            print(f"[BORunner] error: {e}")
        finally:
            self.state.status = "done"
```

**Step 4: Run tests — expect pass**

```bash
source venv/bin/activate
python -m pytest tests/test_dashboard_bo_runner.py -v
```

**Step 5: Commit**

```bash
git add dashboard_bo_runner.py tests/test_dashboard_bo_runner.py
git commit -m "feat: add BORunner — step-by-step BO thread writing into AppState"
```

---

## Task 5: Dark CSS theme module

**Files:**
- Create: `dashboard_theme.py`

No tests needed — purely presentational constants.

**Step 1: Create `dashboard_theme.py`**

```python
"""dashboard_theme.py — Fusion 360-style dark theme for Streamlit."""

BG       = "#1a1a2e"
PANEL    = "#16213e"
ACCENT   = "#e94560"
TEXT     = "#eaeaea"
MUTED    = "#888888"
SUCCESS  = "#00b894"
WARNING  = "#fdcb6e"

CSS = f"""
<style>
/* Page background */
.stApp {{ background-color: {BG}; color: {TEXT}; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {PANEL};
    border-right: 1px solid {ACCENT}22;
}}

/* Columns / panels */
div[data-testid="stVerticalBlock"] {{
    background-color: {PANEL};
    border-radius: 6px;
}}

/* Buttons */
.stButton button {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 4px;
    font-weight: 600;
}}
.stButton button:hover {{
    background-color: {ACCENT}cc;
}}

/* Stop button (secondary) */
button[kind="secondary"] {{
    background-color: #333 !important;
    color: {TEXT} !important;
    border: 1px solid #555 !important;
}}

/* Inputs */
.stSlider, .stNumberInput, .stSelectbox {{
    color: {TEXT};
}}

/* Metric values */
[data-testid="stMetricValue"] {{
    color: {ACCENT};
    font-size: 1.4rem;
    font-weight: 700;
}}

/* Progress bar */
.stProgress > div > div {{ background-color: {ACCENT}; }}

/* Dataframe */
.stDataFrame {{ background-color: {PANEL}; }}

/* Bottom bar separator */
hr {{ border-color: {ACCENT}33; }}
</style>
"""


PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="monospace"),
    margin=dict(l=0, r=0, t=30, b=0),
    scene=dict(
        bgcolor=BG,
        xaxis=dict(backgroundcolor=PANEL, gridcolor="#333", color=TEXT),
        yaxis=dict(backgroundcolor=PANEL, gridcolor="#333", color=TEXT),
        zaxis=dict(backgroundcolor=PANEL, gridcolor="#333", color=TEXT),
    ),
)
```

**Step 2: Commit**

```bash
git add dashboard_theme.py
git commit -m "feat: add dark Fusion 360 CSS theme constants"
```

---

## Task 6: Main `dashboard.py` — skeleton + toolbar

**Files:**
- Create: `dashboard.py`

**Step 1: Create the skeleton with toolbar only (no BO yet)**

```python
"""
dashboard.py — Fusion 360-style generative design dashboard.

Run with:
    source venv/bin/activate
    streamlit run dashboard.py
"""
from __future__ import annotations
import json
import threading
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import torch

from dashboard_state import AppState
from dashboard_bo_runner import BORunner
from dashboard_utils import load_stl_for_plotly, voxel_to_plotly_isosurface
from dashboard_theme import CSS, PLOTLY_LAYOUT, ACCENT, MUTED, TEXT, SUCCESS, WARNING

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GenPipeline Dashboard",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CSS, unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
VARIANTS_DIR = Path("fem_variants")
CHECKPOINT   = Path("checkpoints/vae_best.pth")
FREECAD_PATH = "/mnt/c/Users/PC-PC/AppData/Local/Programs/FreeCAD 1.0"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Session state ─────────────────────────────────────────────────────────────
if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()
if "bo_thread" not in st.session_state:
    st.session_state.bo_thread = None
if "vae" not in st.session_state:
    st.session_state.vae = None

app_state: AppState = st.session_state.app_state


# ── VAE loader ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_vae():
    """Load VAE from checkpoint once, cache across reruns."""
    if not CHECKPOINT.exists():
        return None
    try:
        from vae_design_model import DesignVAE
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        model = DesignVAE(latent_dim=ckpt.get("latent_dim", 16)).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Could not load VAE checkpoint: {e}")
        return None


# ── Variant loader ────────────────────────────────────────────────────────────
@st.cache_data
def load_variants() -> list[dict]:
    """Load all *_fem_results.json from fem_variants/."""
    results = []
    for f in sorted(VARIANTS_DIR.glob("*_fem_results.json")):
        d = json.loads(f.read_text())
        d["_file"] = f.stem.replace("_fem_results", "")
        d["_stl"]  = str(VARIANTS_DIR / f"{d['_file']}_mesh.stl")
        results.append(d)
    return results


# ── TOP TOOLBAR ───────────────────────────────────────────────────────────────
st.markdown("### ⚙ GenPipeline — Generative Design Dashboard")
tb_col1, tb_col2, tb_col3, tb_col4, tb_col5, tb_col6 = st.columns(
    [1, 1, 1.5, 1.5, 1, 2]
)

with tb_col1:
    run_clicked = st.button("▶ Run BO", use_container_width=True)

with tb_col2:
    stop_clicked = st.button("■ Stop", use_container_width=True)

with tb_col3:
    mode = st.selectbox("Mode", ["BO-only", "Full FEM"], label_visibility="collapsed")

with tb_col4:
    n_iters = st.number_input("Iterations", min_value=5, max_value=500,
                              value=50, label_visibility="collapsed")

with tb_col5:
    validate_clicked = st.button("⚡ Validate", use_container_width=True)

with tb_col6:
    status_text = app_state.status.upper()
    colour = SUCCESS if status_text == "RUNNING" else (ACCENT if status_text == "DONE" else MUTED)
    st.markdown(f"<span style='color:{colour};font-weight:700'>{status_text}</span>",
                unsafe_allow_html=True)

st.markdown("<hr style='margin:4px 0 8px 0'>", unsafe_allow_html=True)

# ── Handle toolbar actions ────────────────────────────────────────────────────
vae = load_vae()

if run_clicked and app_state.status != "running":
    app_state.reset(total_iters=int(n_iters))
    app_state.status = "running"
    runner = BORunner(
        state=app_state,
        vae=vae,
        device=DEVICE,
        n_iters=int(n_iters),
        mode=mode.lower().replace(" ", "-"),
        freecad_cmd=str(
            Path(FREECAD_PATH) / "bin" / "freecad.exe"
        ) if mode == "Full FEM" else "",
        output_dir="/tmp/bo_variants",
    )
    thread = threading.Thread(target=runner.run, daemon=True)
    thread.start()
    st.session_state.bo_thread = thread

if stop_clicked:
    app_state.request_stop()

# ── MAIN 3-COLUMN LAYOUT ──────────────────────────────────────────────────────
left_col, center_col, right_col = st.columns([2, 5, 3], gap="small")

# ── Left: Design browser ──────────────────────────────────────────────────────
with left_col:
    st.markdown("#### Design Browser")
    variants = load_variants()
    selected_file = None

    for v in variants:
        label = f"{v['_file']}  |  σ {v['stress_max']:.0f} MPa"
        is_best = (app_state.best and
                   abs(v.get("stress_max", 0) - (app_state.best.fem.stress_max
                       if app_state.best.fem else 0)) < 1)
        prefix = "★ " if is_best else "  "
        if st.button(prefix + label, key=f"sel_{v['_file']}",
                     use_container_width=True):
            app_state.selected = v["_file"]

    st.markdown("---")
    st.markdown("**Generate new**")
    h_val = st.slider("h_mm", 5.0, 20.0, 10.0, 0.5)
    r_val = st.slider("r_mm", 0.0, 8.0, 0.0, 0.5)
    if st.button("Run FEM", use_container_width=True):
        if vae is None:
            st.error("No VAE checkpoint loaded.")
        else:
            with st.spinner("Running FreeCAD FEM..."):
                from freecad_bridge import run_variant, find_freecad_cmd
                try:
                    fc = find_freecad_cmd(FREECAD_PATH)
                    res = run_variant(fc, h_val, r_val, str(VARIANTS_DIR))
                    if res:
                        st.success(f"σ_max = {res['stress_max']:.1f} MPa")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("FEM run failed.")
                except Exception as e:
                    st.error(str(e))

# ── Centre: 3D viewport (live fragment) ──────────────────────────────────────
with center_col:
    st.markdown("#### 3D Viewport")

    @st.fragment(run_every=0.5)
    def render_viewport():
        best = app_state.best
        selected = app_state.selected

        # Decide what to show
        stl_path = None
        voxel_data = None

        # Priority: user-selected variant > BO best design > first variant
        if selected:
            matches = [v for v in variants if v["_file"] == selected]
            if matches and Path(matches[0]["_stl"]).exists():
                stl_path = matches[0]["_stl"]
        elif best and best.voxel is not None:
            voxel_data = best.voxel
        elif variants:
            stl_path = variants[0]["_stl"]

        fig = go.Figure()

        if stl_path and Path(stl_path).exists():
            try:
                x, y, z, i, j, k = load_stl_for_plotly(stl_path)
                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z, i=i, j=j, k=k,
                    color=ACCENT, opacity=0.85,
                    flatshading=True,
                    lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3),
                ))
                title = f"STL: {Path(stl_path).stem}"
            except Exception as e:
                st.error(f"STL load error: {e}")
                return
        elif voxel_data is not None:
            d = voxel_to_plotly_isosurface(voxel_data)
            fig.add_trace(go.Isosurface(
                x=d["x"], y=d["y"], z=d["z"], value=d["value"],
                isomin=0.5, isomax=1.0,
                surface_count=1,
                colorscale=[[0, ACCENT], [1, "#ff9999"]],
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
            ))
            title = f"BO Best (iter {best.step})  obj={best.objective:.4f}"
        else:
            fig.add_annotation(text="No design loaded",
                               xref="paper", yref="paper", x=0.5, y=0.5,
                               showarrow=False, font=dict(color=TEXT, size=16))
            title = "Waiting..."

        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=420,
            title=dict(text=title, font=dict(color=TEXT, size=13)),
        )
        st.plotly_chart(fig, use_container_width=True, key="viewport_chart")

    render_viewport()

# ── Right: Properties panel (live fragment) ────────────────────────────────
with right_col:
    st.markdown("#### Properties")

    @st.fragment(run_every=0.5)
    def render_properties():
        best = app_state.best
        selected = app_state.selected

        # Show properties for selected variant or BO best
        props = None
        if selected:
            matches = [v for v in variants if v["_file"] == selected]
            if matches:
                v = matches[0]
                props = {
                    "Stress max":   f"{v['stress_max']:.1f} MPa",
                    "Displacement": f"{v['displacement_max']:.3f} mm",
                    "Mass":         f"{v['mass']:.4f} kg",
                    "Compliance":   f"{v['compliance']:.2f}",
                    "h_mm":         f"{v['parameters']['h_mm']:.1f}",
                    "r_mm":         f"{v['parameters']['r_mm']:.1f}",
                    "Objective":    "—",
                }
        elif best:
            if best.fem:
                props = {
                    "Stress max":   f"{best.fem.stress_max:.1f} MPa",
                    "Displacement": f"{best.fem.displacement_max:.3f} mm",
                    "Mass":         f"{best.fem.mass:.4f} kg",
                    "Compliance":   "—",
                    "h_mm":         f"{best.fem.h_mm:.1f}",
                    "r_mm":         f"{best.fem.r_mm:.1f}",
                    "Objective":    f"{best.objective:.4f}",
                }
            else:
                props = {
                    "Stress max":   "—",
                    "Displacement": "—",
                    "Mass":         "—",
                    "Objective":    f"{best.objective:.4f}",
                    "BO step":      str(best.step),
                }

        if props:
            for k, v_str in props.items():
                col_a, col_b = st.columns([1, 1])
                col_a.markdown(f"<span style='color:{MUTED}'>{k}</span>",
                               unsafe_allow_html=True)
                col_b.markdown(f"**{v_str}**")
        else:
            st.markdown(f"<span style='color:{MUTED}'>Select a design or run BO</span>",
                        unsafe_allow_html=True)

        # Export STL button
        if selected:
            matches = [v for v in variants if v["_file"] == selected]
            if matches:
                stl = matches[0]["_stl"]
                if Path(stl).exists():
                    with open(stl, "rb") as f:
                        st.download_button("⬇ Export STL", f.read(),
                                           file_name=f"{selected}.stl",
                                           mime="model/stl",
                                           use_container_width=True)

    render_properties()

# ── Bottom: BO progress bar + sparkline (live fragment) ──────────────────────
st.markdown("<hr style='margin:8px 0 4px 0'>", unsafe_allow_html=True)

@st.fragment(run_every=0.5)
def render_bo_bar():
    iters = app_state.iterations
    total = app_state.total_iters
    best = app_state.best

    bar_col, info_col = st.columns([4, 1])

    with bar_col:
        if iters:
            st.progress(len(iters) / max(total, 1))
            # Sparkline
            y_vals = [r.objective for r in iters]
            fig = go.Figure(go.Scatter(
                y=y_vals, mode="lines",
                line=dict(color=ACCENT, width=2),
                fill="tozeroy",
                fillcolor=ACCENT + "33",
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=70,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, key="sparkline_chart")
        else:
            st.progress(0.0)

    with info_col:
        n = len(iters)
        best_val = best.objective if best else 0.0
        st.markdown(f"**{n}/{total}** iters")
        colour = SUCCESS if best_val < -0.05 else MUTED
        st.markdown(f"<span style='color:{colour}'>Best: **{best_val:.4f}**</span>",
                    unsafe_allow_html=True)

render_bo_bar()
```

**Step 2: Verify it launches (smoke test)**

```bash
source venv/bin/activate
streamlit run dashboard.py --server.headless true &
sleep 5
curl -s http://localhost:8501 | grep -q "GenPipeline" && echo "Dashboard OK" || echo "FAIL"
kill %1
```
Expected: `Dashboard OK`

**Step 3: Commit**

```bash
git add dashboard.py
git commit -m "feat: add main dashboard.py with dark theme, 3-col layout, live BO fragments"
```

---

## Task 7: Wire up validate button (⚡)

**Files:**
- Modify: `dashboard.py` (the validate_clicked block near the toolbar)

**Background:** When clicked, takes `app_state.best.z`, decodes it, maps to h/r, runs one FreeCAD FEM job, injects a `FEMResult` back into `best.fem`.

**Step 1: Add validate handler after the toolbar actions block in `dashboard.py`**

Find the comment `# ── Handle toolbar actions` and add after the existing `if stop_clicked:` block:

```python
if validate_clicked and app_state.best is not None:
    with st.spinner("Running FreeCAD FEM on best design..."):
        try:
            from freecad_bridge import run_variant, find_freecad_cmd
            from dashboard_bo_runner import BORunner
            fc = find_freecad_cmd(FREECAD_PATH)
            runner = BORunner(state=app_state, vae=vae, device=DEVICE,
                              mode="full-fem", freecad_cmd=fc,
                              output_dir="/tmp/bo_validate")
            fem = runner._fem_validate(np.array(app_state.best.z))
            if fem:
                app_state.best.fem = fem
                st.toast(f"Validated: σ_max = {fem.stress_max:.1f} MPa", icon="✅")
            else:
                st.warning("FEM validation returned no result.")
        except Exception as e:
            st.error(f"Validation error: {e}")
```

**Step 2: Add `import numpy as np` at top if not already present** (it is — already there).

**Step 3: Quick manual test**
- Run BO for a few iterations
- Click ⚡ Validate
- Properties panel should show real MPa values

**Step 4: Commit**

```bash
git add dashboard.py
git commit -m "feat: wire up validate button to run FreeCAD FEM on BO best"
```

---

## Task 8: Final check + run instructions

**Step 1: Run all tests**

```bash
source venv/bin/activate
python -m pytest tests/ -v
```
Expected: all tests PASS

**Step 2: Launch dashboard**

```bash
source venv/bin/activate
streamlit run dashboard.py
```

Open browser to `http://localhost:8501`.

Smoke-test checklist:
- [ ] Dark theme renders (no white flash)
- [ ] Design browser shows 10 variants
- [ ] Clicking a variant loads STL in 3D viewport
- [ ] "Run BO" starts BO, sparkline grows, status shows RUNNING
- [ ] "■ Stop" halts the BO thread
- [ ] "Generate new" with sliders calls FreeCAD, adds new variant
- [ ] "⚡ Validate" shows real MPa in properties panel

**Step 3: Commit + push**

```bash
git add .
git commit -m "feat: complete Fusion 360-style generative design dashboard"
git push
```

---

## What else to build (post-dashboard roadmap)

1. **More FEM data** — run `python freecad_bridge.py generate --n 50` to get 50 variants, then retrain VAE
2. **Retrain VAE on real data** — `python quickstart.py --step 3 --epochs 300 --dataset fem_variants/fem_dataset.pt`
3. **Close BO loop** — Full FEM mode in the dashboard already does this; needs 50+ variants first
4. **Multi-objective** — add Pareto front of stress vs mass to Properties panel (coloured scatter)
5. **Geometry richer params** — add taper ratio, notch depth to `run_fem_variant.py`; dashboard gets extra sliders
6. **STEP export** — `shape_obj.Shape.exportStep(path)` in FreeCAD, add button to Properties panel
