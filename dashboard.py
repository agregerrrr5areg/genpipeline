"""
dashboard.py — Generative design dashboard.

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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GenPipeline",
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

app_state: AppState = st.session_state.app_state


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_vae():
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
        st.warning(f"VAE load failed: {e}")
        return None


@st.cache_data
def load_variants() -> list[dict]:
    results = []
    for f in sorted(VARIANTS_DIR.glob("*_fem_results.json")):
        d = json.loads(f.read_text())
        d["_file"] = f.stem.replace("_fem_results", "")
        d["_stl"]  = str(VARIANTS_DIR / f"{d['_file']}_mesh.stl")
        results.append(d)
    return results


variants = load_variants()
vae = load_vae()

# ── Fragments (defined at module level — avoids column-context re-render bug) ─

@st.fragment(run_every=0.5)
def render_viewport():
    best     = app_state.best
    selected = app_state.selected

    stl_path   = None
    voxel_data = None

    if selected:
        matches = [v for v in variants if v["_file"] == selected]
        if matches and Path(matches[0]["_stl"]).exists():
            stl_path = matches[0]["_stl"]
    elif best and best.voxel is not None:
        voxel_data = best.voxel
    elif variants and Path(variants[0]["_stl"]).exists():
        stl_path = variants[0]["_stl"]

    fig   = go.Figure()
    title = ""

    if stl_path:
        try:
            x, y, z, i, j, k = load_stl_for_plotly(stl_path)
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k,
                color=ACCENT, opacity=0.9,
                flatshading=True,
            ))
            title = Path(stl_path).stem
        except Exception as e:
            st.error(f"STL error: {e}")
            return
    elif voxel_data is not None:
        d = voxel_to_plotly_isosurface(voxel_data)
        fig.add_trace(go.Isosurface(
            x=d["x"], y=d["y"], z=d["z"], value=d["value"],
            isomin=0.5, isomax=1.0, surface_count=1,
            colorscale=[[0, ACCENT], [1, "#ff9999"]],
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))
        title = f"BO iter {best.step}  obj={best.objective:.4f}"
    else:
        fig.add_annotation(text="Select a design from the browser",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=MUTED, size=14))

    fig.update_layout(**PLOTLY_LAYOUT, height=430,
                      title=dict(text=title, font=dict(color=MUTED, size=12)))
    st.plotly_chart(fig, use_container_width=True, key="viewport_chart")


@st.fragment(run_every=0.5)
def render_properties():
    best     = app_state.best
    selected = app_state.selected

    props = None
    stl_for_download = None

    if selected:
        matches = [v for v in variants if v["_file"] == selected]
        if matches:
            v = matches[0]
            props = {
                "Stress max":   f"{v['stress_max']:.1f} MPa",
                "Displacement": f"{v['displacement_max']:.3f} mm",
                "Mass":         f"{v['mass']:.4f} kg",
                "Compliance":   f"{v['compliance']:.2f}",
                "h":            f"{v['parameters']['h_mm']:.1f} mm",
                "r":            f"{v['parameters']['r_mm']:.1f} mm",
            }
            stl_for_download = (matches[0]["_stl"], selected)
    elif best:
        if best.fem:
            props = {
                "Stress max":   f"{best.fem.stress_max:.1f} MPa",
                "Displacement": f"{best.fem.displacement_max:.3f} mm",
                "Mass":         f"{best.fem.mass:.4f} kg",
                "Objective":    f"{best.objective:.4f}",
                "h":            f"{best.fem.h_mm:.1f} mm",
                "r":            f"{best.fem.r_mm:.1f} mm",
            }
        else:
            props = {
                "Objective": f"{best.objective:.4f}",
                "BO step":   str(best.step),
            }

    if props:
        for k, val in props.items():
            a, b = st.columns([1, 1])
            a.markdown(f"<span style='color:{MUTED};font-size:0.85rem'>{k}</span>",
                       unsafe_allow_html=True)
            b.markdown(f"**{val}**")
    else:
        st.markdown(f"<span style='color:{MUTED}'>Select a design or run BO</span>",
                    unsafe_allow_html=True)

    if stl_for_download:
        path, name = stl_for_download
        if Path(path).exists():
            st.markdown("")
            with open(path, "rb") as f:
                st.download_button("Export STL", f.read(),
                                   file_name=f"{name}.stl",
                                   mime="model/stl",
                                   use_container_width=True)


@st.fragment(run_every=0.5)
def render_bo_bar():
    iters = app_state.iterations
    total = app_state.total_iters
    best  = app_state.best

    bar_col, info_col = st.columns([4, 1])

    with bar_col:
        st.progress(len(iters) / max(total, 1))
        if iters:
            y_vals = [r.objective for r in iters]
            fig = go.Figure(go.Scatter(
                y=y_vals, mode="lines",
                line=dict(color=ACCENT, width=2),
                fill="tozeroy", fillcolor=ACCENT + "33",
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT, height=60,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, key="sparkline_chart")

    with info_col:
        best_val = best.objective if best else 0.0
        st.markdown(f"**{len(iters)}/{total}**")
        colour = SUCCESS if best_val < -0.05 else MUTED
        st.markdown(f"<span style='color:{colour}'>**{best_val:.4f}**</span>",
                    unsafe_allow_html=True)


# ── TOP TOOLBAR ───────────────────────────────────────────────────────────────
st.markdown("**GenPipeline** — Generative Design")
tb1, tb2, tb3, tb4, tb5, tb6 = st.columns([1, 1, 1.5, 1.5, 1, 2])

with tb1:
    run_clicked = st.button("Run BO", use_container_width=True)
with tb2:
    stop_clicked = st.button("Stop", use_container_width=True)
with tb3:
    mode = st.selectbox("Mode", ["BO-only", "Full FEM"], label_visibility="collapsed")
with tb4:
    n_iters = st.number_input("Iterations", min_value=5, max_value=500,
                              value=50, label_visibility="collapsed")
with tb5:
    validate_clicked = st.button("Validate", use_container_width=True)
with tb6:
    status_text = app_state.status.upper()
    colour = SUCCESS if status_text == "RUNNING" else (ACCENT if status_text == "DONE" else MUTED)
    st.markdown(f"<span style='color:{colour};font-weight:700'>{status_text}</span>",
                unsafe_allow_html=True)

st.markdown("<hr style='margin:4px 0 8px 0'>", unsafe_allow_html=True)

# ── Toolbar actions ───────────────────────────────────────────────────────────
if run_clicked and app_state.status != "running":
    app_state.reset(total_iters=int(n_iters))
    app_state.status = "running"
    runner = BORunner(
        state=app_state, vae=vae, device=DEVICE,
        n_iters=int(n_iters),
        mode=mode.lower().replace(" ", "-"),
        freecad_cmd=str(Path(FREECAD_PATH) / "bin" / "freecad.exe") if mode == "Full FEM" else "",
        output_dir="/tmp/bo_variants",
    )
    threading.Thread(target=runner.run, daemon=True).start()

if stop_clicked:
    app_state.request_stop()

if validate_clicked and app_state.best is not None:
    with st.spinner("Running FreeCAD FEM on best design..."):
        try:
            from freecad_bridge import find_freecad_cmd
            fc = find_freecad_cmd(FREECAD_PATH)
            runner = BORunner(state=app_state, vae=vae, device=DEVICE,
                              mode="full-fem", freecad_cmd=fc,
                              output_dir="/tmp/bo_validate")
            fem = runner._fem_validate(np.array(app_state.best.z))
            if fem:
                app_state.best.fem = fem
                st.toast(f"Validated: {fem.stress_max:.1f} MPa")
            else:
                st.warning("FEM validation returned no result.")
        except Exception as e:
            st.error(f"Validation error: {e}")

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
left_col, center_col, right_col = st.columns([2, 5, 3], gap="small")

# Left: Design browser
with left_col:
    st.markdown("**Designs**")
    for v in variants:
        label = f"{v['_file']}  {v['stress_max']:.0f} MPa"
        if st.button(label, key=f"sel_{v['_file']}", use_container_width=True):
            app_state.selected = v["_file"]

    st.markdown("---")
    st.markdown("**Generate**")
    h_val = st.slider("h mm", 5.0, 20.0, 10.0, 0.5)
    r_val = st.slider("r mm", 0.0, 8.0, 0.0, 0.5)
    if st.button("Run FEM", use_container_width=True):
        with st.spinner("Running FreeCAD FEM..."):
            try:
                from freecad_bridge import run_variant, find_freecad_cmd
                fc = find_freecad_cmd(FREECAD_PATH)
                res = run_variant(fc, h_val, r_val, str(VARIANTS_DIR))
                if res:
                    st.success(f"{res['stress_max']:.1f} MPa")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("FEM run failed.")
            except Exception as e:
                st.error(str(e))

# Centre: 3D viewport
with center_col:
    render_viewport()

# Right: Properties
with right_col:
    st.markdown("**Properties**")
    render_properties()

# Bottom: BO bar
st.markdown("<hr style='margin:8px 0 4px 0'>", unsafe_allow_html=True)
render_bo_bar()
