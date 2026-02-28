# Dashboard Design — Generative Design Pipeline
**Date**: 2026-02-28
**Status**: Approved

## Goal
A Fusion 360-style visual interface for running and watching the Bayesian optimisation loop live, exploring generated designs, and optionally validating top candidates with real FreeCAD FEM.

## Audience
Personal exploration tool (no auth, no deployment required).

## Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  [▶ Run BO]  [■ Stop]  Mode: [BO-only | Full FEM]  Iters: [50]  [⚡] │  top toolbar
├─────────────────┬────────────────────────────┬────────────────────────┤
│  Design Browser │   3D VIEWPORT (Plotly)      │  Properties Panel      │
│  ─────────────  │   (current best design,     │  ─────────────────────  │
│  ● h14.6 r0.2   │    interactive rotate/zoom) │  Stress:  99.3 MPa     │
│  ● h5.4  r1.6   │                             │  Disp:    0.22 mm      │
│  ● h18.4 r0.7   │   [voxel/STL mesh]          │  Mass:    0.23 kg      │
│  ...            │                             │  Objective: -0.106     │
│  ─────────────  │                             │  ─────────────────────  │
│  h: [──●──]     │                             │  h_mm:  14.6           │
│  r: [●────]     │                             │  r_mm:  0.2            │
│  [Run FEM]      │                             │  [Export STL]          │
├─────────────────┴────────────────────────────┴────────────────────────┤
│  BO Timeline ████████████░░░░░░  Best: -0.106  Iter: 8/50            │  bottom bar
│  Objective sparkline (live, scrolls right)                            │
└──────────────────────────────────────────────────────────────────────┘
```

**Theme**: Dark (`#1a1a2e` bg, `#e94560` accent, `#16213e` panels), Fusion 360-inspired.

## Components

### 1. Top Toolbar
- **Run / Stop** buttons toggle the BO background thread
- **Mode selector**: `BO-only` (uses VAE surrogate) or `Full FEM` (calls FreeCAD per iteration)
- **Iteration count** number input (default 50)
- **⚡ Validate** button: run FreeCAD FEM on the current best design, inject results

### 2. Design Browser (left column, ~20% width)
- Scrollable list of all variants in `fem_variants/` — click to load into viewport
- **Manual generator**: h_mm slider (5–20), r_mm slider (0–8), [Run FEM] button
- Highlights the "current best" from BO in accent colour

### 3. 3D Viewport (centre column, ~50% width)
- **Plotly `mesh3d`** when STL is available (triangulated surface, coloured by Z-height or uniform)
- **Plotly `isosurface`** fallback when only a 32³ voxel grid is available
- Rotatable / zoomable via Plotly's built-in controls
- Live fragment (`@st.fragment(run_every=0.5)`) — re-renders when best design changes

### 4. Properties Panel (right column, ~30% width)
- Stress max, displacement max, mass, objective score
- h_mm, r_mm parameters
- **[Export STL]** button — writes best STL to `optimization_results/`
- Live fragment (`@st.fragment(run_every=0.5)`)

### 5. Bottom BO Bar
- Progress bar: `current_iter / total_iters`
- Live sparkline chart (Plotly `scatter`) of objective history, scrolls as BO runs
- Best objective value label
- Live fragment (`@st.fragment(run_every=0.5)`)

## Data Flow

```
AppState (threading.Lock-guarded dataclass)
  ├── iterations: list[IterResult]   # each BO step
  ├── status: "idle" | "running" | "done"
  ├── best_design: IterResult | None
  └── selected_design: str | None    # from browser click

BO Thread (threading.Thread)
  └── reads AppState.status to check for stop
  └── writes AppState.iterations + best_design each step

Streamlit fragments (0.5s)
  └── read AppState (no lock needed for reads — Python GIL)
  └── render viewport, properties, sparkline
```

## Live Updates
`@st.fragment(run_every=0.5)` on three fragments:
- `render_viewport()` — redraws 3D if best design changed
- `render_properties()` — updates metrics
- `render_bo_bar()` — updates sparkline + progress

Static parts (toolbar, browser) only re-render on user interaction.

## Tech Stack
- `streamlit >= 1.35` (for `@st.fragment`)
- `plotly` (3D mesh + charts — already in venv)
- `numpy`, `torch` (already in venv)
- `stl` from `numpy-stl` for reading STL files
- No new backend processes needed

## What Else To Build (roadmap)
1. **More FEM data**: generate 50+ variants for retraining VAE on real physics
2. **Retrain VAE**: on `fem_variants/` dataset instead of old synthetic data
3. **Close BO loop**: real FreeCAD oracle per iteration (Full FEM mode above covers this)
4. **Multi-objective**: Pareto front of stress vs mass in the Properties panel
5. **More geometry params**: taper, notches, I-beam profile → richer design space
6. **STEP export**: for CAM/manufacturing via FreeCAD Part.Shape.exportStep()
