# Sim Settings Panel Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a collapsible Settings expander in the dashboard left column that lets the user configure material, load, objective weights, and structural constraints — saved to `sim_config.json` and wired into FreeCAD and the BO objective.

**Architecture:** `sim_config.json` is the single source of truth. `dashboard.py` reads it on load and writes it on Save. `run_fem_variant.py` reads material + force from the config. `optimization_engine.py` objective uses config weights + constraint penalties. `dashboard_bo_runner.py` passes config to the optimizer at run time.

**Tech Stack:** Streamlit, existing `dashboard.py`, `freecad_scripts/run_fem_variant.py`, `optimization_engine.py`, `dashboard_bo_runner.py`. All at `/home/genpipeline/`. Activate venv: `source venv/bin/activate`.

---

## Task 1: `sim_config.py` — load/save helpers + defaults

**Files:**
- Create: `sim_config.py`
- Test: `tests/test_sim_config.py`

**Step 1: Write failing tests**

Create `tests/test_sim_config.py`:
```python
import json
from pathlib import Path
import pytest
from sim_config import load_config, save_config, PRESETS, default_config

def test_default_config_has_required_keys():
    cfg = default_config()
    for k in ["material","E_mpa","poisson","density_kg_m3","yield_mpa",
               "force_n","safety_factor","max_stress_mpa","max_disp_mm",
               "w_stress","w_compliance","w_mass"]:
        assert k in cfg, f"missing key: {k}"

def test_presets_contain_steel():
    assert "Steel" in PRESETS
    assert PRESETS["Steel"]["E_mpa"] == 210000

def test_save_and_load_roundtrip(tmp_path):
    cfg = default_config()
    cfg["force_n"] = 2000
    p = tmp_path / "sim_config.json"
    save_config(cfg, str(p))
    loaded = load_config(str(p))
    assert loaded["force_n"] == 2000

def test_load_missing_file_returns_default(tmp_path):
    p = tmp_path / "nonexistent.json"
    cfg = load_config(str(p))
    assert cfg["material"] == "Steel"

def test_max_stress_auto_computed():
    cfg = default_config()
    # yield=250, SF=1.5 → max_stress = 250/1.5 ≈ 166.7
    assert abs(cfg["max_stress_mpa"] - 250/1.5) < 0.1
```

**Step 2: Run — expect ImportError**
```bash
cd /home/genpipeline && source venv/bin/activate
python -m pytest tests/test_sim_config.py -v
```

**Step 3: Create `sim_config.py`**
```python
"""sim_config.py — load/save sim_config.json with presets and defaults."""
from __future__ import annotations
import json
from pathlib import Path

PRESETS: dict[str, dict] = {
    "Steel":     {"E_mpa": 210000, "poisson": 0.30, "density_kg_m3": 7900, "yield_mpa": 250},
    "Aluminium": {"E_mpa":  70000, "poisson": 0.33, "density_kg_m3": 2700, "yield_mpa": 270},
    "Titanium":  {"E_mpa": 114000, "poisson": 0.34, "density_kg_m3": 4430, "yield_mpa": 880},
}

CONFIG_PATH = Path(__file__).parent / "sim_config.json"


def default_config() -> dict:
    mat = PRESETS["Steel"]
    return {
        "material":        "Steel",
        "E_mpa":           mat["E_mpa"],
        "poisson":         mat["poisson"],
        "density_kg_m3":   mat["density_kg_m3"],
        "yield_mpa":       mat["yield_mpa"],
        "force_n":         1000.0,
        "safety_factor":   1.5,
        "max_stress_mpa":  round(mat["yield_mpa"] / 1.5, 2),
        "max_disp_mm":     1.0,
        "w_stress":        1.0,
        "w_compliance":    0.1,
        "w_mass":          0.01,
    }


def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        return default_config()
    try:
        return json.loads(p.read_text())
    except Exception:
        return default_config()


def save_config(cfg: dict, path: str | None = None) -> None:
    p = Path(path) if path else CONFIG_PATH
    p.write_text(json.dumps(cfg, indent=2))
```

**Step 4: Run tests — expect PASS**
```bash
cd /home/genpipeline && source venv/bin/activate
python -m pytest tests/test_sim_config.py -v
```

**Step 5: Commit**
```bash
cd /home/genpipeline
git add sim_config.py tests/test_sim_config.py
git commit -m "feat: add sim_config load/save with material presets"
```

---

## Task 2: Wire config into `run_fem_variant.py`

**Files:**
- Modify: `freecad_scripts/run_fem_variant.py` (lines ~92-137, material + force blocks)

**Context:** Currently material properties and force are hardcoded. The config file is passed as a `.cfg` JSON argument. We need to read optional keys from it and fall back to defaults.

**Step 1: Read the current file**
```bash
grep -n "YoungsModulus\|PoissonRatio\|Density\|force\|1000 N" freecad_scripts/run_fem_variant.py
```

**Step 2: Modify `make_material` call in `run_fem` to use config values**

In `freecad_scripts/run_fem_variant.py`, find the `run_fem` function signature and add a `cfg` parameter. Then replace the hardcoded material dict and force quantity:

Replace:
```python
def run_fem(doc, shape_obj, h_mm, r_mm, output_dir):
```
With:
```python
def run_fem(doc, shape_obj, h_mm, r_mm, output_dir, cfg=None):
    if cfg is None:
        cfg = {}
```

Replace the material properties block:
```python
        "YoungsModulus": "210000 MPa",
        "PoissonRatio":  "0.30",
        "Density":       "7900 kg/m^3",
```
With:
```python
        "YoungsModulus": f"{cfg.get('E_mpa', 210000)} MPa",
        "PoissonRatio":  str(cfg.get('poisson', 0.30)),
        "Density":       f"{cfg.get('density_kg_m3', 7900)} kg/m^3",
```

Replace the force line:
```python
    force.Force = App.Units.Quantity("1000 N")
```
With:
```python
    force.Force = App.Units.Quantity(f"{cfg.get('force_n', 1000)} N")
```

**Step 3: Pass cfg from `main()`**

Find `run_fem(doc, shape_obj, h_mm, r_mm, out)` in `main()` and change to:
```python
run_fem(doc, shape_obj, h_mm, r_mm, out, cfg=cfg)
```

(The `cfg` dict is already loaded via `_load_config()` in `main()`.)

**Step 4: Smoke test** (no FreeCAD needed — just import check)
```bash
cd /home/genpipeline && source venv/bin/activate
python -c "import ast, sys; ast.parse(open('freecad_scripts/run_fem_variant.py').read()); print('syntax ok')"
```

**Step 5: Commit**
```bash
cd /home/genpipeline
git add freecad_scripts/run_fem_variant.py
git commit -m "feat: run_fem_variant reads material+force from sim config"
```

---

## Task 3: Wire config into BO objective (`optimization_engine.py`)

**Files:**
- Modify: `optimization_engine.py` — `objective_function` method
- Test: `tests/test_objective_weights.py`

**Context:** Currently objective = `stress + 0.1 * compliance` hardcoded. We need it to use weights and constraint penalties from sim_config.

**Step 1: Write failing test**

Create `tests/test_objective_weights.py`:
```python
import sys
sys.path.insert(0, "/home/genpipeline")
from unittest.mock import MagicMock, patch
import numpy as np

def _make_optimizer(cfg):
    from optimization_engine import DesignOptimizer
    vae = MagicMock()
    vae.latent_dim = 16
    vae.performance_predictor = MagicMock(
        return_value=MagicMock(detach=lambda: MagicMock(
            cpu=lambda: MagicMock(numpy=lambda: np.array([[50.0, 0.5]]))
        ))
    )
    opt = DesignOptimizer(vae_model=vae, fem_evaluator=None, device="cpu",
                          latent_dim=16, sim_cfg=cfg)
    return opt

def test_custom_weights_change_objective():
    cfg_a = {"w_stress": 1.0, "w_compliance": 0.0, "w_mass": 0.0,
             "max_stress_mpa": 999, "max_disp_mm": 999}
    cfg_b = {"w_stress": 0.0, "w_compliance": 1.0, "w_mass": 0.0,
             "max_stress_mpa": 999, "max_disp_mm": 999}
    opt_a = _make_optimizer(cfg_a)
    opt_b = _make_optimizer(cfg_b)
    z = np.zeros(16)
    obj_a = opt_a.objective_function(z)
    obj_b = opt_b.objective_function(z)
    assert obj_a != obj_b

def test_constraint_penalty_fires():
    cfg = {"w_stress": 1.0, "w_compliance": 0.1, "w_mass": 0.01,
           "max_stress_mpa": 10.0,   # very low — 50 MPa predicted > 10 → penalty
           "max_disp_mm": 999}
    opt = _make_optimizer(cfg)
    z = np.zeros(16)
    obj = opt.objective_function(z)
    assert obj > 1000  # penalty applied
```

**Step 2: Run — expect failure**
```bash
cd /home/genpipeline && source venv/bin/activate
python -m pytest tests/test_objective_weights.py -v 2>&1 | tail -20
```

**Step 3: Modify `DesignOptimizer.__init__` to accept `sim_cfg`**

In `optimization_engine.py`, find `class DesignOptimizer` and its `__init__`. Add parameter:
```python
def __init__(self, vae_model, fem_evaluator, device, latent_dim=16, sim_cfg=None):
    ...
    self.sim_cfg = sim_cfg or {
        "w_stress": 1.0, "w_compliance": 0.1, "w_mass": 0.01,
        "max_stress_mpa": 1e9, "max_disp_mm": 1e9,
    }
```

**Step 4: Modify `objective_function`**

Find the `objective_function` method and replace the hardcoded formula. The surrogate branch (no real eval) uses performance predictor output `[stress_pred, compliance_pred]`:

```python
def objective_function(self, z: np.ndarray, real_eval=False) -> float:
    cfg = self.sim_cfg
    w_s, w_c, w_m = cfg["w_stress"], cfg["w_compliance"], cfg["w_mass"]
    max_s, max_d  = cfg["max_stress_mpa"], cfg["max_disp_mm"]

    if real_eval and self.fem_evaluator is not None:
        results = self.fem_evaluator.evaluate(z)
        stress     = results["stress"]
        compliance = results["compliance"]
        mass       = results.get("mass", 0.0)
        disp       = results.get("displacement_max", 0.0)
        penalty    = 1e6 if (stress > max_s or disp > max_d) else 0.0
        return w_s * stress + w_c * compliance + w_m * mass + penalty
    else:
        import torch
        zt = torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            perf = self.vae_model.performance_predictor(
                self.vae_model.encode(zt)[0]
            )
        pred = perf.detach().cpu().numpy()
        stress_pred     = float(pred[0, 0])
        compliance_pred = float(pred[0, 1])
        penalty = 1e6 if stress_pred > max_s else 0.0
        return w_s * stress_pred + w_c * compliance_pred + penalty
```

**Step 5: Run tests — expect PASS**
```bash
cd /home/genpipeline && source venv/bin/activate
python -m pytest tests/test_objective_weights.py -v
```

**Step 6: Commit**
```bash
cd /home/genpipeline
git add optimization_engine.py tests/test_objective_weights.py
git commit -m "feat: BO objective uses sim_config weights and constraint penalties"
```

---

## Task 4: Settings expander in `dashboard.py`

**Files:**
- Modify: `dashboard.py` — left column section
- Modify: `dashboard_bo_runner.py` — pass sim_cfg to optimizer

**Step 1: Load config at top of `dashboard.py`**

After the `variants = load_variants()` and `vae = load_vae()` lines, add:
```python
from sim_config import load_config, save_config, PRESETS, default_config
sim_cfg = load_config()
```

**Step 2: Add settings expander in the left column**

In the left column block (`with left_col:`), insert this **above** the `st.markdown("**Designs**")` line:

```python
    with st.expander("Settings", expanded=False):
        preset = st.selectbox("Material", list(PRESETS.keys()) + ["Custom"],
                              index=list(PRESETS.keys()).index(sim_cfg.get("material", "Steel"))
                              if sim_cfg.get("material") in PRESETS else len(PRESETS))
        if preset != "Custom" and preset in PRESETS:
            p = PRESETS[preset]
            sim_cfg["material"]      = preset
            sim_cfg["E_mpa"]         = p["E_mpa"]
            sim_cfg["poisson"]       = p["poisson"]
            sim_cfg["density_kg_m3"] = p["density_kg_m3"]
            sim_cfg["yield_mpa"]     = p["yield_mpa"]

        st.markdown("**Material**")
        sim_cfg["E_mpa"]         = st.number_input("E (MPa)", value=float(sim_cfg["E_mpa"]), step=1000.0)
        sim_cfg["poisson"]       = st.number_input("Poisson ν", value=float(sim_cfg["poisson"]), step=0.01, format="%.2f")
        sim_cfg["density_kg_m3"] = st.number_input("Density (kg/m³)", value=float(sim_cfg["density_kg_m3"]), step=100.0)
        sim_cfg["yield_mpa"]     = st.number_input("Yield strength (MPa)", value=float(sim_cfg["yield_mpa"]), step=10.0)

        st.markdown("**Load**")
        sim_cfg["force_n"] = st.number_input("Force (N)", value=float(sim_cfg["force_n"]), step=100.0)

        st.markdown("**Safety & Constraints**")
        sim_cfg["safety_factor"]  = st.number_input("Safety factor", value=float(sim_cfg["safety_factor"]), step=0.1, format="%.2f")
        auto_max = sim_cfg["yield_mpa"] / max(sim_cfg["safety_factor"], 0.01)
        sim_cfg["max_stress_mpa"] = st.number_input("Max stress (MPa)", value=float(sim_cfg.get("max_stress_mpa", auto_max)), step=5.0,
                                                    help=f"Auto from yield/SF: {auto_max:.1f} MPa")
        sim_cfg["max_disp_mm"]    = st.number_input("Max disp (mm)", value=float(sim_cfg["max_disp_mm"]), step=0.1, format="%.2f")

        st.markdown("**Objective weights**")
        sim_cfg["w_stress"]     = st.slider("w stress",     0.0, 2.0, float(sim_cfg["w_stress"]),     0.05)
        sim_cfg["w_compliance"] = st.slider("w compliance", 0.0, 2.0, float(sim_cfg["w_compliance"]), 0.01)
        sim_cfg["w_mass"]       = st.slider("w mass",       0.0, 2.0, float(sim_cfg["w_mass"]),       0.01)

        if st.button("Save config"):
            save_config(sim_cfg)
            st.success("Saved to sim_config.json")
```

**Step 3: Pass `sim_cfg` to BORunner**

In the `if run_clicked` block in `dashboard.py`, change `BORunner(...)` call to add `sim_cfg=sim_cfg`:
```python
    runner = BORunner(
        state=app_state, vae=vae, device=DEVICE,
        n_iters=int(n_iters),
        mode=mode.lower().replace(" ", "-"),
        freecad_cmd=... ,
        output_dir="/tmp/bo_variants",
        sim_cfg=sim_cfg,
    )
```

**Step 4: Update `BORunner` to accept and use `sim_cfg`**

In `dashboard_bo_runner.py`, add `sim_cfg=None` to `__init__` and store it:
```python
def __init__(self, state, vae, device, n_iters=50, mode="bo-only",
             freecad_cmd="", output_dir="/tmp/bo_variants", sim_cfg=None):
    ...
    self.sim_cfg = sim_cfg or {}
```

In `BORunner.run()`, pass sim_cfg to `DesignOptimizer`:
```python
optimizer = DesignOptimizer(
    vae_model=self.vae,
    fem_evaluator=None,
    device=self.device,
    latent_dim=getattr(self.vae, "latent_dim", 16),
    sim_cfg=self.sim_cfg,
)
```

**Step 5: Smoke test — dashboard loads**
```bash
cd /home/genpipeline && source venv/bin/activate
python -c "import dashboard; print('import ok')" 2>&1 | tail -5
```

**Step 6: Commit**
```bash
cd /home/genpipeline
git add dashboard.py dashboard_bo_runner.py
git commit -m "feat: sim settings expander with material presets, weights, constraints"
```

---

## Task 5: Full test run

**Step 1: Run all tests**
```bash
cd /home/genpipeline && source venv/bin/activate
python -m pytest tests/ -v 2>&1
```
Expected: all pass (existing 9 + new 7 = 16 tests).

**Step 2: Commit if any fixes needed, then push**
```bash
cd /home/genpipeline
git push
```
