"""sim_config.py — load/save sim_config.json with presets and defaults."""
from __future__ import annotations
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "sim_config.json"
MATERIALS_PATH = Path(__file__).parent / "materials.json"

def load_materials() -> dict[str, dict]:
    if not MATERIALS_PATH.exists():
        # Minimal defaults if file is missing
        return {
            "Steel": {"E_mpa": 210000, "poisson": 0.30, "density_kg_m3": 7900, "yield_mpa": 250}
        }
    return json.loads(MATERIALS_PATH.read_text())

def default_config() -> dict:
    presets = load_materials()
    mat = presets.get("Steel", list(presets.values())[0])
    return {
        "material":        "Steel",
        "E_mpa":           mat["E_mpa"],
        "poisson":         mat["poisson"],
        "density_kg_m3":   mat["density_kg_m3"],
        "yield_mpa":       mat["yield_mpa"],
        "tensile_mpa":     mat.get("tensile_mpa", mat["yield_mpa"] * 1.2),
        "thermal_conductivity": mat.get("thermal_conductivity_w_mk", 50.0),
        "specific_heat":   mat.get("specific_heat_j_kgk", 490.0),
        "thermal_expansion": mat.get("thermal_expansion_coeff", 1.2e-5),
        "force_n":         1000.0,
        "safety_factor":   1.5,
        "max_stress_mpa":  round(mat["yield_mpa"] / 1.5, 2),
        "max_disp_mm":     1.0,
        "w_stress":        1.0,
        "w_compliance":    0.1,
        "w_mass":          0.01,
        "constraints":     mat.get("design_constraints", {
            "min_feature_size_mm": 1.0,
            "min_volume_fraction": 0.15,
            "max_volume_fraction": 0.50,
            "organic_smoothness": 0.5
        })
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
