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
