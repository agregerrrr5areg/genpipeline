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
    assert abs(cfg["max_stress_mpa"] - 250/1.5) < 0.1
