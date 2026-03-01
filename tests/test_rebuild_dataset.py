# tests/test_rebuild_dataset.py
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from rebuild_dataset import load_pairs


def _write_pair(d: Path, stem: str, stress: float = 100.0):
    (d / f"{stem}_mesh.stl").write_text("solid empty\nendsolid\n")
    (d / f"{stem}_fem_results.json").write_text(json.dumps({
        "stress_max": stress,
        "stress_mean": stress * 0.3,
        "compliance": 10.0,
        "mass": 0.5,
        "parameters": {"h_mm": 10.0, "r_mm": 2.0},
    }))


class TestLoadPairs:
    def test_empty_dir(self, tmp_path):
        assert load_pairs(tmp_path) == []

    def test_single_valid_pair(self, tmp_path):
        _write_pair(tmp_path, "cant_h10p0_r2p0")
        pairs = load_pairs(tmp_path)
        assert len(pairs) == 1
        stl, d, stem = pairs[0]
        assert stl.name == "cant_h10p0_r2p0_mesh.stl"
        assert d["stress_max"] == 100.0

    def test_skips_json_without_stl(self, tmp_path):
        (tmp_path / "cant_h10p0_r2p0_fem_results.json").write_text(
            json.dumps({"stress_max": 50.0, "compliance": 5.0, "mass": 0.3,
                        "parameters": {}})
        )
        assert load_pairs(tmp_path) == []

    def test_skips_zero_stress_zero_compliance(self, tmp_path):
        _write_pair(tmp_path, "bad_run", stress=0.0)
        # Overwrite compliance to 0 too
        (tmp_path / "bad_run_fem_results.json").write_text(
            json.dumps({"stress_max": 0.0, "compliance": 0.0, "mass": 0.1,
                        "parameters": {}})
        )
        assert load_pairs(tmp_path) == []

    def test_multiple_geometries(self, tmp_path):
        _write_pair(tmp_path, "cant_h10p0_r2p0")
        _write_pair(tmp_path, "lbra_h15p0_r5p0")
        _write_pair(tmp_path, "tape_h12p0_r3p0")
        pairs = load_pairs(tmp_path)
        assert len(pairs) == 3
        stems = {p[2] for p in pairs}
        assert "cant_h10p0_r2p0" in stems
        assert "lbra_h15p0_r5p0" in stems

    def test_geometry_count_reported(self, tmp_path, caplog):
        _write_pair(tmp_path, "cant_h10p0_r2p0")
        _write_pair(tmp_path, "cant_h12p0_r1p0")
        _write_pair(tmp_path, "lbra_h15p0_r5p0")
        import logging
        with caplog.at_level(logging.INFO):
            load_pairs(tmp_path)
        assert "cant=2" in caplog.text
