# tests/test_export_pipeline.py
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "freecad_workbench"))

# export_pipeline imports FreeCAD at module level — mock it before import
import unittest.mock as mock
sys.modules["FreeCAD"] = mock.MagicMock()

from export_pipeline import (
    collect_constraints,
    collect_loads,
    find_seed_part,
    _derive_bc_from_constraints,
)


def _make_constraint(ctype="fixed", label="Fixed_1", refs=None):
    obj = SimpleNamespace(
        Name="Constraint_fixed",
        ConstraintType=ctype,
        Label=label,
        References=refs or [],
    )
    return obj


def _make_load(ltype="force", magnitude=1000.0, direction=(0, 0, -1), label="Load_1", refs=None):
    d = SimpleNamespace(x=direction[0], y=direction[1], z=direction[2])
    obj = SimpleNamespace(
        Name="Load_force",
        LoadType=ltype,
        Magnitude=magnitude,
        Direction=d,
        Label=label,
        References=refs or [],
    )
    return obj


def _make_doc(*objects):
    return SimpleNamespace(Objects=list(objects), FileName="")


class TestCollectConstraints:
    def test_empty_doc(self):
        assert collect_constraints(_make_doc()) == []

    def test_single_fixed_no_refs(self):
        obj = _make_constraint("fixed", "MyFixed")
        result = collect_constraints(_make_doc(obj))
        assert len(result) == 1
        assert result[0]["type"] == "fixed"
        assert result[0]["label"] == "MyFixed"
        assert result[0]["faces"] == []

    def test_ignores_non_constraint_objects(self):
        body = SimpleNamespace(Name="Body", Label="Body")
        obj = _make_constraint()
        result = collect_constraints(_make_doc(body, obj))
        assert len(result) == 1

    def test_multiple_constraints(self):
        c1 = _make_constraint("fixed",    "C1")
        c2 = _make_constraint("symmetry", "C2")
        result = collect_constraints(_make_doc(c1, c2))
        assert len(result) == 2
        types = {r["type"] for r in result}
        assert types == {"fixed", "symmetry"}


class TestCollectLoads:
    def test_empty_doc(self):
        assert collect_loads(_make_doc()) == []

    def test_single_force(self):
        obj = _make_load("force", 500.0, (0, 0, -1), "MyForce")
        result = collect_loads(_make_doc(obj))
        assert len(result) == 1
        assert result[0]["magnitude"] == 500.0
        assert result[0]["direction"] == [0, 0, -1]

    def test_ignores_non_load_objects(self):
        body = SimpleNamespace(Name="Body", Label="Body")
        obj = _make_load()
        result = collect_loads(_make_doc(body, obj))
        assert len(result) == 1


class TestFindSeedPart:
    def test_no_seed_returns_none(self):
        assert find_seed_part(_make_doc()) is None

    def test_finds_seed_part(self):
        seed = SimpleNamespace(Name="SeedPart_cantilever", GeometryType="cantilever")
        result = find_seed_part(_make_doc(seed))
        assert result is seed

    def test_ignores_non_seed(self):
        body = SimpleNamespace(Name="Body")
        assert find_seed_part(_make_doc(body)) is None


class TestDeriveBCFromConstraints:
    def test_force_direction_extracted(self):
        constraints = [{"type": "fixed", "faces": []}]
        loads = [{"type": "force", "direction": [1, 0, 0], "magnitude": 2000.0}]
        _, _, force_dir, force_n = _derive_bc_from_constraints(constraints, loads)
        assert force_dir == [1, 0, 0]
        assert force_n == 2000.0

    def test_defaults_when_no_loads(self):
        _, _, force_dir, force_n = _derive_bc_from_constraints([], [])
        assert force_dir == [0, 0, -1]
        assert force_n == 1000.0
