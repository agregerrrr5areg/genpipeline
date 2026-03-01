"""
export_pipeline.py
==================
Serialises GenDesign document objects (constraints, loads, seed part)
to the JSON format consumed by freecad_bridge.py / sim_config.py.

Output schema (gendesign_config.json):
{
  "geometry_type": "lbracket",
  "checkpoint_path": "/home/genpipeline/checkpoints/vae_best.pth",
  "n_iter": 50,
  "volume_fraction": 0.4,
  "max_stress_mpa": 250.0,
  "no_overhang": false,
  "wsl2_pipeline_path": "/home/genpipeline",
  "constraints": [
    {"type": "fixed", "label": "Constraint_fixed",
     "faces": ["Body.Face1", "Body.Face3"]}
  ],
  "loads": [
    {"type": "force", "magnitude": 1000.0,
     "direction": [0, 0, -1], "label": "Load_force_1000",
     "faces": ["Body.Face5"]}
  ],
  "fixed_face_normal": [0, 0, -1],
  "load_face_normal":  [1, 0, 0],
  "force_n": 1000.0,
  "force_direction": [0, 0, -1]
}
"""

import json
import os
from pathlib import Path

import FreeCAD


def _face_normal(shape_obj, sub_name):
    """Return the outward face normal (as list) for a given sub-element name."""
    try:
        shape = shape_obj.Shape
        face  = shape.getElement(sub_name)
        u0 = (face.ParameterRange[0] + face.ParameterRange[1]) / 2
        v0 = (face.ParameterRange[2] + face.ParameterRange[3]) / 2
        n = face.normalAt(u0, v0)
        return [round(n.x, 3), round(n.y, 3), round(n.z, 3)]
    except Exception:
        return [0, 0, 0]


def collect_constraints(doc):
    out = []
    for obj in doc.Objects:
        if obj.Name.startswith("Constraint") and hasattr(obj, "ConstraintType"):
            faces = []
            if hasattr(obj, "References") and obj.References:
                for parent, subs in obj.References:
                    for sub in subs:
                        faces.append(f"{parent.Name}.{sub}")
            out.append({
                "type":  obj.ConstraintType,
                "label": obj.Label,
                "faces": faces,
            })
    return out


def collect_loads(doc):
    out = []
    for obj in doc.Objects:
        if obj.Name.startswith("Load") and hasattr(obj, "LoadType"):
            faces = []
            if hasattr(obj, "References") and obj.References:
                for parent, subs in obj.References:
                    for sub in subs:
                        faces.append(f"{parent.Name}.{sub}")
            d = obj.Direction
            out.append({
                "type":      obj.LoadType,
                "magnitude": obj.Magnitude,
                "direction": [round(d.x, 3), round(d.y, 3), round(d.z, 3)],
                "label":     obj.Label,
                "faces":     faces,
            })
    return out


def find_seed_part(doc):
    for obj in doc.Objects:
        if obj.Name.startswith("SeedPart") and hasattr(obj, "GeometryType"):
            return obj
    return None


def _derive_bc_from_constraints(constraints, loads):
    """
    Pull fixed_face_normal and load_face_normal from the first fixed constraint
    and first force load respectively, falling back to cantilever defaults.
    """
    fixed_normal = [-1, 0, 0]
    load_normal  = [1, 0, 0]
    force_dir    = [0, 0, -1]
    force_n      = 1000.0

    for c in constraints:
        if c["type"] == "fixed":
            # Try to read the actual face normal from the first referenced face
            # (requires access to the live document — caller may pass shape_obj)
            break  # placeholder; real normal set via shape below

    for ld in loads:
        if ld["type"] == "force":
            force_dir = ld["direction"]
            force_n   = ld["magnitude"]
            break

    return fixed_normal, load_normal, force_dir, force_n


def export_config(doc, output_path: str = None) -> dict:
    """
    Build and write gendesign_config.json from the current document.
    Returns the config dict (also written to output_path or alongside the doc).
    """
    seed = find_seed_part(doc)
    constraints = collect_constraints(doc)
    loads       = collect_loads(doc)

    fixed_normal, load_normal, force_dir, force_n = _derive_bc_from_constraints(
        constraints, loads
    )

    # Per-geometry BC defaults (overrides derived values when seed is set)
    GEOM_BC = {
        "cantilever": {"fixed_face_normal": [-1,0,0], "load_face_normal": [1,0,0]},
        "tapered":    {"fixed_face_normal": [-1,0,0], "load_face_normal": [1,0,0]},
        "ribbed":     {"fixed_face_normal": [-1,0,0], "load_face_normal": [1,0,0]},
        "lbracket":   {"fixed_face_normal": [0,0,-1], "load_face_normal": [1,0,0]},
    }

    geom_type = seed.GeometryType if seed else "cantilever"
    bc = GEOM_BC.get(geom_type, GEOM_BC["cantilever"])

    cfg = {
        "geometry_type":       geom_type,
        "checkpoint_path":     seed.CheckpointPath   if seed else "/home/genpipeline/checkpoints/vae_best.pth",
        "n_iter":              seed.NIter             if seed else 50,
        "volume_fraction":     seed.VolumeFraction    if seed else 0.4,
        "max_stress_mpa":      seed.MaxStressMPa      if seed else 250.0,
        "no_overhang":         bool(seed.NoOverhang)  if seed else False,
        "wsl2_pipeline_path":  seed.WSL2PipelinePath  if seed else "/home/genpipeline",
        "constraints":         constraints,
        "loads":               loads,
        # BC config consumed by freecad_bridge / run_fem_variant
        "fixed_face_normal":   bc["fixed_face_normal"],
        "load_face_normal":    bc["load_face_normal"],
        "force_n":             force_n,
        "force_direction":     force_dir,
    }

    if output_path is None:
        if doc.FileName:
            output_path = str(Path(doc.FileName).parent / "gendesign_config.json")
        else:
            output_path = "C:\\Windows\\Temp\\gendesign_config.json"

    with open(output_path, "w") as f:
        json.dump(cfg, f, indent=2)

    FreeCAD.Console.PrintMessage(f"[GenDesign] Config exported to {output_path}\n")
    return cfg
