"""
run_fem_variant.py — runs INSIDE FreeCAD's Python via FreeCADCmd.exe.

Creates a cantilever beam from scratch, meshes it with Gmsh, runs a
CalculiX static analysis, and exports results + STL.

Geometry:
  Box 100 mm (L) × 20 mm (W) × h_mm (H)
  Optional through-hole (cylinder along Y, radius r_mm, centred at L/2, H/2)
  Fixed: left face (x = 0)
  Load:  1000 N downward (-Z) on right face (x = 100)

Called by freecad_bridge.py generate_variants().

Args (via JSON config file — FreeCAD 1.0 swallows --flag style args):
  Positional: path to a JSON file containing {"h_mm": 10, "r_mm": 3, "output": "C:\\..."}

Writes:
  <stem>_fem_results.json   (stress_max, stress_mean, compliance,
                              displacement_max, mass, parameters)
  <stem>_mesh.stl
"""

import sys
import os
import json


# ── config loading ─────────────────────────────────────────────────────────────
def _load_config():
    # FreeCAD 1.0: argv = [freecad.exe, --console, script.py, config.cfg]
    # FreeCAD 0.x: argv = [script.py, config.cfg]
    # Skip exe names, --flags, and .py files to find the config file arg.
    args = [a for a in sys.argv[1:]
            if not a.endswith(".exe") and not a.startswith("--") and not a.endswith(".py")]
    if not args:
        raise RuntimeError("Usage: freecad.exe --console run_fem_variant.py <config.cfg>")
    with open(args[0]) as f:
        return json.load(f)


# ── geometry ──────────────────────────────────────────────────────────────────

def make_shape(h_mm, r_mm):
    import Part
    import FreeCAD as App

    box = Part.makeBox(100.0, 20.0, h_mm)

    if r_mm > 0.5:
        # Cylinder along Y through the centre of the XZ face
        origin = App.Vector(50.0, 0.0, h_mm / 2.0)
        axis   = App.Vector(0.0, 1.0, 0.0)
        cyl = Part.makeCylinder(r_mm, 20.0, origin, axis)
        shape = box.cut(cyl)
    else:
        shape = box

    return shape


def make_lbracket_shape(arm_len=80.0, arm_h=15.0, thickness=10.0):
    """L-bracket: horizontal arm + vertical arm, fixed at corner."""
    import Part
    h_arm = Part.makeBox(arm_len, thickness, arm_h)          # horizontal
    v_arm = Part.makeBox(thickness, thickness, arm_len)       # vertical
    shape = h_arm.fuse(v_arm)
    return shape

def make_tapered_beam_shape(length=100.0, h_start=20.0, h_end=8.0, width=20.0):
    """Beam that tapers from h_start at x=0 to h_end at x=length."""
    import Part
    import FreeCAD as App
    pts = [
        App.Vector(0, 0, 0), App.Vector(length, 0, 0),
        App.Vector(length, 0, h_end), App.Vector(0, 0, h_start),
    ]
    face = Part.makePolygon(pts + [pts[0]])
    face = Part.Face(face)
    shape = face.extrude(App.Vector(0, width, 0))
    return shape

def make_ribbed_plate_shape(length=100.0, width=20.0, plate_h=5.0,
                             rib_h=12.0, rib_w=4.0, n_ribs=3):
    """Flat plate with evenly spaced ribs."""
    import Part
    plate = Part.makeBox(length, width, plate_h)
    spacing = length / (n_ribs + 1)
    ribs = plate
    for k in range(n_ribs):
        x = spacing * (k + 1)
        rib = Part.makeBox(rib_w, width, rib_h,
                           __import__('FreeCAD').Vector(x - rib_w/2, 0, plate_h))
        ribs = ribs.fuse(rib)
    return ribs


def find_face(shape, normal_target, tol=0.05):
    """Return 'FaceN' whose outward normal is closest to normal_target."""
    import FreeCAD as App
    best_i, best_dot = 0, -2.0
    for i, face in enumerate(shape.Faces, 1):
        try:
            u0 = (face.ParameterRange[0] + face.ParameterRange[1]) / 2
            v0 = (face.ParameterRange[2] + face.ParameterRange[3]) / 2
            n = face.normalAt(u0, v0)
        except Exception:
            n = face.normalAt(0, 0)
        dot = n.dot(normal_target)
        if dot > best_dot:
            best_dot = dot
            best_i = i
    return f"Face{best_i}"


# ── FEM setup + solve ─────────────────────────────────────────────────────────

def run_fem(doc, shape_obj, h_mm, r_mm, output_dir, cfg=None):
    if cfg is None:
        cfg = {}
    import FreeCAD as App
    import Fem          # registers Fem::* C++ types
    import ObjectsFem

    geom_type = cfg.get("geometry", "cantilever")
    stem = f"{geom_type[:4]}_h{h_mm:.1f}_r{r_mm:.1f}".replace(".", "p")

    # ── Analysis container ────────────────────────────────────────────────
    analysis = ObjectsFem.makeAnalysis(doc, "FemAnalysis")

    # ── Material definition ───────────────────────────────────────────────
    mat = ObjectsFem.makeMaterialSolid(doc, "FemMaterialSolid")
    e_mpa = cfg.get('E_mpa', 210000)
    mat.Material = {
        "Name":          "CalculiX-Custom",
        "YoungsModulus": f"{e_mpa} MPa",
        "PoissonRatio":  str(cfg.get('poisson', 0.30)),
        "Density":       f"{cfg.get('density_kg_m3', 7900)} kg/m^3",
        "ThermalConductivity": f"{cfg.get('thermal_conductivity', 50.0)} W/m/K",
        "SpecificHeat":  f"{cfg.get('specific_heat', 490.0)} J/kg/K",
        "ThermalExpansionCoefficient": str(cfg.get('thermal_expansion', 1.2e-5)),
    }
    analysis.addObject(mat)

    # ── Mesh (Adaptive Refinement for Accuracy) ───────────────────────────
    # Mesh size must resolve the smallest feature (hole radius).
    # r < base_length causes Gmsh to fail on the hole → CalculiX returns zeros.
    # Cap base_length at half the hole radius so the feature is always resolved.
    base_length = 3.0 if e_mpa > 10000 else 1.5
    if r_mm > 0.5:
        # Resolve hole with ~2 elements across diameter, but clamp to [1.0, 3.0]mm
        # to avoid timeout: r=0.77 → r/2=0.38mm → 200+ elements/side → hangs
        base_length = min(base_length, max(r_mm / 2.0, 1.0))

    mesh_obj = ObjectsFem.makeMeshGmsh(doc, "FEMMeshGmsh")
    mesh_obj.Shape = shape_obj
    mesh_obj.CharacteristicLengthMax = f"{base_length} mm"
    mesh_obj.CharacteristicLengthMin = f"{base_length / 3.0} mm"
    mesh_obj.ElementOrder = "2nd"               # C3D10
    analysis.addObject(mesh_obj)

    doc.recompute()

    # Generate mesh via GmshTools
    from femmesh.gmshtools import GmshTools
    gmsh = GmshTools(mesh_obj)
    error = gmsh.create_mesh()
    if error:
        print(f"[run_fem_variant] Gmsh warning: {error}")

    doc.recompute()

    # ── Fixed constraint (left face, x=0) ─────────────────────────────────
    import FreeCAD as App
    left_face  = find_face(shape_obj.Shape, App.Vector(-1, 0, 0))
    fixed = ObjectsFem.makeConstraintFixed(doc, "FEMConstraintFixed")
    fixed.References = [(shape_obj, left_face)]
    analysis.addObject(fixed)

    # ── Force constraint (right face, x=100, 1000 N transverse) ──────────
    # Direction face: bottom face (outward normal -Z) — FreeCAD interprets this
    # as transverse direction and populates CLOAD correctly.
    # DirectionVector = App.Vector(0, 0, -1) alone is ignored by the ccx writer.
    right_face = find_face(shape_obj.Shape, App.Vector(1, 0, 0))
    down_face  = find_face(shape_obj.Shape, App.Vector(0, 0, -1))
    force = ObjectsFem.makeConstraintForce(doc, "FEMConstraintForce")
    force.References = [(shape_obj, right_face)]
    force.Force      = App.Units.Quantity(f"{cfg.get('force_n', 1000)} N")
    force.Direction  = (shape_obj, [down_face])
    analysis.addObject(force)

    # ── Solver: CalculiX ──────────────────────────────────────────────────
    solver = ObjectsFem.makeSolverCalculiXCcxTools(doc, "SolverCcxTools")
    solver.AnalysisType     = "static"
    solver.GeometricalNonlinearity = "linear"
    solver.ThermoMechSteadyState   = False
    solver.MatrixSolverType        = "default"
    solver.IterationsControlParameterTimeUse = False
    analysis.addObject(solver)

    doc.recompute()

    # ── Run solver ────────────────────────────────────────────────────────
    from femsolver.run import run_fem_solver
    run_fem_solver(solver)
    doc.recompute()

    # ── Extract results ───────────────────────────────────────────────────
    results = {
        "stress_max":       0.0,
        "stress_mean":      0.0,
        "compliance":       0.0,
        "displacement_max": 0.0,
        "mass":             0.0,
        "parameters": {"h_mm": h_mm, "r_mm": r_mm},
    }

    for obj in doc.Objects:
        if "Result" in obj.TypeId or hasattr(obj, "vonMises"):
            if hasattr(obj, "vonMises") and obj.vonMises:
                vals = list(obj.vonMises)
                results["stress_max"]  = float(max(vals))
                results["stress_mean"] = float(sum(vals) / len(vals))
            if hasattr(obj, "DisplacementLengths") and obj.DisplacementLengths:
                disps = list(obj.DisplacementLengths)
                results["compliance"]       = float(sum(disps))
                results["displacement_max"] = float(max(disps))

    # Mass from geometry (density × volume)
    vol_mm3 = shape_obj.Shape.Volume       # mm³
    rho_kg_mm3 = 7900 / 1e9               # 7900 kg/m³ → kg/mm³
    results["mass"] = float(vol_mm3 * rho_kg_mm3)

    # ── Export STL ────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    stl_path  = os.path.join(output_dir, f"{stem}_mesh.stl")
    json_path = os.path.join(output_dir, f"{stem}_fem_results.json")

    shape_obj.Shape.exportStl(stl_path)
    print(f"[run_fem_variant] STL: {stl_path}")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[run_fem_variant] JSON: {json_path}")
    print(f"[run_fem_variant] OK  h={h_mm} r={r_mm}  "
          f"stress_max={results['stress_max']:.1f} MPa  "
          f"compliance={results['compliance']:.4f}  "
          f"mass={results['mass']:.4f} kg")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = _load_config()

    import FreeCAD as App

    h   = float(cfg["h_mm"])
    r   = float(cfg["r_mm"])
    out = cfg["output"]
    geom_type = cfg.get("geometry", "cantilever")

    stem = f"{geom_type[:4]}_h{h:.1f}_r{r:.1f}".replace(".", "p")
    doc_name = f"FEM_{stem}"

    print(f"[run_fem_variant] h_mm={h}  r_mm={r}  geometry={geom_type}  output={out}")

    try:
        doc = App.newDocument(doc_name)

        import Part
        if geom_type == "lbracket":
            shape = make_lbracket_shape(arm_h=h, thickness=r if r > 1 else 10.0)
        elif geom_type == "tapered":
            shape = make_tapered_beam_shape(h_start=h, h_end=max(4.0, h*0.4))
        elif geom_type == "ribbed":
            shape = make_ribbed_plate_shape(plate_h=h*0.4, rib_h=h)
        else:  # cantilever (default)
            shape = make_shape(h, r)

        feat = doc.addObject("Part::Feature", "CantileverBeam")
        feat.Shape = shape
        doc.recompute()

        results = run_fem(doc, feat, h, r, out, cfg=cfg)
        sys.exit(0)

    except Exception as e:
        print(f"[run_fem_variant] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
