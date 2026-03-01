"""
run_fem_variant.py — runs INSIDE FreeCAD's Python via FreeCADCmd.exe.
"""

import sys
import os
import json

def _load_config():
    args = [a for a in sys.argv[1:]
            if not a.endswith(".exe") and not a.startswith("--") and not a.endswith(".py")]
    if not args:
        raise RuntimeError("Usage: freecad.exe --console run_fem_variant.py <config.cfg>")
    with open(args[0]) as f:
        return json.load(f)

def make_lbracket_shape(arm_h=15.0, thickness=10.0, arm_len=80.0):
    """L-bracket: horizontal arm + vertical arm. Fixed at base of vertical arm."""
    import Part
    h_arm = Part.makeBox(arm_len, thickness, arm_h)        # horizontal along X
    v_arm = Part.makeBox(thickness, thickness, arm_len)    # vertical along Z
    return h_arm.fuse(v_arm)

def make_tapered_beam_shape(h_start=20.0, h_end=None, length=100.0, width=20.0):
    """Beam tapering from h_start at x=0 to h_end at x=length."""
    import Part, FreeCAD as App
    if h_end is None:
        h_end = max(4.0, h_start * 0.4)
    pts = [App.Vector(0,0,0), App.Vector(length,0,0),
           App.Vector(length,0,h_end), App.Vector(0,0,h_start)]
    face = Part.Face(Part.makePolygon(pts + [pts[0]]))
    return face.extrude(App.Vector(0, width, 0))

def make_ribbed_plate_shape(rib_h=12.0, plate_frac=0.4, length=100.0, width=20.0, n_ribs=3):
    """Flat plate with evenly spaced ribs."""
    import Part, FreeCAD as App
    plate_h = max(2.0, rib_h * plate_frac)
    rib_w = 4.0
    plate = Part.makeBox(length, width, plate_h)
    spacing = length / (n_ribs + 1)
    shape = plate
    for k in range(n_ribs):
        x = spacing * (k + 1)
        rib = Part.makeBox(rib_w, width, rib_h,
                           App.Vector(x - rib_w/2, 0, plate_h))
        shape = shape.fuse(rib)
    return shape

def make_shape(h_mm, r_mm):
    import Part
    import FreeCAD as App
    box = Part.makeBox(100.0, 20.0, h_mm)
    if r_mm > 0.5:
        origin = App.Vector(50.0, 0.0, h_mm / 2.0)
        axis   = App.Vector(0.0, 1.0, 0.0)
        cyl = Part.makeCylinder(r_mm, 20.0, origin, axis)
        shape = box.cut(cyl)
    else:
        shape = box
    return shape

def find_nearest_face(shape, point_target):
    import FreeCAD as App
    best_i, best_dist = 0, 1e9
    target_vec = App.Vector(*point_target)
    for i, face in enumerate(shape.Faces, 1):
        # Center of the face
        center = face.CenterOfMass
        dist = (center - target_vec).Length
        if dist < best_dist:
            best_dist = dist
            best_i = i
    return f"Face{best_i}"

def find_face(shape, normal_target, tol=0.05):
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

def run_fem(doc, shape_obj, h_mm, r_mm, output_dir, cfg=None):
    if cfg is None: cfg = {}
    import FreeCAD as App
    import Fem
    import ObjectsFem

    geom_type = cfg.get("geometry", "cantilever")
    stem = cfg.get("stem") or f"{geom_type[:4]}_h{h_mm:.1f}_r{r_mm:.1f}".replace(".", "p")
    analysis = ObjectsFem.makeAnalysis(doc, "FemAnalysis")

    # ── Material definition ───────────────────────────────────────────────
    mat = ObjectsFem.makeMaterialSolid(doc, "FemMaterialSolid")
    e_mpa = cfg.get('E_mpa', 210000)
    rho_kg_m3 = cfg.get('density_kg_m3', 7900)
    mat.Material = {
        "Name":          "CalculiX-Custom",
        "YoungsModulus": f"{e_mpa} MPa",
        "PoissonRatio":  str(cfg.get('poisson', 0.30)),
        "Density":       f"{rho_kg_m3} kg/m^3",
        "ThermalConductivity": f"{cfg.get('thermal_conductivity_w_mk', 50.0)} W/m/K",
        "SpecificHeat":  f"{cfg.get('specific_heat_j_kgk', 490.0)} J/kg/K",
        "ThermalExpansionCoefficient": str(cfg.get('thermal_expansion_coeff', 1.2e-5)),
    }
    analysis.addObject(mat)

    # ── Mesh ─────────────────────────────────────────────────────────────
    base_length = 3.0 if e_mpa > 10000 else 1.5
    if r_mm > 0.5:
        base_length = min(base_length, max(r_mm / 2.0, 1.0))

    mesh_obj = ObjectsFem.makeMeshGmsh(doc, "FEMMeshGmsh")
    mesh_obj.Shape = shape_obj
    mesh_obj.CharacteristicLengthMax = f"{base_length} mm"
    mesh_obj.ElementOrder = "2nd"
    analysis.addObject(mesh_obj)
    doc.recompute()

    from femmesh.gmshtools import GmshTools
    gmsh = GmshTools(mesh_obj)
    gmsh.create_mesh()
    doc.recompute()

    # ── Constraints ──────────────────────────────────────────────────────
    fixed_pt = cfg.get("fixed_point")
    if fixed_pt:
        target_face = find_nearest_face(shape_obj.Shape, fixed_pt)
    else:
        fixed_norm = cfg.get("fixed_face_normal", [-1, 0, 0])
        target_face = find_face(shape_obj.Shape, App.Vector(*fixed_norm))
        
    fixed = ObjectsFem.makeConstraintFixed(doc, "FEMConstraintFixed")
    fixed.References = [(shape_obj, target_face)]
    analysis.addObject(fixed)

    load_pt = cfg.get("load_point")
    if load_pt:
        target_load_face = find_nearest_face(shape_obj.Shape, load_pt)
    else:
        load_norm = cfg.get("load_face_normal", [1, 0, 0])
        target_load_face = find_face(shape_obj.Shape, App.Vector(*load_norm))
    
    # Direction face for the load
    force_dir = cfg.get("force_direction", [0, 0, -1])
    down_face  = find_face(shape_obj.Shape, App.Vector(*force_dir))
    
    force = ObjectsFem.makeConstraintForce(doc, "FEMConstraintForce")
    force.References = [(shape_obj, target_load_face)]
    force.Force      = App.Units.Quantity(f"{cfg.get('force_n', 1000)} N")
    force.Direction  = (shape_obj, [down_face])
    analysis.addObject(force)

    # ── Solver ───────────────────────────────────────────────────────────
    solver = ObjectsFem.makeSolverCalculiXCcxTools(doc, "SolverCcxTools")
    analysis.addObject(solver)
    doc.recompute()

    from femsolver.run import run_fem_solver
    run_fem_solver(solver)
    doc.recompute()

    # ── Results ──────────────────────────────────────────────────────────
    results = {
        "stress_max": 0.0,
        "compliance": 0.0,
        "mass": 0.0,
        "parameters": {"h_mm": h_mm, "r_mm": r_mm},
    }

    for obj in doc.Objects:
        if "Result" in obj.TypeId or hasattr(obj, "vonMises"):
            if hasattr(obj, "vonMises") and obj.vonMises:
                results["stress_max"] = float(max(obj.vonMises))
            if hasattr(obj, "DisplacementLengths") and obj.DisplacementLengths:
                results["compliance"] = float(sum(obj.DisplacementLengths))

    vol_mm3 = shape_obj.Shape.Volume
    results["mass"] = float(vol_mm3 * (rho_kg_m3 / 1e9))

    os.makedirs(output_dir, exist_ok=True)
    shape_obj.Shape.exportStl(os.path.join(output_dir, f"{stem}_mesh.stl"))
    with open(os.path.join(output_dir, f"{stem}_fem_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    cfg = _load_config()
    import FreeCAD as App
    import Part
    h, r = float(cfg["h_mm"]), float(cfg["r_mm"])
    out, geom = cfg["output"], cfg.get("geometry", "cantilever")
    
    doc = App.newDocument(f"FEM_{geom}")
    
    # ── Shape dispatch ─────────────────────────────────────────────────
    if geom == "lbracket":
        shape = make_lbracket_shape(arm_h=h, thickness=max(r, 5.0))
    elif geom == "tapered":
        shape = make_tapered_beam_shape(h_start=h)
    elif geom == "ribbed":
        shape = make_ribbed_plate_shape(rib_h=h, plate_frac=max(r, 2.0) / 10.0)
    else:
        # Custom template hook (checked last so built-in geometries take priority)
        custom_script_path = os.path.join(os.path.dirname(__file__), "custom_template.py")
        if os.path.exists(custom_script_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("custom_template", custom_script_path)
            ct = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ct)
            shape = ct.make_custom_shape(cfg)
        else:
            shape = make_shape(h, r)  # default: cantilever

    feat = doc.addObject("Part::Feature", "Beam")
    feat.Shape = shape
    doc.recompute()
    run_fem(doc, feat, h, r, out, cfg=cfg)
