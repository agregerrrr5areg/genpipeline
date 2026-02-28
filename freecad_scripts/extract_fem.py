"""
FreeCAD extraction script — runs INSIDE FreeCAD's Python via FreeCADCmd.exe.

Called by freecad_bridge.py. Do NOT run with the venv Python directly.

Extracts from a .FCStd file:
  - Max/mean stress, compliance, displacement, mass
  - Parameters spreadsheet (FEMbyGEN)
  - Solid body or FEM mesh exported as STL

Outputs (both in --output dir):
  <stem>_fem_results.json
  <stem>_mesh.stl
"""

import sys
import os
import json
import argparse


def parse_args():
    # FreeCADCmd.exe passes the script path as argv[0], real args start at argv[1]
    parser = argparse.ArgumentParser(description="FEM extraction inside FreeCAD")
    parser.add_argument("--input",  required=True, help="Windows path to .FCStd file")
    parser.add_argument("--output", required=True, help="Windows path to output directory")
    return parser.parse_args(sys.argv[1:])


def extract(fcstd_path, output_dir):
    import FreeCAD

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(fcstd_path))[0]

    print(f"[extract_fem] Opening: {fcstd_path}")
    doc = FreeCAD.open(fcstd_path)

    results = {
        "stress_max":       0.0,
        "stress_mean":      0.0,
        "compliance":       0.0,
        "displacement_max": 0.0,
        "mass":             0.0,
        "parameters":       {},
    }

    for obj in doc.Objects:
        # ── FEM result object ─────────────────────────────────────────────
        if hasattr(obj, "StressValues") and obj.StressValues:
            vals = list(obj.StressValues)
            results["stress_max"]  = float(max(vals))
            results["stress_mean"] = float(sum(vals) / len(vals))

        if hasattr(obj, "DisplacementLengths") and obj.DisplacementLengths:
            disps = list(obj.DisplacementLengths)
            results["compliance"]       = float(sum(disps))
            results["displacement_max"] = float(max(disps))

        if hasattr(obj, "Mass") and obj.Mass:
            results["mass"] = float(obj.Mass)

        # ── FEMbyGEN Parameters spreadsheet ──────────────────────────────
        if obj.TypeId == "Spreadsheet::Sheet" and obj.Name == "Parameters":
            aliases = obj.getAliases() if hasattr(obj, "getAliases") else []
            for alias in aliases:
                try:
                    results["parameters"][alias] = obj.get(alias)
                except Exception:
                    pass

    # ── Export mesh as STL ────────────────────────────────────────────────
    stl_path = os.path.join(output_dir, f"{stem}_mesh.stl")
    exported = False

    for obj in doc.Objects:
        if exported:
            break

        # Prefer solid shape
        if hasattr(obj, "Shape") and obj.Shape and obj.Shape.Volume > 0:
            try:
                obj.Shape.exportStl(stl_path)
                print(f"[extract_fem] Exported shape STL: {stl_path}")
                exported = True
            except Exception as e:
                print(f"[extract_fem] Shape export failed ({obj.Name}): {e}")

        # Fallback: FEM mesh
        if not exported and obj.TypeId == "Fem::FemMesh":
            try:
                obj.FemMesh.write(stl_path)
                print(f"[extract_fem] Exported FEM mesh STL: {stl_path}")
                exported = True
            except Exception as e:
                print(f"[extract_fem] FEM mesh export failed ({obj.Name}): {e}")

    if not exported:
        print("[extract_fem] WARNING: No exportable geometry found — STL not written.")

    doc.close()

    # ── Write JSON ────────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, f"{stem}_fem_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[extract_fem] Results saved: {json_path}")
    return results


if __name__ == "__main__":
    args = parse_args()
    try:
        result = extract(args.input, args.output)
        print(f"[extract_fem] OK  stress_max={result['stress_max']:.2f}  "
              f"compliance={result['compliance']:.4f}  mass={result['mass']:.4f}")
        sys.exit(0)
    except Exception as e:
        print(f"[extract_fem] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
