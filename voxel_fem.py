"""
voxel_fem.py — direct CalculiX voxel FEM path, bypassing FreeCAD.

Converts a binary voxel grid (D×H×W) to a CalculiX C3D8 hex mesh, writes a
.inp file, runs ccx, and parses the .frd output for stress/displacement.

Usage:
    python voxel_fem.py --test   # unit test on 10×10×10 solid cube
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from pipeline_utils import NumpyEncoder as _NumpyEncoder, smooth_voxels, FEM_SENTINEL, FEM_VALID_THRESHOLD, is_valid_fem_result

logger = logging.getLogger(__name__)

# ── CalculiX executable discovery ─────────────────────────────────────────────

def _discover_ccx_paths() -> list:
    """
    Build a prioritised list of CalculiX executable candidates:
      1. CCX_PATH env var (explicit override)
      2. Native Linux ccx on PATH
      3. All Windows FreeCAD installs found via glob (any username)
    """
    candidates = []

    env_path = os.environ.get("CCX_PATH")
    if env_path:
        candidates.append(env_path)

    if shutil.which("ccx"):
        candidates.append(shutil.which("ccx"))

    # Glob covers any Windows username and any FreeCAD version
    import glob
    for pattern in [
        "/mnt/c/Users/*/AppData/Local/Programs/FreeCAD*/bin/ccx.exe",
        "/mnt/c/Program Files/FreeCAD*/bin/ccx.exe",
        "/mnt/c/Program Files (x86)/FreeCAD*/bin/ccx.exe",
    ]:
        candidates.extend(glob.glob(pattern))

    return candidates


def find_ccx() -> Optional[str]:
    """Return the first available CalculiX executable, or None."""
    for p in _discover_ccx_paths():
        if p.endswith(".exe"):
            if Path(p).exists():
                logger.info(f"[VoxelFEM] Found ccx: {p}")
                return p
        elif shutil.which(p) or Path(p).exists():
            logger.info(f"[VoxelFEM] Found ccx: {p}")
            return p
    logger.warning("[VoxelFEM] CalculiX (ccx) not found on this system.")
    return None


def _wsl_to_win(wsl_path: str) -> str:
    """Convert /mnt/c/foo → C:\\foo for Windows executables."""
    if wsl_path.startswith("/mnt/"):
        parts = wsl_path[5:].split("/", 1)
        drive = parts[0].upper()
        rest = parts[1].replace("/", "\\") if len(parts) > 1 else ""
        return f"{drive}:\\{rest}"
    return wsl_path


# ── VoxelHexMesher ─────────────────────────────────────────────────────────────

class VoxelHexMesher:
    """Converts a binary voxel grid to a CalculiX C3D8 hex mesh."""

    @staticmethod
    def voxels_to_inp(
        voxels: np.ndarray,
        bbox: Optional[Dict] = None,
        fixed_face: str = "x_min",
        load_face: str = "x_max",
        force_n: float = 1000.0,
        E_mpa: float = 210000.0,
        poisson: float = 0.30,
        output_path: str = "voxel_fem.inp",
    ) -> str:
        """
        Convert binary voxel grid to CalculiX .inp file.

        voxels: (D, H, W) float array — thresholded at 0.5 → solid/void.
        bbox:   {"x": [xmin, xmax], "y": [...], "z": [...]} in mm.
                Defaults to 1 mm/voxel if None.
        fixed_face / load_face: "x_min" | "x_max" | "y_min" | "y_max" | "z_min" | "z_max"
        Returns path to written .inp file.
        """
        D, H, W = voxels.shape
        solid = voxels > 0.5

        if not solid.any():
            raise ValueError("All voxels are void — cannot mesh empty geometry.")

        # Physical size per voxel
        if bbox is None:
            dx, dy, dz = 1.0, 1.0, 1.0
            x0, y0, z0 = 0.0, 0.0, 0.0
        else:
            x0, x1 = bbox["x"]
            y0, y1 = bbox["y"]
            z0, z1 = bbox["z"]
            dx = (x1 - x0) / D
            dy = (y1 - y0) / H
            dz = (z1 - z0) / W

        # Build node table: grid corner (cx, cy, cz) → global node ID (1-indexed)
        node_map: Dict[tuple, int] = {}
        node_coords = []   # [(x_mm, y_mm, z_mm), ...]

        def get_node(cx: int, cy: int, cz: int) -> int:
            key = (cx, cy, cz)
            if key not in node_map:
                nid = len(node_map) + 1
                node_map[key] = nid
                node_coords.append((x0 + cx * dx, y0 + cy * dy, z0 + cz * dz))
            return node_map[key]

        # Build C3D8 elements — one per solid voxel.
        # CalculiX C3D8 node ordering (counter-clockwise bottom then top):
        #   N1=(0,0,0) N2=(1,0,0) N3=(1,1,0) N4=(0,1,0)
        #   N5=(0,0,1) N6=(1,0,1) N7=(1,1,1) N8=(0,1,1)
        elements = []
        for ix in range(D):
            for iy in range(H):
                for iz in range(W):
                    if not solid[ix, iy, iz]:
                        continue
                    n1 = get_node(ix,   iy,   iz)
                    n2 = get_node(ix+1, iy,   iz)
                    n3 = get_node(ix+1, iy+1, iz)
                    n4 = get_node(ix,   iy+1, iz)
                    n5 = get_node(ix,   iy,   iz+1)
                    n6 = get_node(ix+1, iy,   iz+1)
                    n7 = get_node(ix+1, iy+1, iz+1)
                    n8 = get_node(ix,   iy+1, iz+1)
                    elements.append([n1, n2, n3, n4, n5, n6, n7, n8])

        if not elements:
            raise ValueError("No solid voxels found to mesh.")

        # Identify BC nodes by grid face
        def face_nodes(face: str):
            nids = []
            for (cx, cy, cz), nid in node_map.items():
                if   face == "x_min" and cx == 0: nids.append(nid)
                elif face == "x_max" and cx == D: nids.append(nid)
                elif face == "y_min" and cy == 0: nids.append(nid)
                elif face == "y_max" and cy == H: nids.append(nid)
                elif face == "z_min" and cz == 0: nids.append(nid)
                elif face == "z_max" and cz == W: nids.append(nid)
            return nids

        fixed_nodes = face_nodes(fixed_face)
        load_nodes  = face_nodes(load_face)

        if not fixed_nodes:
            raise ValueError(f"No nodes on fixed face '{fixed_face}'.")
        if not load_nodes:
            raise ValueError(f"No nodes on load face '{load_face}'.")

        force_per_node = force_n / len(load_nodes)

        # Write CalculiX .inp
        lines = ["** CalculiX C3D8 hex mesh — generated by voxel_fem.py"]
        lines.append("*NODE, NSET=NALL")
        for nid, (px, py, pz) in enumerate(node_coords, 1):
            lines.append(f"{nid},{px:.6f},{py:.6f},{pz:.6f}")

        lines.append("*ELEMENT, TYPE=C3D8, ELSET=EALL")
        for eid, enodes in enumerate(elements, 1):
            lines.append(f"{eid}," + ",".join(str(n) for n in enodes))

        lines.append("*MATERIAL, NAME=STEEL")
        lines.append("*ELASTIC")
        lines.append(f"{E_mpa},{poisson}")
        lines.append("*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL")
        lines.append("**")

        # Node sets for BCs
        lines.append("*NSET, NSET=NFIX")
        for chunk in [fixed_nodes[i:i+16] for i in range(0, len(fixed_nodes), 16)]:
            lines.append(",".join(str(n) for n in chunk))

        lines.append("*NSET, NSET=NLOAD")
        for chunk in [load_nodes[i:i+16] for i in range(0, len(load_nodes), 16)]:
            lines.append(",".join(str(n) for n in chunk))

        lines.append("**")
        lines.append("*STEP")
        lines.append("*STATIC")
        lines.append("*BOUNDARY")
        lines.append("NFIX,1,3,0.0")   # u=v=w=0
        lines.append("*CLOAD")
        for nid in load_nodes:
            lines.append(f"{nid},3,{-force_per_node:.6f}")   # −Z direction
        lines.append("*NODE FILE")
        lines.append("U")
        lines.append("*EL FILE")
        lines.append("S")
        lines.append("*END STEP")

        output_path = str(output_path)
        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(
            f"[VoxelFEM] Wrote .inp: {output_path} "
            f"({len(node_coords)} nodes, {len(elements)} elements, "
            f"{len(fixed_nodes)} fixed nodes, {len(load_nodes)} load nodes)"
        )
        return output_path

    @staticmethod
    def run_ccx(inp_path: str, ccx_cmd: str = None, timeout: int = 300) -> Dict[str, float]:
        """
        Run CalculiX on an .inp file.
        Returns {"stress_max": ..., "displacement_max": ..., "compliance": ...}.
        """
        if ccx_cmd is None:
            ccx_cmd = find_ccx()
        if ccx_cmd is None:
            raise RuntimeError("CalculiX (ccx) not found. Set ccx_cmd explicitly.")

        work_dir = Path(inp_path).parent
        stem = Path(inp_path).stem

        # CalculiX takes the stem (no .inp extension).
        # Under WSL2 with a .exe, convert to Windows path.
        if ccx_cmd.endswith(".exe"):
            inp_arg = _wsl_to_win(str(work_dir / stem))
        else:
            inp_arg = str(work_dir / stem)

        cmd = [ccx_cmd, inp_arg]
        logger.info(f"[VoxelFEM] Running: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, cwd=str(work_dir),
            )
            if proc.returncode != 0:
                logger.warning(f"[VoxelFEM] ccx returned non-zero exit code {proc.returncode}")
                logger.debug(f"[VoxelFEM] stderr: {proc.stderr[:500]}")
                return {
                    "stress_max": FEM_SENTINEL, "displacement_max": FEM_SENTINEL,
                    "compliance": FEM_SENTINEL, "failure_reason": "ccx_error",
                }
        except subprocess.TimeoutExpired:
            logger.error(f"[VoxelFEM] ccx timed out after {timeout}s")
            return {
                "stress_max": FEM_SENTINEL, "displacement_max": FEM_SENTINEL,
                "compliance": FEM_SENTINEL, "failure_reason": "timeout",
            }

        frd_path = work_dir / f"{stem}.frd"
        if not frd_path.exists():
            logger.error(f"[VoxelFEM] .frd not found: {frd_path}")
            return {
                "stress_max": FEM_SENTINEL, "displacement_max": FEM_SENTINEL,
                "compliance": FEM_SENTINEL, "failure_reason": "no_frd",
            }

        result = VoxelHexMesher._parse_frd(str(frd_path))
        result["failure_reason"] = None
        return result

    @staticmethod
    def _parse_frd(frd_path: str) -> Dict[str, float]:
        """
        Parse CalculiX .frd ASCII output for max von Mises stress and displacement.

        CalculiX .frd record codes:
          -4  DISP / -4  STRESS  — block header
          -5                     — component name line
          -1                     — data record (fixed-width: 3+10+12*N chars)
          -2 / -3                — end of block

        Fixed-width layout of -1 records (no spaces between values):
          chars  0- 2: record type (' -1')
          chars  3-12: node ID (10 chars)
          chars 13-24: value 1 (12 chars)
          chars 25-36: value 2 (12 chars)
          ... and so on in 12-char slots
        """
        displacements = []
        stresses = []
        in_disp = False
        in_stress = False

        def _read_val(line: str, slot: int) -> float:
            """Extract value from fixed-width slot (0-indexed after node ID)."""
            start = 13 + slot * 12
            return float(line[start:start + 12])

        try:
            with open(frd_path) as f:
                for line in f:
                    stripped = line.strip()

                    if "-4  DISP" in line or "-4 DISP" in line:
                        in_disp, in_stress = True, False
                        continue
                    if "-4  STRESS" in line or "-4 STRESS" in line:
                        in_stress, in_disp = True, False
                        continue
                    if stripped.startswith("-3") or stripped.startswith("-2"):
                        in_disp = in_stress = False
                        continue
                    if stripped.startswith("-5") or stripped.startswith("-4"):
                        continue

                    if line.startswith(" -1") and len(line) > 25:
                        if in_disp:
                            try:
                                dx = _read_val(line, 0)
                                dy = _read_val(line, 1)
                                dz = _read_val(line, 2)
                                displacements.append((dx**2 + dy**2 + dz**2) ** 0.5)
                            except (ValueError, IndexError):
                                pass
                        elif in_stress and len(line) >= 85:
                            try:
                                s11 = _read_val(line, 0)
                                s22 = _read_val(line, 1)
                                s33 = _read_val(line, 2)
                                s12 = _read_val(line, 3)
                                s23 = _read_val(line, 4)
                                s13 = _read_val(line, 5)
                                vm = (0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2
                                             + 6*(s12**2 + s23**2 + s13**2))) ** 0.5
                                stresses.append(vm)
                            except (ValueError, IndexError):
                                pass
        except OSError as e:
            logger.error(f"[VoxelFEM] Could not read .frd: {e}")
            return {
                "stress_max": FEM_SENTINEL, "displacement_max": FEM_SENTINEL,
                "compliance": FEM_SENTINEL, "failure_reason": "no_frd",
            }

        stress_max   = float(max(stresses))      if stresses      else 0.0
        disp_max     = float(max(displacements)) if displacements else 0.0
        compliance   = float(sum(displacements)) if displacements else 0.0

        logger.info(
            f"[VoxelFEM] stress_max={stress_max:.2f} MPa  "
            f"disp_max={disp_max:.4f} mm  compliance={compliance:.4f}"
        )
        return {"stress_max": stress_max, "displacement_max": disp_max, "compliance": compliance}


# ── VoxelFEMEvaluator ──────────────────────────────────────────────────────────

class VoxelFEMEvaluator:
    """FEM evaluator using direct CalculiX hex mesh — bypasses FreeCAD."""

    def __init__(
        self,
        ccx_cmd: str = None,
        output_dir: str = None,
        fixed_face: str = "x_min",
        load_face: str = "x_max",
        force_n: float = 1000.0,
        vae_model=None,
    ):
        self.ccx_cmd = ccx_cmd or find_ccx()
        # Default to a Windows-accessible temp dir when ccx is a Windows .exe,
        # otherwise use a local Linux directory.
        if output_dir is None:
            if self.ccx_cmd and self.ccx_cmd.endswith(".exe"):
                output_dir = "/mnt/c/Windows/Temp/voxel_fem"
            else:
                output_dir = "./optimization_results/voxel_fem"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_face = fixed_face
        self.load_face  = load_face
        self.force_n    = force_n
        self.vae_model  = vae_model
        self.evaluation_history = []
        self._counter   = 0

        if self.ccx_cmd is None:
            logger.warning("[VoxelFEM] CalculiX not found — evaluations will return penalty values.")

    def evaluate(self, z: np.ndarray, vae_model, bbox: dict = None) -> Dict[str, float]:
        """
        Decode latent z → voxels → hex FEM → {stress, compliance, mass}.

        z:         latent vector (latent_dim,) or (1, latent_dim)
        vae_model: a DesignVAE with .decode() method
        bbox:      optional physical bounding box dict
        """
        import torch

        device = next(vae_model.parameters()).device

        with torch.no_grad():
            z_t = torch.from_numpy(np.array(z).reshape(1, -1)).float().to(device)
            voxels = vae_model.decode(z_t).cpu().numpy().squeeze()

        # Organic density filter (matches DesignOptimizer.decode_latent_to_geometry)
        voxels = smooth_voxels(voxels)

        self._counter += 1
        inp_path = str(self.output_dir / f"voxel_fem_{self._counter:04d}.inp")

        try:
            VoxelHexMesher.voxels_to_inp(
                voxels, bbox=bbox,
                fixed_face=self.fixed_face,
                load_face=self.load_face,
                force_n=self.force_n,
                output_path=inp_path,
            )
        except ValueError as e:
            logger.warning(f"[VoxelFEM] Mesh failed: {e}")
            return {"stress": FEM_SENTINEL, "compliance": FEM_SENTINEL, "mass": 1.0,
                    "failure_reason": "mesh_error"}

        if self.ccx_cmd is None:
            return {"stress": FEM_SENTINEL, "compliance": FEM_SENTINEL, "mass": 1.0,
                    "failure_reason": "no_ccx"}

        fem_res = VoxelHexMesher.run_ccx(inp_path, ccx_cmd=self.ccx_cmd)
        # Propagate failure sentinel from run_ccx
        if not is_valid_fem_result({"stress": fem_res["stress_max"]}):
            return {"stress": fem_res["stress_max"], "compliance": fem_res["compliance"],
                    "mass": 1.0, "failure_reason": fem_res.get("failure_reason", "fem_failed")}
        stress     = fem_res["stress_max"]
        compliance = fem_res["compliance"]

        # Mass from volume fraction × total volume × steel density
        solid_frac = float((voxels > 0.5).mean())
        D, H, W = voxels.shape
        if bbox:
            vol_mm3 = ((bbox["x"][1] - bbox["x"][0]) *
                       (bbox["y"][1] - bbox["y"][0]) *
                       (bbox["z"][1] - bbox["z"][0])) * solid_frac
        else:
            vol_mm3 = D * H * W * solid_frac
        mass = vol_mm3 * 7900 / 1e9   # kg/mm³ × mm³ → kg

        result = {"stress": stress, "compliance": compliance, "mass": mass}
        self.evaluation_history.append({
            "z": z.tolist() if hasattr(z, "tolist") else list(z),
            "results": result,
        })
        return result

    def evaluate_batch(self, param_list: list) -> list:
        """BridgeEvaluator-compatible interface. Reads 'z' from each param dict."""
        if self.vae_model is None:
            raise RuntimeError("VoxelFEMEvaluator.evaluate_batch requires vae_model set at construction")
        results = []
        for p in param_list:
            z = p.get("z")
            if z is None:
                results.append({"stress": FEM_SENTINEL, "compliance": FEM_SENTINEL,
                                 "mass": 1.0, "failure_reason": "no_z", "parameters": p})
                continue
            res = self.evaluate(np.asarray(z), self.vae_model)
            res["parameters"] = p
            results.append(res)
        return results

    def save_history(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2, cls=_NumpyEncoder)


# ── CLI test mode ──────────────────────────────────────────────────────────────

def _run_test():
    """Unit test: 10×10×10 solid cube, fixed x_min, loaded x_max (-Z force)."""
    import tempfile
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=== VoxelHexMesher test: 10×10×10 solid cube ===")
    voxels = np.ones((10, 10, 10), dtype=np.float32)
    bbox   = {"x": [0.0, 100.0], "y": [0.0, 20.0], "z": [0.0, 20.0]}

    with tempfile.TemporaryDirectory() as tmp:
        inp_path = os.path.join(tmp, "test_cube.inp")
        VoxelHexMesher.voxels_to_inp(
            voxels, bbox=bbox,
            fixed_face="x_min", load_face="x_max",
            force_n=1000.0, output_path=inp_path,
        )
        size = os.path.getsize(inp_path)
        print(f"  .inp written: {inp_path}  ({size} bytes)")

        with open(inp_path) as f:
            content = f.read()
        assert "*NODE" in content,    ".inp missing *NODE"
        assert "*ELEMENT" in content, ".inp missing *ELEMENT"
        assert "C3D8" in content,     ".inp missing C3D8 element type"
        assert "*BOUNDARY" in content,".inp missing *BOUNDARY"
        assert "*CLOAD" in content,   ".inp missing *CLOAD"
        print("  .inp content checks: PASS")

        ccx = find_ccx()
        if ccx:
            print(f"  Found ccx: {ccx}")
            results = VoxelHexMesher.run_ccx(inp_path, ccx_cmd=ccx, timeout=120)
            print(f"  FEM results: {results}")
            if results["displacement_max"] > 0:
                print("  PASS: non-zero displacements confirmed")
            else:
                print("  WARNING: zero displacements — check ccx output / .frd parser")
        else:
            print("  ccx not found — skipping FEM run")
            print("  PASS: .inp file generated successfully")

    print("=== Test complete ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voxel FEM — direct CalculiX path")
    parser.add_argument("--test", action="store_true", help="Run unit test on 10×10×10 cube")
    args = parser.parse_args()
    if args.test:
        _run_test()
    else:
        print("Usage: python voxel_fem.py --test")
