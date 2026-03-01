"""
WSL2 → Windows FreeCAD bridge.

Finds FreeCADCmd.exe on the Windows filesystem, runs the FEM extraction
script inside it for every .FCStd file in a directory, then builds a
PyTorch dataset ready for VAE training.

Usage:
    source venv/bin/activate

    # Auto-detect FreeCAD, designs on Windows C: drive
    python freecad_bridge.py --designs-dir /mnt/c/Users/YOU/designs

    # Specify FreeCAD install path explicitly
    python freecad_bridge.py \
        --designs-dir  /mnt/c/Users/YOU/designs \
        --output-dir   ./fem_data \
        --freecad-path "/mnt/c/Program Files/FreeCAD 1.0"

    # Then train:
    python quickstart.py --step 3 --epochs 100
"""

import argparse
import concurrent.futures
import json
import logging
import shutil
import subprocess
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── FreeCAD search paths (WSL /mnt/c/ view of Windows C:\) ────────────────────
FREECAD_SEARCH_PATHS = [
    # Per-user install (FreeCAD 1.0 on this machine)
    "/mnt/c/Users/PC-PC/AppData/Local/Programs/FreeCAD 1.0/bin/freecad.exe",
    # System-wide installs
    "/mnt/c/Program Files/FreeCAD 1.0/bin/freecad.exe",
    "/mnt/c/Program Files (x86)/FreeCAD 1.0/bin/freecad.exe",
    # FreeCAD 0.x headless
    "/mnt/c/Program Files/FreeCAD 0.21/bin/FreeCADCmd.exe",
    "/mnt/c/Program Files/FreeCAD 0.20/bin/FreeCADCmd.exe",
    "/mnt/c/Program Files (x86)/FreeCAD 0.21/bin/FreeCADCmd.exe",
]

EXTRACTOR_SCRIPT = Path(__file__).parent / "freecad_scripts" / "extract_fem.py"
# Windows Temp is accessible from both Windows and WSL
WIN_TEMP_WSL  = Path("/mnt/c/Windows/Temp")
WIN_TEMP_WIN  = "C:\\Windows\\Temp"


# ── Path helpers ───────────────────────────────────────────────────────────────

def wsl_to_windows(wsl_path: str) -> str:
    """
    Convert a WSL path to a Windows path string.
      /mnt/c/foo/bar  →  C:\\foo\\bar
      /home/user/...  →  \\\\wsl.localhost\\<distro>\\home\\user\\...
    """
    # Resolve to absolute so relative paths (./fem_data) become /home/...
    p = str(Path(wsl_path).resolve())
    if p.startswith("/mnt/") and len(p) > 6 and p[6] in ("/", ""):
        drive  = p[5].upper()
        rest   = p[6:].replace("/", "\\")
        return f"{drive}:{rest}"
    if p.startswith("/mnt/") and len(p) > 5:
        drive  = p[5].upper()
        rest   = p[6:].replace("/", "\\") if len(p) > 6 else ""
        return f"{drive}:{rest}"
    # Native WSL path — use UNC
    distro = _wsl_distro_name()
    return f"\\\\wsl.localhost\\{distro}" + p.replace("/", "\\")


def _wsl_distro_name() -> str:
    """Best-effort WSL distro name for UNC paths."""
    try:
        text = Path("/etc/os-release").read_text()
        for line in text.splitlines():
            if line.startswith("PRETTY_NAME=") or line.startswith("NAME="):
                name = line.split("=", 1)[1].strip().strip('"')
                # e.g. "Ubuntu 22.04.3 LTS" → "Ubuntu-22.04"
                parts = name.split()
                if len(parts) >= 2 and parts[1][0].isdigit():
                    ver = parts[1].rsplit(".", 1)[0]  # "22.04"
                    return f"{parts[0]}-{ver}"
                return parts[0]
    except Exception:
        pass
    return "Ubuntu"


# ── FreeCAD detection ──────────────────────────────────────────────────────────

def find_freecad_cmd(override: str = None) -> str:
    if override:
        # FreeCAD 1.0: freecad.exe (run with --console for headless)
        # FreeCAD 0.x: FreeCADCmd.exe
        for exe in ("freecad.exe", "FreeCADCmd.exe"):
            candidate = Path(override) / "bin" / exe
            if candidate.exists():
                logger.info(f"Using FreeCAD at: {candidate}")
                return str(candidate)
        raise FileNotFoundError(
            f"freecad.exe / FreeCADCmd.exe not found under {override}/bin/\n"
            f"Check your --freecad-path argument."
        )

    for path in FREECAD_SEARCH_PATHS:
        if Path(path).exists():
            logger.info(f"Found FreeCAD: {path}")
            return path

    raise FileNotFoundError(
        "FreeCADCmd.exe not found on Windows filesystem.\n"
        "Install FreeCAD on Windows (https://www.freecad.org) or use --freecad-path.\n"
        "Searched:\n" + "\n".join(f"  {p}" for p in FREECAD_SEARCH_PATHS)
    )


# ── Extraction ─────────────────────────────────────────────────────────────────

def deploy_extractor() -> str:
    """Copy the extraction script to Windows Temp so FreeCADCmd.exe can read it."""
    dst = WIN_TEMP_WSL / "extract_fem.py"
    shutil.copy(EXTRACTOR_SCRIPT, dst)
    logger.info(f"Extraction script deployed to {dst}")
    return WIN_TEMP_WIN + "\\extract_fem.py"


def run_extraction(freecad_cmd: str, fcstd_wsl: str,
                   output_wsl: str, extractor_win: str) -> dict | None:
    """Run FreeCADCmd.exe to extract one .FCStd file. Returns parsed JSON or None."""
    fcstd_win  = wsl_to_windows(fcstd_wsl)
    output_win = wsl_to_windows(output_wsl)

    console_flag = ["--console"] if freecad_cmd.endswith("freecad.exe") else []
    cmd = [freecad_cmd] + console_flag + [extractor_win,
           "--input",  fcstd_win,
           "--output", output_win]

    logger.info(f"  CMD: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.error("FreeCAD timed out after 300 s")
        return None

    # Print FreeCAD stdout for visibility
    for line in proc.stdout.splitlines():
        logger.info(f"  [freecad] {line}")
    if proc.returncode != 0:
        logger.error(f"FreeCAD exited {proc.returncode}:\n{proc.stderr}")
        return None

    stem      = Path(fcstd_wsl).stem
    json_path = Path(output_wsl) / f"{stem}_fem_results.json"
    if not json_path.exists():
        logger.warning(f"No JSON output found at {json_path}")
        return None

    with open(json_path) as f:
        return json.load(f)


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_dataset(output_dir: Path, voxel_resolution: int = 64):
    from fem_data_pipeline import FEMDataset, DesignSample, VoxelGrid

    json_files = sorted(output_dir.glob("*_fem_results.json"))
    logger.info(f"Building dataset from {len(json_files)} result files...")

    if not json_files:
        logger.error("No *_fem_results.json files found — nothing to build.")
        return None, None

    voxelizer = VoxelGrid(resolution=voxel_resolution)
    samples   = []

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        stl_path = Path(str(jf).replace("_fem_results.json", "_mesh.stl"))
        if not stl_path.exists():
            logger.warning(f"  Skipping {jf.name} — no matching STL")
            continue

        voxel = voxelizer.mesh_to_voxel(str(stl_path))

        samples.append(DesignSample(
            geometry_path=str(stl_path),
            stress_max=data.get("stress_max",  0.0),
            stress_mean=data.get("stress_mean", 0.0),
            compliance=data.get("compliance",   0.0),
            mass=data.get("mass",               1.0),
            parameters=data.get("parameters",   {}),
            voxel_grid=voxel,
        ))

    logger.info(f"Loaded {len(samples)} samples with voxel grids")

    if not samples:
        logger.error("No samples could be loaded (all missing STL?).")
        return None, None

    dataset   = FEMDataset(samples, voxel_resolution=voxel_resolution)
    n = len(dataset)
    split     = max(1, min(n - 1, int(0.8 * n)))  # at least 1 in each split
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [split, n - split]
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=8, shuffle=True,  num_workers=0
    )
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=8, shuffle=False, num_workers=0
    )

    out = output_dir / "fem_dataset.pt"
    torch.save({
        "dataset":      dataset,
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "samples":      samples,
    }, out)
    logger.info(f"Dataset saved → {out}")
    logger.info(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
    return train_loader, val_loader


# ── Variant generator ──────────────────────────────────────────────────────────

VARIANT_SCRIPT = Path(__file__).parent / "freecad_scripts" / "run_fem_variant.py"


def deploy_variant_script() -> str:
    """Copy the variant script to Windows Temp so FreeCAD can access it."""
    dst = WIN_TEMP_WSL / "run_fem_variant.py"
    shutil.copy(VARIANT_SCRIPT, dst)
    logger.info(f"Variant script deployed to {dst}")
    return WIN_TEMP_WIN + "\\run_fem_variant.py"


def run_variant(freecad_cmd: str, h_mm: float, r_mm: float,
                output_wsl: str, variant_win: str, geometry: str = "cantilever", material_cfg: dict = None) -> dict | None:
    """Run one FEM variant via FreeCADCmd. Includes material properties if provided."""
    output_win  = wsl_to_windows(output_wsl)

    # Per-geometry BC normals for run_fem_variant.py
    _GEOM_BC = {
        "cantilever": {"fixed_face_normal": [-1,0,0], "load_face_normal": [1,0,0], "force_direction": [0,0,-1]},
        "tapered":    {"fixed_face_normal": [-1,0,0], "load_face_normal": [1,0,0], "force_direction": [0,0,-1]},
        "ribbed":     {"fixed_face_normal": [-1,0,0], "load_face_normal": [1,0,0], "force_direction": [0,0,-1]},
        "lbracket":   {"fixed_face_normal": [0,0,-1], "load_face_normal": [1,0,0], "force_direction": [0,0,-1]},
    }

    # Use a unique ID to prevent collisions during parallel runs
    import uuid
    uid  = str(uuid.uuid4())[:8]
    stem = f"{geometry[:4]}_h{h_mm:.1f}_r{r_mm:.1f}_{uid}".replace(".", "p")

    # Prepare config payload
    from sim_config import SIM_PHYSICS
    cfg_data = {
        "h_mm": h_mm,
        "r_mm": r_mm,
        "output": output_win,
        "geometry": geometry,
        "stem": stem,   # script uses this for consistent output filenames
    }
    cfg_data.update(_GEOM_BC.get(geometry, _GEOM_BC["cantilever"]))
    cfg_data.update(SIM_PHYSICS)
    if material_cfg:
        cfg_data.update(material_cfg)
    
    # FreeCAD 1.0 intercepts --flag style args; pass params via a config file.
    cfg_wsl    = WIN_TEMP_WSL / f"fem_cfg_{stem}.cfg"
    cfg_win    = WIN_TEMP_WIN + f"\\fem_cfg_{stem}.cfg"
    with open(cfg_wsl, "w") as f:
        json.dump(cfg_data, f)

    # FreeCAD 1.0 (freecad.exe) needs --console for headless; 0.x FreeCADCmd.exe doesn't
    console_flag = ["--console"] if freecad_cmd.endswith("freecad.exe") else []
    cmd = [freecad_cmd] + console_flag + [variant_win, cfg_win]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.error(f"FreeCAD timed out for {stem}")
        return None

    for line in proc.stdout.splitlines():
        if "[run_fem_variant]" in line:
            logger.info(f"    {line}")
    if proc.returncode != 0:
        logger.error(f"FreeCAD exited {proc.returncode}:\n{proc.stderr[-500:]}")
        return None

    json_path = Path(output_wsl) / f"{stem}_fem_results.json"
    if not json_path.exists():
        logger.warning(f"No JSON at {json_path}")
        return None

    with open(json_path) as f:
        return json.load(f)


def generate_variants(freecad_cmd: str, output_dir: Path,
                      n: int = 50,
                      h_range: tuple = (5.0, 20.0),
                      r_range: tuple = (0.0,  8.0),
                      voxel_resolution: int = 64,
                      seed: int = 42,
                      n_workers: int = 4,
                      geometry_types: list = None):
    """
    Generate n FEM variants by sampling (h_mm, r_mm) uniformly at random.

    Args:
        freecad_cmd:      path to FreeCADCmd.exe (WSL path)
        output_dir:       WSL directory for JSON/STL output
        n:                number of variants to generate
        h_range:          (min, max) beam height in mm
        r_range:          (min, max) hole radius in mm (0 = solid)
        voxel_resolution: voxel grid size passed to build_dataset()
        seed:             RNG seed for reproducibility
        n_workers:        number of parallel processes
        geometry_types:   list of geometry types to cycle through
    """
    import random
    random.seed(seed)
    if geometry_types is None:
        geometry_types = ["cantilever"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-generate parameters to ensure reproducibility
    tasks = []
    for i in range(n):
        h = round(random.uniform(*h_range), 2)
        r = round(random.uniform(*r_range), 2)
        geom = geometry_types[i % len(geometry_types)]
        tasks.append((h, r, geom))

    ok, fail = 0, 0
    logger.info(f"Starting generation of {n} variants with {n_workers} workers...")
    variant_win = deploy_variant_script()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        # map run_variant across tasks
        future_to_params = {
            executor.submit(run_variant, freecad_cmd, h, r, str(output_dir), variant_win, geom): (h, r, geom)
            for h, r, geom in tasks
        }

        for i, future in enumerate(concurrent.futures.as_completed(future_to_params)):
            h, r, geom = future_to_params[future]
            try:
                result = future.result()
                if result:
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                logger.error(f"Task {geom} h={h} r={r} raised exception: {e}")
                fail += 1

            if (i + 1) % 5 == 0 or (i + 1) == n:
                logger.info(f"Progress: {i+1}/{n} completed ({ok} ok, {fail} fail)")

    logger.info(f"Generation: {ok} succeeded, {fail} failed")
    train_loader, val_loader = build_dataset(output_dir, voxel_resolution)
    if train_loader:
        logger.info("Dataset ready. Next step:")
        logger.info("  python quickstart.py --step 3 --epochs 100")
    return train_loader, val_loader


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WSL2 bridge: extract FEM data from Windows FreeCAD"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── extract: pull results from existing .FCStd files ──────────────────
    ext_p = subparsers.add_parser("extract", help="Extract FEM from existing .FCStd files")
    ext_p.add_argument("--designs-dir", required=True)
    ext_p.add_argument("--output-dir",  default="./fem_data")
    ext_p.add_argument("--freecad-path", default=None)
    ext_p.add_argument("--voxel-resolution", type=int, default=32)

    # ── generate: create parametric cantilever variants from scratch ───────
    gen_p = subparsers.add_parser("generate",
        help="Generate parametric cantilever beam variants and run FEM")
    gen_p.add_argument("--n-variants",       type=int,   default=50)
    gen_p.add_argument("--h-min",            type=float, default=5.0,
                       help="Min beam height mm")
    gen_p.add_argument("--h-max",            type=float, default=20.0,
                       help="Max beam height mm")
    gen_p.add_argument("--r-min",            type=float, default=0.0,
                       help="Min hole radius mm (0 = solid)")
    gen_p.add_argument("--r-max",            type=float, default=8.0,
                       help="Max hole radius mm")
    gen_p.add_argument("--output-dir",       default="./fem_data")
    gen_p.add_argument("--freecad-path",     default=None)
    gen_p.add_argument("--voxel-resolution", type=int, default=32)
    gen_p.add_argument("--seed",             type=int, default=42)
    gen_p.add_argument("--n-workers",        type=int, default=4, help="Number of parallel FreeCAD processes")
    gen_p.add_argument("--geometry-types", nargs="+",
                       default=["cantilever"],
                       choices=["cantilever", "lbracket", "tapered", "ribbed"],
                       help="Geometry types to cycle through")

    # ── run: single variant ──────────────────────────────────────────────
    run_p = subparsers.add_parser("run", help="Run a single FEM variant")
    run_p.add_argument("--h-mm", type=float, required=True)
    run_p.add_argument("--r-mm", type=float, required=True)
    run_p.add_argument("--geometry", default="cantilever")
    run_p.add_argument("--output-dir", default="./fem_data")
    run_p.add_argument("--freecad-path", default=None)

    # Backwards-compat: no subcommand → treat as extract (requires --designs-dir)
    parser.add_argument("--designs-dir",     default=None)
    parser.add_argument("--output-dir",      default="./fem_data")
    parser.add_argument("--freecad-path",    default=None)
    parser.add_argument("--voxel-resolution",type=int, default=32)

    args = parser.parse_args()

    # ── Find FreeCAD ──────────────────────────────────────────────────────
    freecad_path_arg = getattr(args, "freecad_path", None)
    try:
        freecad_cmd = find_freecad_cmd(freecad_path_arg)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    output_dir = Path(getattr(args, "output_dir", "./fem_data"))

    # ── generate subcommand ───────────────────────────────────────────────
    if args.command == "generate":
        generate_variants(
            freecad_cmd  = freecad_cmd,
            output_dir   = output_dir,
            n            = args.n_variants,
            h_range      = (args.h_min, args.h_max),
            r_range      = (args.r_min, args.r_max),
            voxel_resolution = args.voxel_resolution,
            seed         = args.seed,
            n_workers    = args.n_workers,
            geometry_types = args.geometry_types,
        )
        return

    # ── run subcommand ──────────────────────────────────────────────────
    if args.command == "run":
        variant_win = deploy_variant_script()
        result = run_variant(
            freecad_cmd,
            args.h_mm,
            args.r_mm,
            str(output_dir),
            variant_win,
            geometry=args.geometry
        )
        if result:
            logger.info(f"Success: {result}")
        else:
            logger.error("Failed to run variant")
        return

    # ── extract subcommand (or legacy positional mode) ────────────────────
    designs_dir_arg = getattr(args, "designs_dir", None)
    if not designs_dir_arg:
        parser.error("Specify a subcommand (generate / extract) or --designs-dir")

    extractor_win = deploy_extractor()

    designs_dir = Path(designs_dir_arg)
    if not designs_dir.exists():
        logger.error(f"Designs directory not found: {designs_dir}")
        return

    fcstd_files = sorted(designs_dir.glob("**/*.FCStd"))
    logger.info(f"Found {len(fcstd_files)} .FCStd files in {designs_dir}")
    if not fcstd_files:
        logger.error("No .FCStd files found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for i, fcstd in enumerate(fcstd_files):
        logger.info(f"[{i+1}/{len(fcstd_files)}] {fcstd.name}")
        result = run_extraction(freecad_cmd, str(fcstd), str(output_dir), extractor_win)
        if result:
            ok += 1
        else:
            fail += 1

    logger.info(f"Extraction: {ok} succeeded, {fail} failed")

    train_loader, val_loader = build_dataset(output_dir, args.voxel_resolution)
    if train_loader:
        logger.info("Dataset ready. Next step:")
        logger.info("  python quickstart.py --step 3 --epochs 100")


if __name__ == "__main__":
    main()
