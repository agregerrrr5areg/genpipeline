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
    "/mnt/c/Program Files/FreeCAD 1.0/bin/FreeCADCmd.exe",
    "/mnt/c/Program Files/FreeCAD 0.21/bin/FreeCADCmd.exe",
    "/mnt/c/Program Files/FreeCAD 0.20/bin/FreeCADCmd.exe",
    "/mnt/c/Program Files (x86)/FreeCAD 1.0/bin/FreeCADCmd.exe",
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
    p = str(wsl_path)
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
        candidate = Path(override) / "bin" / "FreeCADCmd.exe"
        if candidate.exists():
            logger.info(f"Using FreeCAD at: {candidate}")
            return str(candidate)
        raise FileNotFoundError(
            f"FreeCADCmd.exe not found at {candidate}\n"
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

    cmd = [freecad_cmd, extractor_win,
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

def build_dataset(output_dir: Path, voxel_resolution: int = 32):
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
    split     = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [split, len(dataset) - split]
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WSL2 bridge: extract FEM data from Windows FreeCAD"
    )
    parser.add_argument(
        "--designs-dir", required=True,
        help="Directory containing .FCStd files (WSL path, e.g. /mnt/c/Users/you/designs)"
    )
    parser.add_argument(
        "--output-dir", default="./fem_data",
        help="Output directory for JSON, STL, and dataset (WSL path)"
    )
    parser.add_argument(
        "--freecad-path", default=None,
        help='Override FreeCAD install dir (WSL path, e.g. "/mnt/c/Program Files/FreeCAD 1.0")'
    )
    parser.add_argument("--voxel-resolution", type=int, default=32)
    args = parser.parse_args()

    # ── Find FreeCAD ──────────────────────────────────────────────────────
    try:
        freecad_cmd = find_freecad_cmd(args.freecad_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # ── Deploy extractor ──────────────────────────────────────────────────
    extractor_win = deploy_extractor()

    # ── Collect .FCStd files ──────────────────────────────────────────────
    designs_dir = Path(args.designs_dir)
    if not designs_dir.exists():
        logger.error(f"Designs directory not found: {designs_dir}")
        return

    fcstd_files = sorted(designs_dir.glob("**/*.FCStd"))
    logger.info(f"Found {len(fcstd_files)} .FCStd files in {designs_dir}")
    if not fcstd_files:
        logger.error("No .FCStd files found.")
        return

    # ── Extract ───────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
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

    # ── Build dataset ──────────────────────────────────────────────────────
    train_loader, val_loader = build_dataset(output_dir, args.voxel_resolution)
    if train_loader:
        logger.info("Dataset ready. Next step:")
        logger.info("  python quickstart.py --step 3 --epochs 100")


if __name__ == "__main__":
    main()
