#!/usr/bin/env bash
# deploy.sh — Install / update the GenDesign workbench into FreeCAD's Mod directory.
# Run from WSL2. Copies the workbench files to the Windows FreeCAD Mod folder.
#
#   ./freecad_workbench/deploy.sh
#   ./freecad_workbench/deploy.sh --freecad-path "/mnt/c/Program Files/FreeCAD 1.0"

set -euo pipefail

FREECAD_DEFAULT="/mnt/c/Users/PC-PC/AppData/Local/Programs/FreeCAD 1.0"
FREECAD_PATH="${1:-${FREECAD_DEFAULT}}"
DEST="${FREECAD_PATH}/Mod/GenDesign"
SRC="$(cd "$(dirname "$0")" && pwd)"

echo "[deploy] Source : $SRC"
echo "[deploy] Dest   : $DEST"

if [[ ! -d "${FREECAD_PATH}/bin" ]]; then
    echo "[deploy] ERROR: FreeCAD not found at '${FREECAD_PATH}'"
    echo "         Pass the correct path as first argument:"
    echo "         $0 '/mnt/c/Program Files/FreeCAD 1.0'"
    exit 1
fi

mkdir -p "$DEST"
for f in __init__.py gendesign_wb.py constraint_obj.py load_obj.py \
          seed_part.py export_pipeline.py commands.py; do
    cp "${SRC}/${f}" "${DEST}/${f}"
    echo "[deploy] Copied $f"
done

echo "[deploy] Done. Restart FreeCAD and switch to the GenDesign workbench."
