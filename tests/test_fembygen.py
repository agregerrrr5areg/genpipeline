
import logging
from pathlib import Path
from genpipeline.freecad_bridge import find_freecad_cmd, run_variant, deploy_variant_script, wsl_to_windows
from genpipeline.config import load_config

logging.basicConfig(level=logging.INFO)

def test():
    config = load_config("pipeline_config.json")
    try:
        freecad_cmd = find_freecad_cmd(config.freecad_path)
    except FileNotFoundError as e:
        print(f"FreeCAD not found: {e}")
        return

    output_dir = Path("./fem_test_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    variant_win = deploy_variant_script()
    
    print(f"Running FEMbyGEN variant via bridge...")
    result = run_variant(
        freecad_cmd=freecad_cmd,
        h_mm=15.0,
        r_mm=3.0,
        output_wsl=str(output_dir),
        variant_win=variant_win,
        geometry="cantilever"
    )
    
    if result:
        print(f"SUCCESS: {result}")
    else:
        print("FAILED to run variant")

if __name__ == "__main__":
    test()
