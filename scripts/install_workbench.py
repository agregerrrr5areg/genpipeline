
import os
import shutil
import sys

def install():
    print("--- GenDesign Workbench Installer ---")
    
    # 1. Detection
    freecad_path = r"C:\Users\PC-PC\AppData\Local\Programs\FreeCAD 1.0"
    dest_path = os.path.join(freecad_path, "Mod", "GenDesign")
    
    # Convert to WSL path if needed (this script runs on Windows)
    if not os.path.exists(os.path.join(freecad_path, "bin")):
        print(f"ERROR: FreeCAD not found at {freecad_path}")
        return

    # 2. Source path (current directory)
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "freecad_workbench")
    if not os.path.exists(src_path):
        # If run from within the workbench folder
        src_path = os.path.dirname(os.path.abspath(__file__))

    print(f"Source: {src_path}")
    print(f"Destination: {dest_path}")

    # 3. Clean and Copy
    try:
        if os.path.exists(dest_path):
            print("Cleaning old installation...")
            shutil.rmtree(dest_path)
        
        print("Copying workbench files...")
        shutil.copytree(src_path, dest_path)
        print("SUCCESS: Workbench deployed.")
        print("
Please RESTART FreeCAD.")
        print("Check C:\Users\PC-PC\Documents\gendesign_debug.log if the menu does not appear.")
        
    except Exception as e:
        print(f"INSTALLATION FAILED: {e}")

if __name__ == "__main__":
    install()
