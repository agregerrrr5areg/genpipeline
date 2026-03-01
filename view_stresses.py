import pyvista as pv
import json
import glob
import os
from pathlib import Path
import numpy as np
import argparse

def visualize_stress(result_json_path):
    """
    Renders a 3D STL mesh with a color heatmap based on the stress data.
    """
    # 1. Load Data
    with open(result_json_path, 'r') as f:
        data = json.load(f)
    
    stl_path = result_json_path.replace('_fem_results.json', '_mesh.stl')
    if not os.path.exists(stl_path):
        print(f"Error: STL mesh not found for {result_json_path}")
        return

    stress_max = data.get('stress_max', 0.0)
    mass = data.get('mass', 0.0)
    
    print(f"📦 Visualizing Design: {Path(stl_path).name}")
    print(f"🔥 Max Stress: {stress_max:.2f} MPa")
    print(f"⚖️  Mass: {mass:.4f} kg")

    # 2. Load Mesh
    mesh = pv.read(stl_path)
    
    # 3. Simulate Stress Distribution
    # Note: Real nodal stress data is inside FreeCAD. Since we only have the mesh + max_stress, 
    # we simulate a realistic distribution (highest stress at the fixed end x=0)
    # for visualization purposes. 
    # For a cantilever, stress is highest at the root (x_min) and lowest at the tip (x_max).
    bounds = mesh.bounds
    x_coords = mesh.points[:, 0]
    
    # Normalize X to [0, 1] where 1 is the root (highest stress)
    stress_dist = 1.0 - ((x_coords - bounds[0]) / (bounds[1] - bounds[0]))
    
    # Scale by the actual max stress from CalculiX
    simulated_stresses = stress_dist * stress_max
    
    mesh.point_data['Stress (MPa)'] = simulated_stresses

    # 4. Render
    plotter = pv.Plotter(title=f"Design Stress Heatmap - {stress_max:.1f} MPa")
    plotter.add_mesh(mesh, scalars='Stress (MPa)', cmap='jet', clim=[0, stress_max], 
                     show_edges=True, edge_color='black', line_width=0.5)
    
    plotter.add_text(f"Max Stress: {stress_max:.1f} MPa
Mass: {mass:.3f} kg", position='upper_left', font_size=10)
    
    # Add a floor for context
    plotter.add_floor()
    plotter.show_axes()
    
    print("Opening 3D Viewer...")
    plotter.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest", action='store_true', help="View the absolute latest simulation")
    parser.add_argument("--best", action='store_true', help="View the strongest design found so far")
    parser.add_argument("--path", type=str, help="Path to a specific results JSON")
    args = parser.parse_args()

    results_dir = "genpipeline/optimization_results/fem"
    
    if args.path:
        visualize_stress(args.path)
    elif args.best:
        files = glob.glob(f"{results_dir}/*.json")
        if not files:
            print("No designs found.")
        else:
            # Find strongest (min stress)
            best_file = min(files, key=lambda x: json.load(open(x))['stress_max'])
            visualize_stress(best_file)
    else:
        files = glob.glob(f"{results_dir}/*.json")
        if not files:
            print("No designs found.")
        else:
            # Sort by file time
            latest_file = max(files, key=os.path.getctime)
            visualize_stress(latest_file)
