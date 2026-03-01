import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

def generate_plot():
    files = glob.glob("optimization_results/fem/*.json")
    print(f"Analyzing {len(files)} discovery points...")
    
    data_points = []
    for path in files:
        try:
            with open(path, 'r') as j:
                data = json.load(j)
            stress = data.get('stress_max', 1e6)
            mass = data.get('mass', 1.0)
            if stress > 0 and stress < 1e5:
                data_points.append([stress, mass])
        except:
            pass

    if not data_points:
        print("No results found yet. Wait for FreeCAD to finish the first batch.")
        return

    pts = np.array(data_points)
    pareto_mask = is_pareto_efficient(pts)
    pareto_pts = pts[pareto_mask]
    pareto_pts = pareto_pts[pareto_pts[:, 1].argsort()] # sort by mass

    plt.figure(figsize=(10, 6))
    plt.scatter(pts[:, 1], pts[:, 0], c='blue', alpha=0.3, label='All Designs')
    plt.plot(pareto_pts[:, 1], pareto_pts[:, 0], 'r-o', linewidth=2, label='Pareto Front (Optimal)')
    
    plt.title(f"Plastic Pareto Front Discovery (N={len(pts)})")
    plt.xlabel("Mass (kg)")
    plt.ylabel("Max Stress (MPa)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_path = "pareto_discovery.png"
    plt.savefig(output_path)
    print(f"📊 Visualization saved to {output_path}")

if __name__ == "__main__":
    generate_plot()
