import sys
import time
import json
import glob
import numpy as np
from pathlib import Path

LINES = 15  # number of lines in the dashboard block

def move_up(n):
    sys.stdout.write(f"\033[{n}A\033[J")
    sys.stdout.flush()

def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

def get_progress():
    files = glob.glob("optimization_results/fem/*.json")
    count = len(files)

    data_points = []
    
    for path in files:
        try:
            with open(path, 'r') as j:
                data = json.load(j)
            stress = data.get('stress_max', 1e6)
            mass = data.get('mass', 1.0)
            if stress > 0 and stress < 1e5:
                data_points.append({
                    "stress": stress,
                    "mass": mass,
                    "params": data.get('parameters', {})
                })
        except:
            pass

    if not data_points:
        return count, [], None

    costs = np.array([[p['stress'], p['mass']] for p in data_points])
    pareto_mask = is_pareto_efficient(costs)
    pareto_points = [data_points[i] for i in range(len(data_points)) if pareto_mask[i]]
    
    # Sort Pareto points by mass for display
    pareto_points.sort(key=lambda x: x['mass'])
    
    latest = data_points[-1] if data_points else None
    
    return count, pareto_points, latest

def render_dashboard():
    total_target = 1000
    start_time = time.time()
    
    # Use clear screen for first run
    print("\033[2J\033[H", end="")
    
    while True:
        count, pareto, latest = get_progress()
        pct = min(100.0, (count / total_target) * 100)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))

        elapsed = time.time() - start_time
        
        move_up(0) # Reset to top
        print("\033[H", end="")
        print("="*70)
        print(f"🚀 GENPIPELINE PARETO DISCOVERY (RTX 5080 BLACKWELL)")
        print("="*70)
        print(f"Progress:   [{bar}] {pct:.1f}% ({count}/{total_target})")
        print(f"Time Active: {int(elapsed/60)}m {int(elapsed%60)}s")
        print("-" * 70)
        
        if latest:
            print(f"Latest:     Stress: {latest['stress']:7.1f} MPa | Mass: {latest['mass']:6.3f} kg | H={latest['params'].get('h_mm',0):.1f} R={latest['params'].get('r_mm',0):.1f}")
        else:
            print("Latest:     Waiting for simulations...")
            
        print("-" * 70)
        print(f"PARETO FRONT ({len(pareto)} optimal designs found):")
        print(f"  {'Type':<15} | {'Stress (MPa)':<12} | {'Mass (kg)':<10} | {'Dimensions'}")
        
        if len(pareto) > 0:
            # Show top 3: Lightest, Strongest, and Median
            lightest = pareto[0]
            strongest = pareto[-1]
            median = pareto[len(pareto)//2]
            
            for label, p in [("FEATHERWEIGHT", lightest), ("BALANCED", median), ("REINFORCED", strongest)]:
                dims = f"H={p['params'].get('h_mm',0):.1f} R={p['params'].get('r_mm',0):.1f}"
                print(f"  {label:<15} | {p['stress']:12.1f} | {p['mass']:10.3f} | {dims}")
        else:
            print("  Collecting data points to map the frontier...")

        print("-" * 70)
        print("\n(Press Ctrl+C to exit - Optimization will continue in background)")
        
        if count >= total_target:
            print("\n✅ DISCOVERY COMPLETE! Run Step 5 to export Pareto STLs.")
            break
            
        time.sleep(5)

if __name__ == "__main__":
    try:
        render_dashboard()
    except KeyboardInterrupt:
        print("\nDashboard closed.")
