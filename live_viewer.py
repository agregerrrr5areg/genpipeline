import os
import time
import json
import glob
from pathlib import Path

def get_progress():
    files = glob.glob("optimization_results/fem/*.json")
    count = len(files)

    best_stress = float('inf')
    best_params = {}
    latest_params = {}

    for path in files:
        try:
            with open(path, 'r') as j:
                data = json.load(j)
            stress = data.get('stress_max', 0.0)
            if stress > 0 and stress < best_stress:
                best_stress = stress
                best_params = data.get('parameters', {})
            if path == files[-1]:
                latest_params = data.get('parameters', {})
        except:
            pass

    return count, best_stress, best_params, latest_params

def render_dashboard():
    total_target = 1000
    start_time = time.time()
    
    while True:
        count, best_stress, best_params, latest_params = get_progress()
        pct = min(100.0, (count / total_target) * 100)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))

        elapsed = time.time() - start_time
        best_str = f"{best_stress:.2f} MPa" if best_stress < float('inf') else "N/A"

        # Clear and Print
        os.system('clear')
        print("="*60)
        print("GENPIPELINE LIVE DISCOVERY VIEWER (RTX 5080)")
        print("="*60)
        print(f"\nProgress: [{bar}] {pct:.1f}%")
        print(f"Iterations: {count} / {total_target}")
        print(f"Status:     {'STAGE 1: GPU SEEDING' if count < 100 else 'STAGE 2: BO REFINEMENT'}")
        print("-" * 60)
        print(f"Best Stress:   {best_str}  (H={best_params.get('h_mm', 0):.2f}mm, R={best_params.get('r_mm', 0):.2f}mm)")
        print(f"Latest Design: H={latest_params.get('h_mm', 0):.2f}mm, R={latest_params.get('r_mm', 0):.2f}mm")
        print(f"Time Active:   {int(elapsed/60)}m {int(elapsed%60)}s")
        print("-" * 60)
        print("\n(Press Ctrl+C to exit viewer - Background run will continue)")
        
        if count >= total_target:
            print("\n✅ DISCOVERY COMPLETE!")
            break
            
        time.sleep(5)

if __name__ == "__main__":
    try:
        render_dashboard()
    except KeyboardInterrupt:
        print("\nViewer closed.")
