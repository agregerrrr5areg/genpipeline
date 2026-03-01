import os
import time
import json
import glob
from pathlib import Path

def get_progress():
    files = glob.glob("optimization_results/fem/*.json")
    count = len(files)
    
    best_stress = float('inf')
    latest_params = {}
    
    # Just check count mostly, reading 1000 jsons is slow
    if files:
        try:
            with open(files[-1], 'r') as j:
                data = json.load(j)
                latest_params = data.get('parameters', {})
        except:
            pass
            
    return count, best_stress, latest_params

def render_dashboard():
    total_target = 1000
    start_time = time.time()
    
    while True:
        count, best, params = get_progress()
        pct = min(100.0, (count / total_target) * 100)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        
        elapsed = time.time() - start_time
        
        # Clear and Print
        os.system('clear')
        print("="*60)
        print("🚀 GENPIPELINE LIVE DISCOVERY VIEWER (RTX 5080)")
        print("="*60)
        print(f"\nProgress: [{bar}] {pct:.1f}%")
        print(f"Iterations: {count} / {total_target}")
        print(f"Status:     {'STAGE 1: GPU SEEDING' if count < 100 else 'STAGE 2: BO REFINEMENT'}")
        print("-" * 60)
        print(f"Latest Design: H={params.get('h_mm', 0):.2f}mm, R={params.get('r_mm', 0):.2f}mm")
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
