"""
plot_pareto.py — plot the stress vs mass Pareto front from optimization results.

Usage:
    python plot_pareto.py --results optimization_results/production_run
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_pareto(results_dir: str, output: str = None):
    results_dir = Path(results_dir)
    hist_path = results_dir / "optimization_history.json"
    if not hist_path.exists():
        print(f"No optimization_history.json in {results_dir}")
        return

    with open(hist_path) as f:
        hist = json.load(f)

    pareto = hist.get("pareto_front", [])
    all_y  = hist.get("history_y", [])

    if not all_y:
        print("No history found.")
        return

    all_y = np.array(all_y)
    valid  = all_y[(all_y[:, 0] > 0) & (all_y[:, 0] < 1e5)]

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(valid):
        ax.scatter(valid[:, 1], valid[:, 0], s=18, alpha=0.4,
                   color="steelblue", label=f"All valid ({len(valid)})")

    if pareto:
        px = [p["mass"]   for p in pareto]
        py = [p["stress"] for p in pareto]
        order = np.argsort(px)
        ax.scatter(np.array(px)[order], np.array(py)[order],
                   s=60, color="crimson", zorder=5, label=f"Pareto front ({len(pareto)})")
        ax.plot(np.array(px)[order], np.array(py)[order],
                color="crimson", linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Mass (kg)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title("Stress vs Mass — Pareto Front")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = output or str(results_dir / "pareto_plot.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()

    # Print Pareto summary
    if pareto:
        print(f"\n{'Design':<20} {'Stress (MPa)':>14} {'Mass (kg)':>12}")
        print("-" * 48)
        for i, p in enumerate(sorted(pareto, key=lambda x: x["stress"])):
            print(f"  design_{i:<14} {p['stress']:>14.1f} {p['mass']:>12.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="optimization_results/production_run")
    parser.add_argument("--output",  default=None, help="Output PNG path")
    args = parser.parse_args()
    plot_pareto(args.results, args.output)
