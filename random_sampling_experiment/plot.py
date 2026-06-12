"""
plot.py - Read all results_sample_*.csv files and plot accuracy curves.

Usage:  python3 plot.py
Output: random_sampling.png + printed metrics table
"""

import csv
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    files = sorted(glob.glob("results_sample_*.csv"))
    if not files:
        print("No results_sample_*.csv files found in current directory.")
        sys.exit(1)

    colors = {2: "#f6a623", 4: "#d9534f", 6: "#5cb85c", 10: "#2e86ab"}
    markers = {2: "s", 4: "^", 6: "D", 10: "o"}

    all_data = {}
    plt.figure(figsize=(12, 7))

    for f in files:
        rows = load_csv(f)
        label = f.replace("results_sample_", "").replace(".csv", "")
        try:
            k = int(label)
        except ValueError:
            k = label

        rounds = [int(r["round"]) for r in rows]
        accs = [float(r["global_accuracy"]) for r in rows]

        all_data[k] = {"rounds": rounds, "accs": accs}

        color = colors.get(k, "#888")
        marker = markers.get(k, ".")
        plt.plot(rounds, accs, color=color, marker=marker, linewidth=1.5,
                 markersize=4, label=f"K={k} of 10")

    plt.xlabel("Round", fontsize=13)
    plt.ylabel("Global Test Accuracy", fontsize=13)
    plt.title("Random Per-Round Client Sampling — Accuracy vs Round",
              fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("random_sampling.png", dpi=150)
    print("Saved to random_sampling.png")
    print()

    # Metrics table
    print("─" * 80)
    print(f"{'Metric':<35}", end="")
    for k in sorted(all_data):
        print(f"{f'K={k} of 10':<18}", end="")
    print()
    print("─" * 80)

    print(f"{'Final accuracy (round 40)':<35}", end="")
    for k in sorted(all_data):
        print(f"{all_data[k]['accs'][-1]:<18.4f}", end="")
    print()

    print(f"{'Rounds to >= 85%':<35}", end="")
    for k in sorted(all_data):
        d = all_data[k]
        r2conv = next((r for r, a in zip(d["rounds"], d["accs"]) if a >= 0.85), "-")
        print(f"{str(r2conv):<18}", end="")
    print()

    print(f"{'Accuracy variance (r31–40)':<35}", end="")
    for k in sorted(all_data):
        last10 = all_data[k]["accs"][-10:]
        print(f"{np.std(last10):<18.4f}", end="")
    print()

    if 10 in all_data:
        baseline_final = all_data[10]["accs"][-1]
        print(f"{'Gap vs K=10 (final acc)':<35}", end="")
        for k in sorted(all_data):
            gap = all_data[k]["accs"][-1] - baseline_final
            sign = "+" if gap >= 0 else ""
            print(f"{sign}{gap:<17.4f}", end="")
        print()

    print("─" * 80)


if __name__ == "__main__":
    main()
