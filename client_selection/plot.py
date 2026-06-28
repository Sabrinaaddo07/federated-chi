import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


RATES = [100, 80, 60, 40, 20]
COLORS = ["#2e86ab", "#e67e22", "#27ae60", "#e74c3c", "#8e44ad"]


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    csv_dir = os.path.dirname(os.path.abspath(__file__))
    data = {}
    for rate in RATES:
        path = os.path.join(csv_dir, f"results_clientsel_{rate}.csv")
        if not os.path.exists(path):
            print(f"Missing: results_clientsel_{rate}.csv")
            sys.exit(1)
        rows = load_csv(path)
        rounds = [int(r["round"]) for r in rows]
        accs = [float(r["global_accuracy"]) for r in rows]
        overhead_mb = [float(r["cumulative_overhead_bytes"]) / 1e6 for r in rows]
        data[rate] = {"rounds": rounds, "accs": accs, "overhead_mb": overhead_mb}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for i, rate in enumerate(RATES):
        d = data[rate]
        c = COLORS[i]
        label = f"{rate}% ({int(10 * rate / 100)} clients/round)"
        ax1.plot(d["rounds"], d["accs"], color=c, marker="o",
                 linewidth=1.5, markersize=3, label=label)
        ax2.plot(d["overhead_mb"], d["accs"], color=c, marker="o",
                 linewidth=1.5, markersize=3, label=label)

    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("Global Test Accuracy", fontsize=12)
    ax1.set_title("Accuracy vs Rounds", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.set_xlabel("Cumulative Communication Overhead (MB)", fontsize=12)
    ax2.set_ylabel("Global Test Accuracy", fontsize=12)
    ax2.set_title("Accuracy vs Overhead", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    out = os.path.join(csv_dir, "client_selection_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()

    print()
    print("=" * 70)
    print(f"{'Rate':<8}{'Final Acc':<12}{'Peak Acc':<12}{'Peak Round':<12}"
          f"{'Total Overhead (MB)':<20}{'Time (s)':<10}")
    print("=" * 70)
    for rate in RATES:
        d = data[rate]
        final = d["accs"][-1]
        peak = max(d["accs"])
        peak_r = d["rounds"][d["accs"].index(peak)]
        total_mb = d["overhead_mb"][-1]
        print(f"{rate:<8}{final:<12.4f}{peak:<12.4f}{peak_r:<12}"
              f"{total_mb:<20.2f}{'—':<10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
