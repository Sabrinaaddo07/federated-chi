import csv
import os
import sys

import matplotlib.pyplot as plt


DISTS = ["iid", "noniid"]
DIST_LABELS = ["IID", "Non-IID"]
COLORS = ["#2e86ab", "#e74c3c"]
MARKERS = ["o", "s"]


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    csv_dir = os.path.dirname(os.path.abspath(__file__))
    data = {}
    for dist, label in zip(DISTS, DIST_LABELS):
        path = os.path.join(csv_dir, f"results_distribution_{dist}.csv")
        if not os.path.exists(path):
            print(f"Missing: results_distribution_{dist}.csv")
            sys.exit(1)
        rows = load_csv(path)
        rounds = [int(r["round"]) for r in rows]
        accs = [float(r["global_accuracy"]) for r in rows]
        overhead_mb = [float(r["cumulative_overhead_bytes"]) / 1e6 for r in rows]
        data[label] = {"rounds": rounds, "accs": accs, "overhead_mb": overhead_mb}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for i, label in enumerate(DIST_LABELS):
        d = data[label]
        c = COLORS[i]
        m = MARKERS[i]
        ax1.plot(d["rounds"], d["accs"], color=c, marker=m,
                 linewidth=1.5, markersize=4, label=label)
        ax2.plot(d["overhead_mb"], d["accs"], color=c, marker=m,
                 linewidth=1.5, markersize=4, label=label)

    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("Global Test Accuracy", fontsize=12)
    ax1.set_title("Accuracy vs Rounds", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.set_xlabel("Cumulative Communication Overhead (MB)", fontsize=12)
    ax2.set_ylabel("Global Test Accuracy", fontsize=12)
    ax2.set_title("Accuracy vs Overhead", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11, loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    out = os.path.join(csv_dir, "data_distribution_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()

    print()
    print("=" * 70)
    h = f"{'Distribution':<12}{'Final Acc':<12}{'Peak Acc':<12}{'Total Overhead (MB)':<22}{'Time (s)':<10}"
    print(h)
    print("=" * 70)
    for label in DIST_LABELS:
        d = data[label]
        final = d["accs"][-1]
        peak = max(d["accs"])
        total_mb = d["overhead_mb"][-1]
        print(f"{label:<12}{final:<12.4f}{peak:<12.4f}{total_mb:<22.2f}{'—':<10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
