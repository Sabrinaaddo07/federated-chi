import csv
import os
import sys

import matplotlib.pyplot as plt


M_VALUES = [1, 2, 4, 6, 8, 10]
COLORS = ["#8e44ad", "#e74c3c", "#e67e22", "#f1c40f", "#27ae60", "#2e86ab"]
MARKERS = ["x", "s", "^", "D", "v", "o"]


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    csv_dir = os.path.dirname(os.path.abspath(__file__))
    data = {}
    for m in M_VALUES:
        path = os.path.join(csv_dir, f"results_revised_clientsel_{m}.csv")
        if not os.path.exists(path):
            print(f"Missing: results_revised_clientsel_{m}.csv")
            sys.exit(1)
        rows = load_csv(path)
        rounds = [int(r["round"]) for r in rows]
        accs = [float(r["global_accuracy"]) for r in rows]
        overhead_mb = [float(r["cumulative_overhead_bytes"]) / 1e6 for r in rows]
        elapsed = [float(r["elapsed_time_seconds"]) for r in rows]
        data[m] = {"rounds": rounds, "accs": accs,
                   "overhead_mb": overhead_mb, "elapsed": elapsed}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    for i, m in enumerate(M_VALUES):
        d = data[m]
        c = COLORS[i]
        mk = MARKERS[i]
        label = f"M={m}"
        ax1.plot(d["rounds"], d["accs"], color=c, marker=mk,
                 linewidth=1.5, markersize=3, label=label)
        ax2.plot(d["overhead_mb"], d["accs"], color=c, marker=mk,
                 linewidth=1.5, markersize=3, label=label)
        ax3.plot(d["elapsed"], d["accs"], color=c, marker=mk,
                 linewidth=1.5, markersize=3, label=label)

    ax1.set_xlabel("Communication Round", fontsize=12)
    ax1.set_ylabel("Global Test Accuracy", fontsize=12)
    ax1.set_title("Rounds vs Accuracy", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.set_xlabel("Cumulative Overhead (MB)", fontsize=12)
    ax2.set_ylabel("Global Test Accuracy", fontsize=12)
    ax2.set_title("Overhead vs Accuracy", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    ax3.set_xlabel("Elapsed Wall-Clock Time (s)", fontsize=12)
    ax3.set_ylabel("Global Test Accuracy", fontsize=12)
    ax3.set_title("Time vs Accuracy", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9, loc="lower right")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    plt.tight_layout()
    out = os.path.join(csv_dir, "revised_client_selection_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()

    print()
    print("=" * 85)
    h = f"{'M':<4}{'Final Acc':<12}{'Peak Acc':<12}{'Peak Round':<12}" \
        f"{'Total Overhead (MB)':<20}{'Time (s)':<12}{'Rounds':<8}"
    print(h)
    print("=" * 85)
    for m in M_VALUES:
        d = data[m]
        final = d["accs"][-1]
        peak = max(d["accs"])
        peak_r = d["rounds"][d["accs"].index(peak)]
        total_mb = d["overhead_mb"][-1]
        total_t = d["elapsed"][-1]
        n_rounds = d["rounds"][-1]
        print(f"{m:<4}{final:<12.4f}{peak:<12.4f}{peak_r:<12}"
              f"{total_mb:<20.2f}{total_t:<12.2f}{n_rounds:<8}")
    print("=" * 85)


if __name__ == "__main__":
    main()
