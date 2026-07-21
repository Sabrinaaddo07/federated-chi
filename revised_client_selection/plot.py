import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


TARGET = 0.08
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
        pattern = os.path.join(csv_dir, f"results_revised_clientsel_{m}_seed*.csv")
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"Missing seed files for M={m}: {pattern}")
            sys.exit(1)

        all_accs = []
        all_overhead_mb = []
        all_elapsed = []
        for path in paths:
            rows = load_csv(path)
            all_accs.append([float(r["global_accuracy"]) for r in rows])
            all_overhead_mb.append([float(r["cumulative_overhead_bytes"]) / 1e6 for r in rows])
            all_elapsed.append([float(r["elapsed_time_seconds"]) for r in rows])

        accs_arr = np.array(all_accs)
        overhead_arr = np.array(all_overhead_mb)
        elapsed_arr = np.array(all_elapsed)

        data[m] = {
            "rounds": [int(r["round"]) for r in load_csv(paths[0])],
            "accs_mean": accs_arr.mean(axis=0),
            "accs_std": accs_arr.std(axis=0, ddof=1),
            "overhead_mean": overhead_arr.mean(axis=0),
            "elapsed_mean": elapsed_arr.mean(axis=0),
        }

    crossings = {}
    for m in M_VALUES:
        d = data[m]
        idx = next((i for i, a in enumerate(d["accs_mean"]) if a >= TARGET), None)
        if idx is not None:
            crossings[m] = {
                "round": d["rounds"][idx],
                "overhead_mb": d["overhead_mean"][idx],
                "elapsed": d["elapsed_mean"][idx],
            }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    for i, m in enumerate(M_VALUES):
        d = data[m]
        c = COLORS[i]
        mk = MARKERS[i]
        label = f"M={m}"

        r = d["rounds"]
        m_acc = d["accs_mean"]
        s_acc = d["accs_std"]
        m_ovh = d["overhead_mean"]
        m_elp = d["elapsed_mean"]

        ax1.plot(r, m_acc, color=c, marker=mk, linewidth=1.5, markersize=3, label=label)
        ax1.fill_between(r, m_acc - s_acc, m_acc + s_acc, color=c, alpha=0.15)

        ax2.plot(m_ovh, m_acc, color=c, marker=mk, linewidth=1.5, markersize=3, label=label)
        ax2.fill_betweenx(m_acc, m_ovh - s_acc, m_ovh + s_acc, color=c, alpha=0.15)

        ax3.plot(m_elp, m_acc, color=c, marker=mk, linewidth=1.5, markersize=3, label=label)
        ax3.fill_betweenx(m_acc, m_elp - s_acc, m_elp + s_acc, color=c, alpha=0.15)

        if m in crossings:
            cpt = crossings[m]
            ax1.plot(cpt["round"], TARGET, color=COLORS[i], marker="D", markersize=8, zorder=5)
            ax2.plot(cpt["overhead_mb"], TARGET, color=COLORS[i], marker="D", markersize=8, zorder=5)
            ax3.plot(cpt["elapsed"], TARGET, color=COLORS[i], marker="D", markersize=8, zorder=5)

    for ax in [ax1, ax2, ax3]:
        ax.axhline(y=TARGET, color="gray", linestyle="--", linewidth=1, label=f"Target {TARGET*100:.0f}%")

    ax1.set_xlabel("Communication Round", fontsize=12)
    ax1.set_ylabel("Global Test Accuracy", fontsize=12)
    ax1.set_title("Rounds vs Accuracy", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.20)

    ax2.set_xlabel("Cumulative Overhead (MB)", fontsize=12)
    ax2.set_ylabel("Global Test Accuracy", fontsize=12)
    ax2.set_title("Overhead vs Accuracy", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.20)

    ax3.set_xlabel("Elapsed Wall-Clock Time (s)", fontsize=12)
    ax3.set_ylabel("Global Test Accuracy", fontsize=12)
    ax3.set_title("Time vs Accuracy", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=8, loc="lower right")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 0.20)

    plt.tight_layout()
    out = os.path.join(csv_dir, "revised_client_selection_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()

    print(f"\n{'='*70}")
    print(f"  Cost to reach target accuracy = {TARGET*100:.0f}%")
    print(f"{'='*70}")
    h = f"{'M':<6}{'Reached?':<10}{'Rounds':<10}{'Overhead (MB)':<16}{'Time (s)':<10}"
    print(h)
    print("-" * len(h))
    for m in M_VALUES:
        if m in crossings:
            c = crossings[m]
            print(f"{m:<6}{'Yes':<10}{c['round']:<10}{c['overhead_mb']:<16.2f}{c['elapsed']:<10.1f}")
        else:
            d = data[m]
            peak = max(d["accs_mean"])
            print(f"{m:<6}{'No (peak '+str(round(peak*100))+'%)':<10}{'—':<10}{'—':<16}{'—':<10}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
