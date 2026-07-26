import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ['mnist', 'cifar10', 'cifar100']
HD_VALUES = [0.0, 0.25, 0.5, 0.75, 0.9]
COLORS = ["#8e44ad", "#e74c3c", "#e67e22", "#f1c40f", "#27ae60"]
MARKERS = ["s", "^", "D", "v", "o"]
DATASET_COLORS = {"mnist": "#2e86ab", "cifar10": "#e74c3c", "cifar100": "#27ae60"}
DATASET_MARKERS = {"mnist": "o", "cifar10": "s", "cifar100": "^"}


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    csv_dir = os.path.dirname(os.path.abspath(__file__))

    all_data = {}
    for ds in DATASETS:
        all_data[ds] = {}
        for hd in HD_VALUES:
            pattern = os.path.join(csv_dir, f"results_{ds}_hd{hd}_seed*.csv")
            paths = sorted(glob.glob(pattern))
            if not paths:
                print(f"Missing: {pattern}")
                sys.exit(1)

            rounds_vals = []
            best_accs = []
            for p in paths:
                rows = load_csv(p)
                row = rows[0]
                best_accs.append(float(row["max_accuracy"]))
                rtt = row.get("rounds_to_target", "NA")
                rounds_vals.append(int(rtt) if rtt != "NA" else None)

            best_arr = np.array(best_accs)
            rounds_arr = np.array([r for r in rounds_vals if r is not None])

            all_data[ds][hd] = {
                "acc_mean": best_arr.mean(),
                "acc_std": best_arr.std(ddof=1) if len(best_arr) > 1 else 0,
                "rounds_mean": rounds_arr.mean() if len(rounds_arr) > 0 else None,
                "rounds_std": rounds_arr.std(ddof=1) if len(rounds_arr) > 1 else 0,
            }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    x = np.arange(len(HD_VALUES))
    width = 0.22

    for i, ds in enumerate(DATASETS):
        offset = (i - 1) * width
        means = [all_data[ds][hd]["acc_mean"] for hd in HD_VALUES]
        stds = [all_data[ds][hd]["acc_std"] for hd in HD_VALUES]
        color = DATASET_COLORS[ds]
        marker = DATASET_MARKERS[ds]
        ax1.errorbar(x + offset, means, yerr=stds, color=color, marker=marker,
                     linestyle='-', linewidth=1.5, markersize=6, capsize=3,
                     label=ds.upper())

    ax1.set_xlabel("Hellinger Distance (HD)", fontsize=12)
    ax1.set_ylabel("Max Accuracy", fontsize=12)
    ax1.set_title("Accuracy vs HD (Figure 3 replication)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(h) for h in HD_VALUES])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    for i, ds in enumerate(DATASETS):
        offset = (i - 1) * width
        means = [all_data[ds][hd]["rounds_mean"] for hd in HD_VALUES]
        stds = [all_data[ds][hd]["rounds_std"] for hd in HD_VALUES]
        color = DATASET_COLORS[ds]
        marker = DATASET_MARKERS[ds]
        ax2.errorbar(x + offset, means, yerr=stds, color=color, marker=marker,
                     linestyle='-', linewidth=1.5, markersize=6, capsize=3,
                     label=ds.upper())

    ax2.set_xlabel("Hellinger Distance (HD)", fontsize=12)
    ax2.set_ylabel("Rounds to 90% of Max Accuracy", fontsize=12)
    ax2.set_title("Rounds-to-Target vs HD", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(h) for h in HD_VALUES])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    for i, ds in enumerate(DATASETS):
        pattern = os.path.join(csv_dir, f"results_{ds}_hd*_seed*.csv")
        paths = sorted(glob.glob(pattern))
        hd_overhead = {}
        hd_time = {}
        for p in paths:
            rows = load_csv(p)
            row = rows[0]
            hd_val = float(row["hd"])
            overhead_mb = float(row["cumulative_overhead_bytes"]) / 1e6
            elapsed = float(row["elapsed_time_seconds"])
            if hd_val not in hd_overhead:
                hd_overhead[hd_val] = []
                hd_time[hd_val] = []
            hd_overhead[hd_val].append(overhead_mb)
            hd_time[hd_val].append(elapsed)

        ovh_means = [np.mean(hd_overhead[hd]) for hd in HD_VALUES]
        ovh_stds = [np.std(hd_overhead[hd], ddof=1) if len(hd_overhead[hd]) > 1 else 0 for hd in HD_VALUES]
        time_means = [np.mean(hd_time[hd]) for hd in HD_VALUES]
        time_stds = [np.std(hd_time[hd], ddof=1) if len(hd_time[hd]) > 1 else 0 for hd in HD_VALUES]

        color = DATASET_COLORS[ds]
        marker = DATASET_MARKERS[ds]
        offset = (i - 1) * width
        ax3.errorbar(x + offset, ovh_means, yerr=ovh_stds, color=color, marker=marker,
                     linestyle='-', linewidth=1.5, markersize=6, capsize=3,
                     label=ds.upper())

        # Overhead on ax3
        ax3.set_xlabel("Hellinger Distance (HD)", fontsize=12)
        ax3.set_ylabel("Cumulative Overhead (MB)", fontsize=12)
        ax3.set_title("Overhead vs HD", fontsize=13, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels([str(h) for h in HD_VALUES])
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(csv_dir, "hd_experiment_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()

    print(f"\n{'='*90}")
    print(f"  Results Summary")
    print(f"{'='*90}")
    for ds in DATASETS:
        print(f"\n--- {ds.upper()} ---")
        print(f"{'HD':<8}{'Accuracy':<16}{'Rounds-to-90%':<16}")
        print("-" * 40)
        for hd in HD_VALUES:
            d = all_data[ds][hd]
            r_str = f"{d['rounds_mean']:.1f} ± {d['rounds_std']:.1f}" if d['rounds_mean'] else "NA"
            print(f"{hd:<8}{d['acc_mean']*100:.1f}% ± {d['acc_std']*100:.1f}%  {r_str:<16}")


if __name__ == "__main__":
    main()
