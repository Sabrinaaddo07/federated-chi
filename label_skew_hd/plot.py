import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ['mnist', 'cifar10', 'cifar100']
HD_VALUES = [0.0, 0.25, 0.5, 0.75, 0.9]
DATASET_COLORS = {"mnist": "#2e86ab", "cifar10": "#e74c3c", "cifar100": "#27ae60"}
DATASET_MARKERS = {"mnist": "o", "cifar10": "s", "cifar100": "^"}

# Paper's centralized baseline accuracies (approximate from Table II)
CL_BASELINES = {
    'cifar10': 0.7050,
}


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_all_summaries(csv_dir):
    data = {}
    for ds in DATASETS:
        data[ds] = {}
        for hd in HD_VALUES:
            pattern = os.path.join(csv_dir, f"results_{ds}_hd{hd}_seed*.csv")
            paths = sorted(glob.glob(pattern))
            if not paths:
                print(f"Missing: {pattern}")
                sys.exit(1)

            best_accs = []
            rounds_vals = []
            time_vals = []
            overhead_vals = []
            for p in paths:
                rows = load_csv(p)
                row = rows[0]
                best_accs.append(float(row["max_accuracy"]))
                rtt = row.get("rounds_to_target", "NA")
                rounds_vals.append(int(rtt) if rtt != "NA" else None)
                overhead_vals.append(float(row["cumulative_overhead_bytes"]) / 1e6)
                time_vals.append(float(row["elapsed_time_seconds"]))

            best_arr = np.array(best_accs)
            rounds_arr = np.array([r for r in rounds_vals if r is not None])
            overhead_arr = np.array(overhead_vals)
            time_arr = np.array(time_vals)

            data[ds][hd] = {
                "acc_mean": best_arr.mean(),
                "acc_std": best_arr.std(ddof=1) if len(best_arr) > 1 else 0,
                "rounds_mean": rounds_arr.mean() if len(rounds_arr) > 0 else None,
                "rounds_std": rounds_arr.std(ddof=1) if len(rounds_arr) > 1 else 0,
                "overhead_mean": overhead_arr.mean(),
                "overhead_std": overhead_arr.std(ddof=1) if len(overhead_arr) > 1 else 0,
                "time_mean": time_arr.mean(),
                "time_std": time_arr.std(ddof=1) if len(time_arr) > 1 else 0,
            }
    return data


def load_centralized_baselines(csv_dir):
    baselines = {}
    for ds in DATASETS:
        pattern = os.path.join(csv_dir, f"results_{ds}_hd*_seed*.csv")
        paths = sorted(glob.glob(pattern))
        for p in paths:
            rows = load_csv(p)
            if len(rows) > 1:
                cl_row = rows[1]
                if "centralized_baseline_accuracy" in cl_row:
                    baselines[ds] = float(cl_row["centralized_baseline_accuracy"])
                    break
    return baselines


def main():
    csv_dir = os.path.dirname(os.path.abspath(__file__))
    data = load_all_summaries(csv_dir)
    cl_baselines = load_centralized_baselines(csv_dir)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    x = np.arange(len(HD_VALUES))
    width = 0.22

    # ─── Panel 1: Accuracy vs HD (Figure 3 replication) ───
    for i, ds in enumerate(DATASETS):
        offset = (i - 1) * width
        means = [data[ds][hd]["acc_mean"] for hd in HD_VALUES]
        stds = [data[ds][hd]["acc_std"] for hd in HD_VALUES]
        color = DATASET_COLORS[ds]
        marker = DATASET_MARKERS[ds]
        ax1.errorbar(x + offset, means, yerr=stds, color=color, marker=marker,
                     linestyle='-', linewidth=1.5, markersize=6, capsize=3,
                     label=ds.upper())

        # Centralized baseline line
        if ds in cl_baselines:
            cl_acc = cl_baselines[ds]
            ax1.axhline(y=cl_acc, color=color, linestyle=':', linewidth=1.5, alpha=0.6)
            ax1.text(4.2, cl_acc + 0.01, f"CL {ds.upper()}", color=color, fontsize=8,
                     fontweight='bold', alpha=0.6)

    ax1.axvspan(2.5 - 0.5, 4.5 - 0.5, color='red', alpha=0.06, label='Double threshold')
    ax1.axvline(x=2, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.axvline(x=3, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.text(1.9, ax1.get_ylim()[1] * 0.95, 'HD=0.5', color='red', fontsize=8, rotation=90, va='top')
    ax1.text(2.9, ax1.get_ylim()[1] * 0.95, 'HD=0.75', color='red', fontsize=8, rotation=90, va='top')

    ax1.set_xlabel("Hellinger Distance (HD)", fontsize=12)
    ax1.set_ylabel("Max Accuracy", fontsize=12)
    ax1.set_title("Accuracy vs HD  (Figure 3 replication)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(h) for h in HD_VALUES])
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(True, alpha=0.3)

    # ─── Panel 2: Cumulative Overhead vs Accuracy ───
    for i, ds in enumerate(DATASETS):
        color = DATASET_COLORS[ds]
        marker = DATASET_MARKERS[ds]
        acc_means = [data[ds][hd]["acc_mean"] for hd in HD_VALUES]
        acc_stds = [data[ds][hd]["acc_std"] for hd in HD_VALUES]
        ovh_means = [data[ds][hd]["overhead_mean"] for hd in HD_VALUES]
        ovh_stds = [data[ds][hd]["overhead_std"] for hd in HD_VALUES]
        ax2.errorbar(acc_means, ovh_means,
                     xerr=acc_stds, yerr=ovh_stds,
                     color=color, marker=marker, linestyle='-',
                     linewidth=1.5, markersize=6, capsize=3, label=ds.upper())
        for j, hd in enumerate(HD_VALUES):
            ax2.annotate(f"HD={hd}", (acc_means[j], ovh_means[j]),
                         textcoords="offset points", xytext=(5, 5), fontsize=6)

    ax2.set_xlabel("Accuracy", fontsize=12)
    ax2.set_ylabel("Cumulative Overhead (MB)", fontsize=12)
    ax2.set_title("Communication Overhead vs Accuracy", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ─── Panel 3: Cumulative Time vs Accuracy ───
    for i, ds in enumerate(DATASETS):
        color = DATASET_COLORS[ds]
        marker = DATASET_MARKERS[ds]
        acc_means = [data[ds][hd]["acc_mean"] for hd in HD_VALUES]
        acc_stds = [data[ds][hd]["acc_std"] for hd in HD_VALUES]
        time_means = [data[ds][hd]["time_mean"] for hd in HD_VALUES]
        time_stds = [data[ds][hd]["time_std"] for hd in HD_VALUES]
        ax3.errorbar(acc_means, time_means,
                     xerr=acc_stds, yerr=time_stds,
                     color=color, marker=marker, linestyle='-',
                     linewidth=1.5, markersize=6, capsize=3, label=ds.upper())
        for j, hd in enumerate(HD_VALUES):
            ax3.annotate(f"HD={hd}", (acc_means[j], time_means[j]),
                         textcoords="offset points", xytext=(5, 5), fontsize=6)

    ax3.set_xlabel("Accuracy", fontsize=12)
    ax3.set_ylabel("Cumulative Time (s)", fontsize=12)
    ax3.set_title("Wall-Clock Time vs Accuracy", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(csv_dir, "hd_experiment_figure3.png")
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()

    print(f"\n{'=' * 90}")
    print(f"  Results Summary")
    print(f"{'=' * 90}")
    for ds in DATASETS:
        print(f"\n--- {ds.upper()} ---")
        hdr = f"{'HD':<8}{'Accuracy':<20}{'Rounds-to-90%':<20}{'Overhead (MB)':<18}{'Time (s)':<12}"
        print(hdr)
        print("-" * len(hdr))
        for hd in HD_VALUES:
            d = data[ds][hd]
            a = f"{d['acc_mean']*100:.1f}% ± {d['acc_std']*100:.1f}%"
            r = f"{d['rounds_mean']:.1f} ± {d['rounds_std']:.1f}" if d['rounds_mean'] else "NA"
            o = f"{d['overhead_mean']:.1f} ± {d['overhead_std']:.1f}"
            t = f"{d['time_mean']:.0f} ± {d['time_std']:.0f}"
            print(f"{hd:<8}{a:<20}{r:<20}{o:<18}{t:<12}")

    if cl_baselines:
        print(f"\nCentralized baselines (CL):")
        for ds, acc in cl_baselines.items():
            print(f"  {ds}: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
