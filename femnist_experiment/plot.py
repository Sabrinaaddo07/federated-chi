"""
plot.py - Read FEMNIST experiment results and plot accuracy curves.

Usage:  python3 plot.py
Output: femnist_experiment.png + printed metrics table
"""

import csv
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_float(row, key):
    return float(row[key])


def get_int(row, key):
    return int(row[key])


def main():
    files = sorted(glob.glob("results_femnist_*.csv"))
    if not files:
        print("No results_femnist_*.csv files found in current directory.")
        sys.exit(1)

    scheme_labels = {
        "iid": "IID (shuffled, round-robin)",
        "non_iid": f"Non-IID (Dirichlet)",
    }
    colors = {"iid": "#2e86ab", "non_iid": "#d9534f"}
    markers = {"iid": "o", "non_iid": "s"}

    all_data = {}

    plt.figure(figsize=(12, 7))

    for f in files:
        rows = load_csv(f)
        scheme = f.replace("results_femnist_", "").replace(".csv", "")

        rounds = [get_int(r, "round") for r in rows]
        accs = [get_float(r, "global_accuracy") for r in rows]

        all_data[scheme] = {
            "rounds": rounds,
            "accs": accs,
        }

        color = colors.get(scheme, "#888")
        marker = markers.get(scheme, ".")
        label = scheme_labels.get(scheme, scheme)
        plt.plot(rounds, accs, color=color, marker=marker, linewidth=1.5,
                 markersize=4, label=label)

    plt.xlabel("Round", fontsize=13)
    plt.ylabel("Global Test Accuracy", fontsize=13)
    plt.title("Federated Learning on EMNIST — IID vs Non-IID (sklearn MLP)",
              fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("femnist_experiment.png", dpi=150)
    print("Saved to femnist_experiment.png")
    print()

    # Metrics table
    header_order = sorted(all_data.keys())

    print("=" * 70)
    print(f"{'Metric':<40}", end="")
    for s in header_order:
        print(f"{scheme_labels.get(s, s):<25}", end="")
    print()
    print("=" * 70)

    final_accs = {}
    for s in header_order:
        d = all_data[s]
        final_accs[s] = d["accs"][-1]

    print(f"{'Final accuracy (round ' + str(d['rounds'][-1]) + ')':<40}", end="")
    for s in header_order:
        print(f"{final_accs[s]:<25.4f}", end="")
    print()

    print(f"{'Rounds to >= 70%':<40}", end="")
    for s in header_order:
        d = all_data[s]
        r2conv = next((r for r, a in zip(d["rounds"], d["accs"]) if a >= 0.70), "-")
        print(f"{str(r2conv):<25}", end="")
    print()

    print(f"{'Accuracy variance (last 10 rounds)':<40}", end="")
    for s in header_order:
        d = all_data[s]
        last10 = d["accs"][-10:]
        var = np.std(last10)
        print(f"{var:<25.4f}", end="")
    print()

    if len(header_order) > 1:
        baseline = header_order[0]
        baseline_final = final_accs[baseline]
        print(f"{'Gap vs ' + baseline + ' baseline':<40}", end="")
        for s in header_order:
            gap = final_accs[s] - baseline_final
            sign = "+" if gap >= 0 else ""
            print(f"{sign}{gap:<24.4f}", end="")
        print()
    print("=" * 70)


if __name__ == "__main__":
    main()
