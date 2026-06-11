"""
plot_non_iid.py - Read all results_non_iid_*.csv files, plot combined accuracy
                  curves, and print a summary metrics table.

Usage:  python3 plot_non_iid.py
Output: non_iid_experiment.png + printed table
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
    files = sorted(glob.glob("results_non_iid_*.csv"))
    if not files:
        print("No results_non_iid_*.csv files found in current directory.")
        sys.exit(1)

    scheme_labels = {
        "iid": "IID (balanced, all 10 classes)",
        "a": "Non-IID A (1 class/client, classes 8-9 unseen)",
        "b": "Non-IID B (client 7 gets classes 7-9, 3x data)",
    }
    colors = {"iid": "#2e86ab", "a": "#d9534f", "b": "#f6a623"}
    markers = {"iid": "o", "a": "s", "b": "^"}

    all_data = {}

    plt.figure(figsize=(12, 7))

    for f in files:
        rows = load_csv(f)
        scheme = f.replace("results_non_iid_", "").replace(".csv", "")

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
    plt.title("Federated Learning — IID vs Non-IID Data Splits",
              fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("non_iid_experiment.png", dpi=150)
    print("Saved to non_iid_experiment.png")
    print()

    # ── Metrics table ──
    header_order = ["iid", "a", "b"]
    header_order = [s for s in header_order if s in all_data]

    print("─" * 90)
    print(f"{'Metric':<35}", end="")
    for s in header_order:
        print(f"{scheme_labels.get(s, s):<25}", end="")
    print()
    print("─" * 90)

    # Row 1: Final accuracy
    print(f"{'Final accuracy (round 40)':<35}", end="")
    for s in header_order:
        d = all_data[s]
        final = d["accs"][-1]
        print(f"{final:<25.4f}", end="")
    print()

    # Row 2: Rounds to reach 85%
    print(f"{'Rounds to >= 85%':<35}", end="")
    for s in header_order:
        d = all_data[s]
        r2conv = next((r for r, a in zip(d["rounds"], d["accs"]) if a >= 0.85), "-")
        print(f"{str(r2conv):<25}", end="")
    print()

    # Row 3: Accuracy variance (last 10 rounds)
    print(f"{'Accuracy variance (r31–40)':<35}", end="")
    for s in header_order:
        d = all_data[s]
        last10 = d["accs"][-10:]
        var = np.std(last10)
        print(f"{var:<25.4f}", end="")
    print()

    # Row 4: Gap vs IID baseline
    if "iid" in all_data:
        baseline_final = all_data["iid"]["accs"][-1]
        print(f"{'Gap vs IID baseline (final acc)':<35}", end="")
        for s in header_order:
            d = all_data[s]
            gap = d["accs"][-1] - baseline_final
            sign = "+" if gap >= 0 else ""
            print(f"{sign}{gap:<24.4f}", end="")
        print()
    print("─" * 90)

    # ── Additional analysis: per-class accuracy breakdown (scheme A only) ──
    print()
    print("Note: The server test set includes ALL 10 digit classes.")
    print("  - Scheme A: model NEVER saw classes 8-9 during training.")
    print("    Global accuracy above ~80% means it's generalizing to unseen digits.")
    print("  - Scheme B: Client 7 has 3x the data (classes 7-9).")
    print("    Model may overperform on 7-9 and underperform on 0-6.")


if __name__ == "__main__":
    main()
