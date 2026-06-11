"""
plot.py - Read all results_dropout_*.csv files, plot combined accuracy curves,
          and print a summary metrics table.

Usage:  python3 plot.py
Output: dropout_experiment.png + printed table
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
    files = sorted(glob.glob("results_dropout_*.csv"))
    if not files:
        print("No results_dropout_*.csv files found in current directory.")
        sys.exit(1)

    colors = {0: "#2e86ab", 2: "#f6a623", 4: "#d9534f", 6: "#5cb85c"}
    markers = {0: "o", 2: "s", 4: "^", 6: "D"}

    all_data = {}

    plt.figure(figsize=(12, 7))

    for f in files:
        rows = load_csv(f)
        # Extract dropout count from filename
        label = f.replace("results_dropout_", "").replace(".csv", "")
        try:
            dropout_n = int(label)
        except ValueError:
            dropout_n = label

        rounds = [get_int(r, "round") for r in rows]
        accs = [get_float(r, "global_accuracy") for r in rows]
        dropout_round = get_int(rows[0], "dropout_round") if rows else 0
        data_shares = [get_float(r, "dropped_data_share") for r in rows]

        all_data[dropout_n] = {
            "rounds": rounds,
            "accs": accs,
            "dropout_round": dropout_round,
            "data_shares": data_shares,
        }

        color = colors.get(dropout_n, "#888")
        marker = markers.get(dropout_n, ".")
        plt.plot(rounds, accs, color=color, marker=marker, linewidth=1.5,
                 markersize=4, label=f"Dropout {dropout_n}")

        # Vertical line at dropout round
        if dropout_round > 0:
            plt.axvline(x=dropout_round, color=color, linestyle="--",
                        alpha=0.3, linewidth=1)

    plt.xlabel("Round", fontsize=13)
    plt.ylabel("Global Test Accuracy", fontsize=13)
    plt.title("Federated Learning — Mid-Experiment Dropout at Round 20",
              fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("dropout_experiment.png", dpi=150)
    print("Saved to dropout_experiment.png")
    print()

    # ── Metrics table ──
    print("─" * 80)
    print(f"{'Metric':<30}", end="")
    for n in sorted(all_data):
        print(f"{f'Dropout {n}':<18}", end="")
    print()
    print("─" * 80)

    # Row 1: Final accuracy
    print(f"{'Final accuracy (round 40)':<30}", end="")
    for n in sorted(all_data):
        d = all_data[n]
        final = d["accs"][-1]
        print(f"{final:<18.4f}", end="")
    print()

    # Row 2: Rounds to reach 85%
    print(f"{'Rounds to >= 85%':<30}", end="")
    for n in sorted(all_data):
        d = all_data[n]
        r2conv = next((r for r, a in zip(d["rounds"], d["accs"]) if a >= 0.85), "-")
        print(f"{str(r2conv):<18}", end="")
    print()

    # Row 3: Accuracy variance (last 10 rounds)
    print(f"{'Accuracy variance (r31–40)':<30}", end="")
    for n in sorted(all_data):
        d = all_data[n]
        last10 = d["accs"][-10:]
        var = np.std(last10)
        print(f"{var:<18.4f}", end="")
    print()

    # Row 4: Gap vs baseline
    if 0 in all_data:
        baseline_final = all_data[0]["accs"][-1]
        print(f"{'Gap vs baseline (final acc)':<30}", end="")
        for n in sorted(all_data):
            d = all_data[n]
            gap = d["accs"][-1] - baseline_final
            sign = "+" if gap >= 0 else ""
            print(f"{sign}{gap:<17.4f}", end="")
        print()
    else:
        print(f"{'Gap vs baseline (final acc)':<30}{'N/A (no baseline)':<18}")

    # Row 5: Data lost at dropout
    print(f"{'Data lost at dropout (max)':<30}", end="")
    for n in sorted(all_data):
        d = all_data[n]
        max_share = max(d["data_shares"]) if d["data_shares"] else 0.0
        print(f"{max_share:<18.1%}", end="")
    print()
    print("─" * 80)


if __name__ == "__main__":
    main()
