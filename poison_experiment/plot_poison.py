"""
plot_poison.py - Read results_poison_*.csv, generate figures + metrics table.

Outputs:
  poison_experiment.png    — 3-panel accuracy curves with poison-round bands
  poison_detectability.png — malicious vs honest local accuracy
  poison_per_class.png     — per-class accuracy bar chart (final round)

Usage:  python3 plot_poison.py
"""

import csv
import glob
import json
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
    try:
        return float(row[key]) if row[key] else None
    except (ValueError, KeyError):
        return None


def main():
    files = sorted(glob.glob("results_poison_*.csv"))
    if not files:
        print("No results_poison_*.csv files found.")
        sys.exit(1)

    scheme_labels = {
        "iid": "Exp 1: IID (balanced)",
        "single": "Exp 2: Non-IID (malicious has only 8s)",
        "multi": "Exp 3: Non-IID (malicious has 1,2,3)",
    }
    colors = {"iid": "#2e86ab", "single": "#d9534f", "multi": "#f6a623"}

    all_data = {}

    for f in files:
        rows = load_csv(f)
        scheme = f.replace("results_poison_", "").replace(".csv", "")

        rounds = [int(r["round"]) for r in rows]
        accs = [get_float(r, "global_accuracy") for r in rows]
        poison_rounds = [int(r["round"]) for r in rows if int(r["is_poison_round"])]
        mal_accs = [get_float(r, "malicious_local_acc") for r in rows]
        hon_accs = [get_float(r, "honest_avg_local_acc") for r in rows]
        mal_norms = [get_float(r, "malicious_update_norm") for r in rows]
        hon_norms = [get_float(r, "honest_avg_update_norm") for r in rows]

        last_row = rows[-1]
        per_class = json.loads(last_row["per_class_accuracy"]) if last_row.get("per_class_accuracy") else {}

        all_data[scheme] = {
            "rounds": rounds,
            "accs": accs,
            "poison_rounds": poison_rounds,
            "mal_accs": mal_accs,
            "hon_accs": hon_accs,
            "mal_norms": mal_norms,
            "hon_norms": hon_norms,
            "per_class": per_class,
        }

    # ── Figure 1: Global accuracy curves with poison bands ──
    fig1, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for idx, (scheme, data) in enumerate(all_data.items()):
        ax = axes[idx]
        color = colors.get(scheme, "#888")

        ax.plot(data["rounds"], data["accs"], color=color, linewidth=2,
                marker=".", markersize=3)

        for pr in data["poison_rounds"]:
            ax.axvspan(pr - 0.5, pr + 0.5, color="red", alpha=0.08)

        ax.set_title(scheme_labels.get(scheme, scheme), fontsize=12, fontweight="bold")
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel("Global Test Accuracy", fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.label_outer()

    fig1.suptitle("Federated Learning — Random Label Poisoning (malicious client poisons ~20/40 rounds)",
                  fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("poison_experiment.png", dpi=150, bbox_inches="tight")
    print("Saved to poison_experiment.png")

    # ── Figure 2: Detectability — malicious vs honest local accuracy ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for idx, (scheme, data) in enumerate(all_data.items()):
        ax = axes2[idx]
        color = colors.get(scheme, "#888")

        ax.plot(data["rounds"], data["mal_accs"], color="red", linewidth=1.5,
                marker=".", markersize=3, label="Malicious client")
        ax.plot(data["rounds"], data["hon_accs"], color="green", linewidth=1.5,
                marker=".", markersize=3, label="Avg honest clients")

        for pr in data["poison_rounds"]:
            ax.axvspan(pr - 0.5, pr + 0.5, color="red", alpha=0.06)

        ax.set_title(scheme_labels.get(scheme, scheme), fontsize=12, fontweight="bold")
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel("Local Test Accuracy", fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.label_outer()

    fig2.suptitle("Detectability — Malicious vs Honest Client Local Accuracy",
                  fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("poison_detectability.png", dpi=150, bbox_inches="tight")
    print("Saved to poison_detectability.png")

    # ── Figure 3: Per-class accuracy bar chart (final round) ──
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for idx, (scheme, data) in enumerate(all_data.items()):
        ax = axes3[idx]
        per_class = data["per_class"]
        digits = sorted(per_class.keys(), key=int)

        # Color bars: red for classes owned by malicious, blue for others
        malicious_classes = set()
        if scheme == "single":
            malicious_classes = {"8"}
        elif scheme == "multi":
            malicious_classes = {"1", "2", "3"}

        bar_colors = [
            "#d9534f" if d in malicious_classes else "#2e86ab"
            for d in digits
        ]

        bars = ax.bar(digits, [per_class[d] for d in digits], color=bar_colors,
                      edgecolor="black", linewidth=0.5)

        # Highlight the bars for malicious-owned classes
        for bar, d in zip(bars, digits):
            if d in malicious_classes:
                bar.set_edgecolor("black")
                bar.set_linewidth(2)

        ax.set_title(scheme_labels.get(scheme, scheme), fontsize=12, fontweight="bold")
        ax.set_xlabel("Digit Class", fontsize=11)
        ax.set_ylabel("Test Accuracy", fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")
        ax.label_outer()

    fig3.suptitle("Per-Class Accuracy on Server Test Set (Final Round)",
                  fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("poison_per_class.png", dpi=150, bbox_inches="tight")
    print("Saved to poison_per_class.png")

    # ── Metrics table ──
    print()
    print("─" * 110)
    header = f"{'Metric':<40}"
    for scheme in all_data:
        header += f"{scheme_labels.get(scheme, scheme):<33}"
    print(header)
    print("─" * 110)

    # Row 1: Final accuracy
    row = f"{'Final accuracy (final round)':<40}"
    for scheme, data in all_data.items():
        row += f"{data['accs'][-1]:<33.4f}"
    print(row)

    # Row 2: Rounds to 80%
    row = f"{'Rounds to >= 80%':<40}"
    for scheme, data in all_data.items():
        r2conv = next(
            (r for r, a in zip(data["rounds"], data["accs"]) if a is not None and a >= 0.80),
            "-",
        )
        row += f"{str(r2conv):<33}"
    print(row)

    # Row 3: Accuracy variance (last 10 rounds)
    row = f"{'Accuracy variance (last 10 rounds)':<40}"
    for scheme, data in all_data.items():
        last10 = [a for a in data["accs"][-10:] if a is not None]
        var = np.std(last10) if last10 else 0.0
        row += f"{var:<33.4f}"
    print(row)

    # Row 4: Malicious vs honest local acc gap (avg over poison rounds)
    row = f"{'Local acc gap (honest − malicious on poison)':<40}"
    for scheme, data in all_data.items():
        gaps = []
        for i, r in enumerate(data["rounds"]):
            if (r in data["poison_rounds"] and data["mal_accs"][i] is not None
                    and data["hon_accs"][i] is not None):
                gaps.append(data["hon_accs"][i] - data["mal_accs"][i])
        avg_gap = np.mean(gaps) if gaps else 0.0
        row += f"{avg_gap:<33.4f}"
    print(row)

    # Row 5: Update norm ratio on poison rounds
    row = f"{'Update norm ratio (malicious / honest on poison)':<40}"
    for scheme, data in all_data.items():
        ratios = []
        for i, r in enumerate(data["rounds"]):
            if (r in data["poison_rounds"] and data["mal_norms"][i] is not None
                    and data["hon_norms"][i] is not None
                    and data["hon_norms"][i] > 0):
                ratios.append(data["mal_norms"][i] / data["hon_norms"][i])
        avg_ratio = np.mean(ratios) if ratios else 0.0
        row += f"{avg_ratio:<33.2f}"
    print(row)

    # Row 6: Worst per-class accuracy
    row = f"{'Worst per-class accuracy (final round)':<40}"
    for scheme, data in all_data.items():
        worst = min(data["per_class"].values()) if data["per_class"] else 0.0
        row += f"{worst:<33.4f}"
    print(row)

    print("─" * 110)

    # Interpretation notes
    print()
    print("How to interpret:")
    print("  - Local acc gap: if malicious local acc drops to ~10% on poison rounds")
    print("    while honest clients stay high → attack is trivially detectable.")
    print("  - Update norm ratio: >>1 means robust aggregation (Krum, Median, Clip)")
    print("    could filter the malicious update based on magnitude alone.")
    print("  - Per-class accuracy: shows which digit classes are most impacted.")
    print("    Red bars = classes owned by the malicious client.")
    print("  - Compare with clean baseline (e.g., previous results_non_iid_iid.csv)")
    print("    to quantify the accuracy drop from poisoning.")


if __name__ == "__main__":
    main()
