"""
plot_poison.py - Read results_poison_*.csv, generate figures + metrics table.

Outputs:
  poison_experiment.png       — accuracy curves: baseline vs uniform vs early_only vs late_only
  poison_detectability.png    — avg malicious vs avg honest local accuracy
  poison_per_class.png        — per-class accuracy bar chart (final round)

Usage:  python3 plot_poison.py
"""

import csv
import glob
import json
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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

    mode_config = {
        "baseline": {"color": "#2e86ab", "linestyle": "--", "marker": "o",
                     "label": "Baseline (no malicious)"},
        "uniform": {"color": "#d9534f", "linestyle": "-", "marker": "s",
                    "label": "Uniform (~20 rounds all 40)"},
        "early_only": {"color": "#5cb85c", "linestyle": "-", "marker": "^",
                       "label": "Early-only (~10 rounds 1-20)"},
        "late_only": {"color": "#f6a623", "linestyle": "-", "marker": "D",
                      "label": "Late-only (~10 rounds 21-40)"},
    }

    all_data = {}

    for f in files:
        rows = load_csv(f)
        mode = f.replace("results_poison_", "").replace(".csv", "")

        rounds = [int(r["round"]) for r in rows]
        accs = [get_float(r, "global_accuracy") for r in rows]
        poison_rounds = [int(r["round"]) for r in rows if int(r["is_poison_round"])]
        mal_accs = [get_float(r, "avg_malicious_acc") for r in rows]
        hon_accs = [get_float(r, "avg_honest_acc") for r in rows]
        mal_norms = [get_float(r, "avg_malicious_norm") for r in rows]
        hon_norms = [get_float(r, "avg_honest_norm") for r in rows]

        last_row = rows[-1]
        per_class = json.loads(last_row["per_class_accuracy"]) if last_row.get("per_class_accuracy") else {}

        all_data[mode] = {
            "rounds": rounds,
            "accs": accs,
            "poison_rounds": poison_rounds,
            "mal_accs": mal_accs,
            "hon_accs": hon_accs,
            "mal_norms": mal_norms,
            "hon_norms": hon_norms,
            "per_class": per_class,
        }

    # ── Figure 1: Accuracy curves overlaid ──
    fig1, ax = plt.subplots(figsize=(12, 6))

    # Shade poison regions for each non-baseline mode
    poison_rounds_shown = set()
    for mode in ["uniform", "early_only", "late_only"]:
        if mode in all_data:
            poison_rounds_shown.update(all_data[mode]["poison_rounds"])
    for pr in sorted(poison_rounds_shown):
        ax.axvspan(pr - 0.5, pr + 0.5, color="red", alpha=0.06)

    # Horizontal separator at round 20
    ax.axvline(x=20.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)

    for mode in ["baseline", "uniform", "early_only", "late_only"]:
        if mode not in all_data:
            continue
        data = all_data[mode]
        cfg = mode_config.get(mode, {})
        ax.plot(data["rounds"], data["accs"],
                color=cfg.get("color", "#888"),
                linestyle=cfg.get("linestyle", "-"),
                marker=cfg.get("marker", "."),
                linewidth=2, markersize=4,
                label=cfg.get("label", mode))

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Global Test Accuracy", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    ax.set_title("Federated Learning — Random Label Poisoning (3 malicious, IID, 40 rounds)",
                 fontsize=13, fontweight="bold")

    # Zoom inset for the 0.8-1.0 region
    inset = inset_axes(ax, width="50%", height="45%", loc="center left",
                       bbox_to_anchor=(0.35, 0.08, 1, 1),
                       bbox_transform=ax.transAxes)
    for mode in ["baseline", "uniform", "early_only", "late_only"]:
        if mode not in all_data:
            continue
        data = all_data[mode]
        cfg = mode_config.get(mode, {})
        inset.plot(data["rounds"], data["accs"],
                   color=cfg.get("color", "#888"),
                   linestyle=cfg.get("linestyle", "-"),
                   marker=cfg.get("marker", "."),
                   linewidth=2, markersize=4)
    inset.set_ylim(0.8, 1.0)
    inset.set_xlim(0, 40)
    inset.set_ylabel("Zoom (0.8-1.0)", fontsize=10, fontweight="bold")
    inset.set_xlabel("Round", fontsize=10)
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.3)
    for spine in inset.spines.values():
        spine.set_color("red")
        spine.set_linewidth(2)
    inset.text(0.5, 0.05, "ZOOM", transform=inset.transAxes,
               fontsize=11, fontweight="bold", color="red",
               ha="center", va="bottom",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor="red", linewidth=1.5))
    mark_inset(ax, inset, loc1=1, loc2=3, fc="none", ec="red",
               linestyle="--", linewidth=1.2)

    # 0.8 boundary line in inset
    inset.axhline(y=0.8, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    plt.savefig("poison_experiment.png", dpi=150)
    print("Saved to poison_experiment.png")

    # ── Figure 2: Detectability ──
    attack_modes = [m for m in ["uniform", "early_only", "late_only"] if m in all_data]
    n_attack = len(attack_modes)
    if n_attack > 0:
        fig2, axes2 = plt.subplots(1, n_attack, figsize=(7 * n_attack, 5), sharey=True)
        if n_attack == 1:
            axes2 = [axes2]

        for idx, mode in enumerate(attack_modes):
            data = all_data[mode]
            ax = axes2[idx]

            ax.plot(data["rounds"], data["mal_accs"], color="red", linewidth=1.5,
                    marker=".", markersize=3, label="Avg malicious clients")
            ax.plot(data["rounds"], data["hon_accs"], color="green", linewidth=1.5,
                    marker=".", markersize=3, label="Avg honest clients")

            for pr in data["poison_rounds"]:
                ax.axvspan(pr - 0.5, pr + 0.5, color="red", alpha=0.06)

            ax.set_title(f"{mode_config.get(mode, {}).get('label', mode)}",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel("Round", fontsize=11)
            ax.set_ylabel("Local Test Accuracy", fontsize=11)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        fig2.suptitle("Detectability — Avg Malicious vs Avg Honest Local Accuracy",
                      fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("poison_detectability.png", dpi=150)
        print("Saved to poison_detectability.png")

    # ── Figure 3: Per-class accuracy bar chart ──
    present_modes = [m for m in ["baseline", "uniform", "early_only", "late_only"] if m in all_data]
    n_modes = len(present_modes)
    if n_modes > 0:
        fig3, axes3 = plt.subplots(1, n_modes, figsize=(5.5 * n_modes, 5), sharey=True)
        if n_modes == 1:
            axes3 = [axes3]

        for idx, mode in enumerate(present_modes):
            data = all_data[mode]
            ax = axes3[idx]
            per_class = data["per_class"]
            digits = sorted(per_class.keys(), key=int)

            ax.bar(digits, [per_class[d] for d in digits],
                   color="#2e86ab", edgecolor="black", linewidth=0.5)

            cfg = mode_config.get(mode, {})
            ax.set_title(cfg.get("label", mode), fontsize=11, fontweight="bold")
            ax.set_xlabel("Digit Class", fontsize=11)
            ax.set_ylabel("Test Accuracy", fontsize=11)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis="y")

        fig3.suptitle("Per-Class Accuracy on Server Test Set (Final Round)",
                      fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("poison_per_class.png", dpi=150)
        print("Saved to poison_per_class.png")

    # ── Metrics table ──
    header_cols = [m for m in ["baseline", "uniform", "early_only", "late_only"] if m in all_data]
    if not header_cols:
        return

    print()
    print("─" * 120)
    header = f"{'Metric':<42}"
    for m in header_cols:
        header += f"{mode_config.get(m, {}).get('label', m):<26}"
    print(header)
    print("─" * 120)

    # Row 1: Final accuracy
    row = f"{'Final accuracy (final round)':<42}"
    for m in header_cols:
        row += f"{all_data[m]['accs'][-1]:<26.4f}"
    print(row)

    # Row 2: Rounds to 80%
    row = f"{'Rounds to >= 80%':<42}"
    for m in header_cols:
        r2conv = next(
            (r for r, a in zip(all_data[m]["rounds"], all_data[m]["accs"])
             if a is not None and a >= 0.80),
            "-",
        )
        row += f"{str(r2conv):<26}"
    print(row)

    # Row 3: Accuracy variance (last 10 rounds)
    row = f"{'Accuracy variance (last 10 rounds)':<42}"
    for m in header_cols:
        last10 = [a for a in all_data[m]["accs"][-10:] if a is not None]
        var = np.std(last10) if last10 else 0.0
        row += f"{var:<26.4f}"
    print(row)

    # Row 4: Accuracy drop vs baseline
    if "baseline" in all_data:
        baseline_final = all_data["baseline"]["accs"][-1]
        row = f"{'Accuracy drop vs baseline (final round)':<42}"
        for m in header_cols:
            if m == "baseline":
                row += f"{'—':<26}"
            else:
                drop = baseline_final - all_data[m]["accs"][-1]
                row += f"{drop:<26.4f}"
        print(row)

    # Row 5: Malicious vs honest local acc gap
    row = f"{'Local acc gap (malicious vs honest on poison)':<42}"
    for m in header_cols:
        if m == "baseline":
            row += f"{'—':<26}"
        else:
            data = all_data[m]
            gaps = []
            for i, r in enumerate(data["rounds"]):
                if (r in data["poison_rounds"] and data["mal_accs"][i] is not None
                        and data["hon_accs"][i] is not None):
                    gaps.append(data["hon_accs"][i] - data["mal_accs"][i])
            avg_gap = np.mean(gaps) if gaps else 0.0
            row += f"{avg_gap:<26.4f}"
    print(row)

    # Row 6: Update norm ratio on poison rounds
    row = f"{'Update norm ratio (malicious / honest on poison)':<42}"
    for m in header_cols:
        if m == "baseline":
            row += f"{'—':<26}"
        else:
            data = all_data[m]
            ratios = []
            for i, r in enumerate(data["rounds"]):
                if (r in data["poison_rounds"] and data["mal_norms"][i] is not None
                        and data["hon_norms"][i] is not None
                        and data["hon_norms"][i] > 0):
                    ratios.append(data["mal_norms"][i] / data["hon_norms"][i])
            avg_ratio = np.mean(ratios) if ratios else 0.0
            row += f"{avg_ratio:<26.2f}"
    print(row)

    # Row 7: Total poison events
    row = f"{'Total poison events (rounds × attackers)':<42}"
    for m in header_cols:
        if m == "baseline":
            row += f"{'0':<26}"
        else:
            n_pr = len(all_data[m]["poison_rounds"])
            row += f"{n_pr * 3:<26}"
    print(row)

    print("─" * 120)
    print()
    print("Key comparisons:")
    print("  - early_only vs late_only: same number of poison events (30),")
    print("    differing only by timing. If late_only is worse, the paper's")
    print("    claim about late-round availability amplifying attacks is confirmed.")
    print("  - uniform has 60 poison events (2x early/late). If late_only with")
    print("    30 events approaches uniform's damage, timing matters more than count.")
    print("  - The gray dotted line at round 20.5 separates early/late phases.")


if __name__ == "__main__":
    main()
