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
    return float(row[key])


def get_int(row, key):
    return int(row[key])


def main():
    model_files = sorted(glob.glob("results_model_dropout_*.csv"))
    client_files = sorted(glob.glob("results_client_dropout_*.csv"))

    if not model_files and not client_files:
        print("No result CSV files found.")
        sys.exit(1)

    # ── Parse model dropout results ──
    model_data = {}
    for f in model_files:
        rows = load_csv(f)
        rate = float(f.replace("results_model_dropout_", "").replace(".csv", ""))
        rounds = [get_int(r, "round") for r in rows]
        accs = [get_float(r, "global_accuracy") for r in rows]
        per_class = []
        for r in rows:
            pc = json.loads(r["per_class_accuracy"])
            per_class.append({int(k): float(v) for k, v in pc.items() if v is not None})
        model_data[rate] = {"rounds": rounds, "accs": accs, "per_class": per_class}

    # ── Parse client dropout results ──
    client_data = {}
    for f in client_files:
        rows = load_csv(f)
        k = int(f.replace("results_client_dropout_", "").replace(".csv", ""))
        rounds = [get_int(r, "round") for r in rows]
        accs = [get_float(r, "global_accuracy") for r in rows]
        client_data[k] = {"rounds": rounds, "accs": accs}

    # ── Figure 1: Global accuracy + per-class bias ──
    if model_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        colors = {0.0: "#2e86ab", 0.25: "#f6a623", 0.5: "#d9534f", 0.75: "#5cb85c"}
        markers = {0.0: "o", 0.25: "s", 0.5: "^", 0.75: "D"}

        for rate in sorted(model_data):
            d = model_data[rate]
            color = colors.get(rate, "#888")
            marker = markers.get(rate, ".")
            ax1.plot(d["rounds"], d["accs"], color=color, marker=marker,
                     linewidth=1.5, markersize=4, label=f"Dropout {rate:.2f}")

        ax1.set_xlabel("Round", fontsize=12)
        ax1.set_ylabel("Global Test Accuracy", fontsize=12)
        ax1.set_title("Model Dropout — Accuracy vs Round", fontsize=13, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # ── Per-class accuracy at final round ──
        digits = list(range(10))
        x = np.arange(len(digits))
        width = 0.18
        rates_sorted = sorted(model_data)

        for i, rate in enumerate(rates_sorted):
            d = model_data[rate]
            final_pc = d["per_class"][-1]
            acc_vals = [final_pc.get(digit, 0.0) for digit in digits]
            offset = (i - (len(rates_sorted) - 1) / 2) * width
            color = colors.get(rate, "#888")
            ax2.bar(x + offset, acc_vals, width, label=f"Dropout {rate:.2f}", color=color, alpha=0.85)

        ax2.set_xlabel("Digit Class", fontsize=12)
        ax2.set_ylabel("Per-Class Accuracy", fontsize=12)
        ax2.set_title("Per-Class Accuracy at Round 40", fontsize=13, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(d) for d in digits])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig("model_dropout_experiment.png", dpi=150)
        print("Saved to model_dropout_experiment.png")
        print()

    # ── Figure 2: Resource comparison (model vs client dropout) ──
    if model_data and client_data:
        fig2, ax = plt.subplots(figsize=(10, 6))

        for rate in sorted(model_data):
            d = model_data[rate]
            color = colors.get(rate, "#888")
            marker = markers.get(rate, ".")
            ax.plot(d["rounds"], d["accs"], color=color, marker=marker,
                    linewidth=1.5, markersize=4, label=f"Model dropout {rate:.2f}")

        client_colors = {2: "#9b59b6", 4: "#e67e22"}
        client_markers = {2: "v", 4: "<"}
        for k in sorted(client_data):
            d = client_data[k]
            color = client_colors.get(k, "#888")
            marker = client_markers.get(k, ".")
            ax.plot(d["rounds"], d["accs"], color=color, marker=marker,
                    linewidth=1.5, markersize=4, linestyle="--",
                    label=f"Client dropout (K={k})")

        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Global Test Accuracy", fontsize=12)
        ax.set_title("Resource Comparison — Model Dropout vs Client Dropout", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig("resource_comparison.png", dpi=150)
        print("Saved to resource_comparison.png")
        print()

    # ── Metrics table ──
    if model_data:
        print("─" * 80)
        print(f"{'Metric':<35}", end="")
        for rate in sorted(model_data):
            print(f"{f'Dropout {rate}':<18}", end="")
        print()
        print("─" * 80)

        print(f"{'Final accuracy (round 40)':<35}", end="")
        for rate in sorted(model_data):
            d = model_data[rate]
            print(f"{d['accs'][-1]:<18.4f}", end="")
        print()

        print(f"{'Rounds to >= 85%':<35}", end="")
        for rate in sorted(model_data):
            d = model_data[rate]
            r2conv = next((r for r, a in zip(d["rounds"], d["accs"]) if a >= 0.85), "-")
            print(f"{str(r2conv):<18}", end="")
        print()

        print(f"{'Accuracy variance (r31–40)':<35}", end="")
        for rate in sorted(model_data):
            d = model_data[rate]
            last10 = d["accs"][-10:]
            print(f"{np.std(last10):<18.4f}", end="")
        print()

        baseline_final = model_data[sorted(model_data)[0]]["accs"][-1]
        print(f"{'Gap vs baseline':<35}", end="")
        for rate in sorted(model_data):
            gap = model_data[rate]["accs"][-1] - baseline_final
            sign = "+" if gap >= 0 else ""
            print(f"{sign}{gap:<17.4f}", end="")
        print()

        # Per-class accuracy variance (bias measure)
        print(f"{'Per-class acc std (round 40)':<35}", end="")
        for rate in sorted(model_data):
            d = model_data[rate]
            final_pc = d["per_class"][-1]
            vals = [v for v in final_pc.values() if v is not None]
            print(f"{np.std(vals):<18.4f}", end="")
        print()
        print("─" * 80)

    if client_data:
        print()
        print("─" * 60)
        print(f"{'Metric':<30}", end="")
        for k in sorted(client_data):
            print(f"{f'Client K={k}':<18}", end="")
        print()
        print("─" * 60)
        print(f"{'Final accuracy (round 40)':<30}", end="")
        for k in sorted(client_data):
            print(f"{client_data[k]['accs'][-1]:<18.4f}", end="")
        print()
        print("─" * 60)


if __name__ == "__main__":
    main()
