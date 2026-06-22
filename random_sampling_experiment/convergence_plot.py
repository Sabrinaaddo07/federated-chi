import csv
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np

CONVERGENCE_THRESHOLD = 0.85
STABILITY_ROUNDS = 3

COLORS = {
    2: "#f6a623",
    4: "#d9534f",
    6: "#5cb85c",
    8: "#9b59b6",
    10: "#2e86ab",
}
MARKERS = {2: "s", 4: "^", 6: "D", 8: "v", 10: "o"}

K_ORDER = [2, 4, 6, 8, 10]


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_convergence_round(accs):
    for i in range(len(accs) - STABILITY_ROUNDS + 1):
        if accs[i] >= CONVERGENCE_THRESHOLD:
            window = accs[i : i + STABILITY_ROUNDS]
            if sum(window) / len(window) >= CONVERGENCE_THRESHOLD:
                return i + 1
    return None


def main():
    files = sorted(glob.glob("results_sample_*.csv"))
    if not files:
        print("No results_sample_*.csv files found.")
        sys.exit(1)

    all_data = {}
    convergence_rounds = {}

    for f in files:
        rows = load_csv(f)
        label = f.replace("results_sample_", "").replace(".csv", "")
        try:
            k = int(label)
        except ValueError:
            continue
        rounds = [int(r["round"]) for r in rows]
        accs = [float(r["global_accuracy"]) for r in rows]
        all_data[k] = {"rounds": rounds, "accs": accs}
        conv_r = find_convergence_round(accs)
        convergence_rounds[k] = conv_r

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    for k in K_ORDER:
        if k not in all_data:
            continue
        d = all_data[k]
        color = COLORS.get(k, "#888")
        marker = MARKERS.get(k, ".")
        label = f"K={k} of 10"

        for ax in (ax1, ax2):
            ax.plot(d["rounds"], d["accs"], color=color, marker=marker,
                    linewidth=1.5, markersize=4, label=label)

        conv_r = convergence_rounds.get(k)
        if conv_r is not None:
            conv_idx = conv_r - 1
            conv_acc = d["accs"][conv_idx]
            for ax in (ax1, ax2):
                ax.annotate(
                    f"K={k}  r{conv_r}",
                    xy=(conv_r, conv_acc),
                    xytext=(conv_r + 4, conv_acc - 0.05 + 0.02 * (K_ORDER.index(k) - 2)),
                    fontsize=8, fontweight="bold",
                    color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                )
                ax.axvline(x=conv_r, color=color, linestyle=":", alpha=0.5, linewidth=1)
                ax.plot(conv_r, conv_acc, "*", color=color, markersize=18, zorder=5)

    for ax in (ax1, ax2):
        ax.axhline(y=CONVERGENCE_THRESHOLD, color="gray", linestyle="--",
                   alpha=0.6, linewidth=1, label=f"{CONVERGENCE_THRESHOLD:.0%} threshold")
        ax.grid(True, alpha=0.3)

    ax1.set_xlabel("Round", fontsize=13)
    ax1.set_ylabel("Global Test Accuracy", fontsize=13)
    ax1.set_title("Full View (40 rounds)", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=9, loc="lower right")

    ax2.set_xlabel("Round", fontsize=13)
    ax2.set_ylabel("Global Test Accuracy", fontsize=13)
    ax2.set_title("Zoomed View — Convergence Region", fontsize=14, fontweight="bold")
    ax2.set_xlim(0, 25)
    ax2.set_ylim(0.55, 1.0)
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig("convergence_plot.png", dpi=150)
    print("Saved to convergence_plot.png")
    print()

    print("=" * 70)
    print(f"{'Metric':<30}", end="")
    for k in K_ORDER:
        if k in all_data:
            print(f"{f'K={k} of 10':<15}", end="")
    print()
    print("=" * 70)

    print(f"{'Convergence round (>=85%)':<30}", end="")
    for k in K_ORDER:
        if k in all_data:
            r = convergence_rounds.get(k, "-")
            print(f"{str(r):<15}", end="")
    print()

    print(f"{'Final accuracy (round 40)':<30}", end="")
    for k in K_ORDER:
        if k in all_data:
            print(f"{all_data[k]['accs'][-1]:<15.4f}", end="")
    print()

    print(f"{'Rounds behind K=10':<30}", end="")
    if 10 in convergence_rounds:
        baseline = convergence_rounds[10]
        for k in K_ORDER:
            if k in all_data:
                r = convergence_rounds.get(k)
                if r is not None:
                    print(f"+{r - baseline:<14}", end="")
                else:
                    print(f"{'N/A':<15}", end="")
    print()

    print(f"{'Accuracy variance (r31-40)':<30}", end="")
    for k in K_ORDER:
        if k in all_data:
            last10 = all_data[k]["accs"][-10:]
            print(f"{np.std(last10):<15.4f}", end="")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
