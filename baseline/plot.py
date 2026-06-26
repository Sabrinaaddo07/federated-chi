import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    csv_path = "results_baseline.csv"
    if not os.path.exists(csv_path):
        print(f"No results file found ({csv_path}). Run the experiment first.")
        sys.exit(1)

    import os

    rows = load_csv(csv_path)
    rounds = [int(r["round"]) for r in rows]
    accs = [float(r["global_accuracy"]) for r in rows]

    plt.figure(figsize=(12, 6))

    plt.plot(rounds, accs, color="#2e86ab", marker="o", linewidth=2, markersize=5,
             label="Global test accuracy")

    plt.xlabel("Round", fontsize=13)
    plt.ylabel("Global Test Accuracy", fontsize=13)
    plt.title("Baseline Federated Learning — 10 clients, 40 rounds, IID",
              fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("baseline_experiment.png", dpi=150)
    print("Saved to baseline_experiment.png")

    print()
    print("=" * 60)
    print(f"{'Metric':<45}{'Value':<15}")
    print("=" * 60)
    print(f"{'Final accuracy (round 40)':<45}{accs[-1]:<15.4f}")
    print(f"{'Rounds to reach >= 60%':<45}", end="")
    r60 = next((r for r, a in zip(rounds, accs) if a >= 0.60), "-")
    print(f"{str(r60):<15}")

    print(f"{'Rounds to reach >= 80%':<45}", end="")
    r80 = next((r for r, a in zip(rounds, accs) if a >= 0.80), "-")
    print(f"{str(r80):<15}")

    print(f"{'Accuracy variance (last 10 rounds)':<45}{np.std(accs[-10:]):<15.4f}")
    print(f"{'Accuracy range (last 10 rounds)':<45}{max(accs[-10:]) - min(accs[-10:]):<15.4f}")
    print(f"{'Peak accuracy':<45}{max(accs):<15.4f}")
    print(f"{'Peak accuracy at round':<45}{rounds[accs.index(max(accs))]:<15}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
