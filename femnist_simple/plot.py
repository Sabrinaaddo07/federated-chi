import csv
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    files = sorted(glob.glob("results_femnist_simple_*.csv"))
    if not files:
        print("No results_femnist_simple_*.csv files found.")
        sys.exit(1)

    scheme_labels = {"iid": "IID (balanced)", "non_iid": "Non-IID (Dirichlet alpha=0.1)"}
    colors = {"iid": "#2e86ab", "non_iid": "#d9534f"}
    markers = {"iid": "o", "non_iid": "s"}
    all_data = {}

    plt.figure(figsize=(12, 7))

    for f in files:
        rows = load_csv(f)
        scheme = f.replace("results_femnist_simple_", "").replace(".csv", "")
        rounds = [int(r["round"]) for r in rows]
        accs = [float(r["global_accuracy"]) for r in rows]
        all_data[scheme] = {"rounds": rounds, "accs": accs}

        plt.plot(rounds, accs, color=colors.get(scheme, "#888"),
                 marker=markers.get(scheme, "."), linewidth=1.5, markersize=4,
                 label=scheme_labels.get(scheme, scheme))

    plt.xlabel("Round", fontsize=13)
    plt.ylabel("Global Test Accuracy", fontsize=13)
    plt.title("Federated Learning on EMNIST Subset — IID vs Non-IID", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("femnist_simple_experiment.png", dpi=150)
    print("Saved to femnist_simple_experiment.png\n")

    header_order = sorted(all_data.keys())
    print("=" * 70)
    print(f"{'Metric':<40}", end="")
    for s in header_order:
        print(f"{scheme_labels.get(s, s):<30}", end="")
    print()
    print("=" * 70)

    print(f"{'Final accuracy':<40}", end="")
    for s in header_order:
        print(f"{all_data[s]['accs'][-1]:<30.4f}", end="")
    print()

    print(f"{'Rounds to >= 60%':<40}", end="")
    for s in header_order:
        d = all_data[s]
        r = next((r for r, a in zip(d["rounds"], d["accs"]) if a >= 0.60), "-")
        print(f"{str(r):<30}", end="")
    print()

    print(f"{'Accuracy variance (last 10 rounds)':<40}", end="")
    for s in header_order:
        print(f"{np.std(all_data[s]['accs'][-10:]):<30.4f}", end="")
    print()

    if len(header_order) > 1:
        base = all_data[header_order[0]]["accs"][-1]
        print(f"{'Gap vs ' + header_order[0]:<40}", end="")
        for s in header_order:
            gap = all_data[s]["accs"][-1] - base
            print(f"{gap:<+30.4f}", end="")
        print()
    print("=" * 70)


if __name__ == "__main__":
    main()
