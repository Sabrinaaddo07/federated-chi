import argparse
import csv
import os
from common import DATASET_INFO, train_centralized_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=['mnist', 'cifar10', 'cifar100'])
    args = parser.parse_args()

    acc, model = train_centralized_baseline(args.dataset)
    print(f"{args.dataset} centralized baseline accuracy: {acc*100:.2f}%")

    outfile = "centralized_baselines.csv"
    rows = []
    if os.path.exists(outfile):
        with open(outfile, newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["dataset"] != args.dataset]
    rows.append({"dataset": args.dataset, "accuracy": f"{acc:.4f}"})
    rows.sort(key=lambda r: r["dataset"])
    with open(outfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "accuracy"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()
