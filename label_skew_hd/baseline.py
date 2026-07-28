import argparse
from common import DATASET_INFO, train_centralized_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=['mnist', 'cifar10', 'cifar100'])
    args = parser.parse_args()

    acc, model = train_centralized_baseline(args.dataset)
    print(f"{args.dataset} centralized baseline accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
