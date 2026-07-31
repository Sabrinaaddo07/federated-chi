import argparse
import logging
import os
import warnings

import numpy as np

os.environ["GRPC_VERBOSITY"] = "ERROR"
warnings.filterwarnings("ignore")

import flwr as fl
from common import (
    create_model, get_parameters, set_parameters,
    load_dataset, dirichlet_partition, HD_ALPHA_MAP, DATASET_INFO,
)

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)


class HDCilent(fl.client.NumPyClient):
    def __init__(self, cid, dataset, alpha, num_clients, seed):
        self.cid = cid
        self.dataset = dataset
        X_train, y_train, _, _ = load_dataset(dataset)
        self.X_train, self.y_train = dirichlet_partition(
            alpha, num_clients, X_train, y_train, seed=seed, client_idx=self.cid,
        )
        self.model = create_model(dataset)
        self.fit_count = 0
        self.all_classes = DATASET_INFO[dataset]['num_classes']
        print(f"  Client {cid} — {len(self.X_train)} train samples")

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.fit_count += 1
        self.set_parameters(parameters)

        if len(self.X_train) > 0:
            self.model.partial_fit(
                self.X_train, self.y_train, classes=np.arange(self.all_classes),
            )
        return self.get_parameters(config), max(len(self.X_train), 1), {"cid": self.cid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['mnist', 'cifar10', 'cifar100'])
    parser.add_argument("--hd", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    alpha = HD_ALPHA_MAP[args.hd]
    print(f"--- Client {args.cid} | {args.dataset} | HD={args.hd} α={alpha} | "
          f"connecting to {args.server_address} ---")

    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=HDCilent(args.cid, args.dataset, alpha, args.num_clients, args.seed),
    )


if __name__ == "__main__":
    main()
