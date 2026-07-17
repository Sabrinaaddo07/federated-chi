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
    load_partitioned_data, NUM_CLASSES,
)

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)


class BaselineClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients, seed):
        self.cid = cid
        self.seed = seed
        self.X_data, self.y_data = load_partitioned_data(cid, num_clients)
        self.model = create_model()
        self.all_classes = NUM_CLASSES
        self.fit_count = 0
        self.num_clients = num_clients
        print(f"  Client {cid} — {len(self.X_data)} train pool (exclusive partition), seed={seed}")

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.fit_count += 1
        self.set_parameters(parameters)

        # Sample a fresh 1/500th (=100) from this client's exclusive partition
        n_sample = len(self.X_data) // 50  # 5000 / 50 = 100
        rng = np.random.RandomState(seed=self.seed + self.cid * 1000 + self.fit_count)
        idx = rng.choice(len(self.X_data), n_sample, replace=False)
        X_sample, y_sample = self.X_data[idx], self.y_data[idx]

        self.model.partial_fit(
            X_sample, y_sample,
            classes=np.arange(self.all_classes),
        )
        return self.get_parameters(config), len(X_sample), {"cid": self.cid}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, 0, {"cid": self.cid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID (0-9)")
    parser.add_argument("--num_clients", type=int, default=10, help="Total clients")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data sampling")
    args = parser.parse_args()

    print(f"--- Client {args.cid} starting ---")
    print(f"  Connecting to server at {args.server_address}")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=BaselineClient(args.cid, args.num_clients, args.seed),
    )


if __name__ == "__main__":
    main()
