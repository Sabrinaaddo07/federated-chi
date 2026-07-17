import argparse
import logging
import os
import warnings

import numpy as np

os.environ["GRPC_VERBOSITY"] = "ERROR"
warnings.filterwarnings("ignore")

import flwr as fl
from sklearn.model_selection import train_test_split
from common import (
    create_model, get_parameters, set_parameters,
    load_full_data, NUM_CLASSES,
)

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)


class BaselineClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients):
        self.cid = cid
        self.X_full, self.y_full = load_full_data()
        # Reserve a small fixed test set per client (1/500th, same as training sample)
        rng = np.random.RandomState(seed=999 + cid)
        n_test = len(self.X_full) // 500
        idx = rng.choice(len(self.X_full), n_test, replace=False)
        self.X_test, self.y_test = self.X_full[idx], self.y_full[idx]
        # Remove test samples from full set to avoid data leakage
        mask = np.ones(len(self.X_full), dtype=bool)
        mask[idx] = False
        self.X_full, self.y_full = self.X_full[mask], self.y_full[mask]
        self.model = create_model()
        self.all_classes = NUM_CLASSES
        self.fit_count = 0
        self.num_clients = num_clients
        print(f"  Client {cid} — {len(self.X_full)} train pool, {len(self.X_test)} test samples")

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.fit_count += 1
        self.set_parameters(parameters)

        # Sample a fresh 1/500th of the data each round
        # Paper: each client holds 1/500th of total data
        n_sample = len(self.X_full) // 500  # = 100
        rng = np.random.RandomState(seed=42 + self.cid * 1000 + self.fit_count)
        idx = rng.choice(len(self.X_full), n_sample, replace=False)
        X_sample, y_sample = self.X_full[idx], self.y_full[idx]

        # Split into train/test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=0.2, random_state=42,
        )

        self.model.partial_fit(
            X_train, y_train,
            classes=np.arange(self.all_classes),
        )
        return self.get_parameters(config), len(X_train), {"cid": self.cid}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = self.model.score(self.X_test, self.y_test)
        return 1.0 - acc, len(self.X_test), {"accuracy": acc, "cid": self.cid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID (0-9)")
    parser.add_argument("--num_clients", type=int, default=10, help="Total clients")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    print(f"--- Client {args.cid} starting ---")
    print(f"  Connecting to server at {args.server_address}")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=BaselineClient(args.cid, args.num_clients),
    )


if __name__ == "__main__":
    main()
