"""
client.py - Flower client for FEMNIST IID vs non-IID experiment.

Usage:
    python3 client.py --cid 0 --num_clients 50 --scheme iid
    python3 client.py --cid 0 --num_clients 50 --scheme non_iid
"""

import argparse
import logging
import os
import warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
warnings.filterwarnings("ignore")

import numpy as np
import flwr as fl
from common import (
    create_model, get_parameters, set_parameters,
    load_client_data_iid, load_client_data_non_iid, load_emnist_data,
)

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)


class FemnistClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients, scheme):
        self.cid = cid

        X_train_full, y_train_full, X_test_full, y_test_full = load_emnist_data()
        X_all = np.concatenate([X_train_full, X_test_full])
        y_all = np.concatenate([y_train_full, y_test_full])
        self.num_classes = len(np.unique(y_all))
        self.all_classes = np.arange(self.num_classes)

        if scheme == "iid":
            self.X_train, self.X_test, self.y_train, self.y_test = \
                load_client_data_iid(cid, num_clients, X_all, y_all)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                load_client_data_non_iid(cid, num_clients, X_all, y_all, alpha=0.5)

        self.model = create_model(num_classes=self.num_classes)
        print(f"  Client {cid} — {len(self.X_train)} train, {len(self.X_test)} test samples")

    def get_parameters(self, config):
        return get_parameters(self.model, num_classes=self.num_classes)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.partial_fit(self.X_train, self.y_train, classes=self.all_classes)
        return self.get_parameters(config), len(self.X_train), {"cid": self.cid}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = self.model.score(self.X_test, self.y_test)
        return 1.0 - accuracy, len(self.X_test), {"accuracy": accuracy, "cid": self.cid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--num_clients", type=int, default=50)
    parser.add_argument("--scheme", type=str, default="iid", choices=["iid", "non_iid"])
    args = parser.parse_args()

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FemnistClient(args.cid, args.num_clients, args.scheme),
    )


if __name__ == "__main__":
    main()
