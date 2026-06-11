"""
client_non_iid.py - Flower client for non-IID / IID data split experiments.

Usage:
  python3 client_non_iid.py --cid 0 --scheme iid
  python3 client_non_iid.py --cid 0 --scheme a
  python3 client_non_iid.py --cid 0 --scheme b
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
    create_sgd_model,
    get_parameters,
    set_parameters,
    load_client_data_iid,
    load_client_data_non_iid,
)

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)


class DigitClient(fl.client.NumPyClient):

    def __init__(self, cid, num_clients, scheme):
        self.cid = cid
        self.scheme = scheme

        if scheme == "iid":
            self.X_train, self.X_test, self.y_train, self.y_test = \
                load_client_data_iid(cid, num_clients=num_clients)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                load_client_data_non_iid(cid, num_clients=num_clients, scheme=scheme)

        self.model = create_sgd_model()
        self.all_classes = np.arange(10)

        if scheme != "iid":
            classes_present = sorted(set(self.y_train))
            print(f"  Client {cid} (scheme {scheme}) — {len(self.X_train)} train, "
                  f"{len(self.X_test)} test samples | classes: {classes_present}")
        else:
            print(f"  Client {cid} (IID) — {len(self.X_train)} train, "
                  f"{len(self.X_test)} test samples")
        print(f"  This client's data is PRIVATE — never sent to the server\n")

    def get_parameters(self, config):
        return get_parameters(self.model)

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
    parser = argparse.ArgumentParser(
        description="Flower client for non-IID data split experiment."
    )
    parser.add_argument("--cid", type=int, required=True,
                        help="Client ID (0, 1, 2, ..., 7)")
    parser.add_argument("--num_clients", type=int, default=8,
                        help="Total number of clients")
    parser.add_argument("--scheme", type=str, default="iid",
                        choices=["iid", "a", "b"],
                        help="Data split scheme: iid (balanced), a (1 class/client, "
                             "classes 0-7 only), b (1 class/client but client 7 "
                             "gets classes 7-9)")
    args = parser.parse_args()

    scheme_names = {
        "iid": "IID (balanced across all 10 classes)",
        "a": "Non-IID A (1 digit class per client, classes 8-9 held out)",
        "b": "Non-IID B (1 class/client except client 7 gets classes 7,8,9)",
    }
    print(f"--- Client {args.cid} starting [{scheme_names[args.scheme]}] ---")

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=DigitClient(args.cid, args.num_clients, args.scheme),
    )


if __name__ == "__main__":
    main()
