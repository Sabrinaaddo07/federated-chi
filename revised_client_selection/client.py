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
    load_client_data_iid, NUM_CLASSES,
)

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)


class BaselineClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients):
        self.cid = cid
        self.X_train, self.X_test, self.y_train, self.y_test = \
            load_client_data_iid(cid, num_clients)
        self.model = create_model()
        self.all_classes = NUM_CLASSES

        print(f"  Client {cid} — {len(self.X_train)} train, {len(self.X_test)} test samples")

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.partial_fit(
            self.X_train, self.y_train,
            classes=np.arange(self.all_classes),
        )
        return self.get_parameters(config), len(self.X_train), {"cid": self.cid}

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
