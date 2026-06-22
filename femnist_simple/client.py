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
    load_client_data_iid, load_client_data_non_iid,
    load_emnist_subset, NUM_CLASSES,
)

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)


class FemnistClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients, scheme):
        X_tr, y_tr, X_te, y_te = load_emnist_subset()
        X_all = np.concatenate([X_tr, X_te])
        y_all = np.concatenate([y_tr, y_te])

        if scheme == "iid":
            self.X_train, self.X_test, self.y_train, self.y_test = \
                load_client_data_iid(cid, num_clients, X_all, y_all)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                load_client_data_non_iid(cid, num_clients, X_all, y_all, alpha=0.1)

        self.cid = cid
        self.model = create_model()
        self.all_classes = np.arange(NUM_CLASSES)

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
        acc = self.model.score(self.X_test, self.y_test)
        return 1.0 - acc, len(self.X_test), {"accuracy": acc, "cid": self.cid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--scheme", type=str, default="iid", choices=["iid", "non_iid"])
    args = parser.parse_args()

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FemnistClient(args.cid, args.num_clients, args.scheme),
    )


if __name__ == "__main__":
    main()
