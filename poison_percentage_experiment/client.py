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
    load_client_data, NUM_CLASSES,
)

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)


class PoisonClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients, malicious):
        self.cid = cid
        self.malicious = malicious
        self.X_train, self.X_test, self.y_train, self.y_test = \
            load_client_data(cid, num_clients)
        self.model = create_model()
        self.all_classes = np.arange(NUM_CLASSES)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if self.malicious:
            rng = np.random.RandomState(seed=42 + self.cid)
            y_poison = rng.randint(0, NUM_CLASSES, size=len(self.y_train))
            self.model.partial_fit(self.X_train, y_poison, classes=self.all_classes)
        else:
            self.model.partial_fit(self.X_train, self.y_train, classes=self.all_classes)
        return self.get_parameters(config), len(self.X_train), {"cid": self.cid}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = self.model.score(self.X_test, self.y_test)
        return 1.0 - acc, len(self.X_test), {"accuracy": acc, "cid": self.cid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--malicious", action="store_true")
    args = parser.parse_args()

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=PoisonClient(args.cid, args.num_clients, args.malicious),
    )


if __name__ == "__main__":
    main()
