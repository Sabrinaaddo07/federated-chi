"""
client_poison.py - Flower client with optional label-flipping poisoning.

All clients use IID data. Use --malicious to mark this client as the attacker.

Usage:
  python3 client_poison.py --cid 0
  python3 client_poison.py --cid 7 --malicious
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
)

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)


class PoisonClient(fl.client.NumPyClient):

    def __init__(self, cid, num_clients, malicious):
        self.cid = cid
        self.malicious = malicious
        self.all_classes = np.arange(10)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            load_client_data_iid(cid, num_clients=num_clients)

        self.model = create_sgd_model()

        classes_present = sorted(set(self.y_train))
        mal_str = " [MALICIOUS]" if malicious else ""
        print(f"  Client {cid}{mal_str} — {len(self.X_train)} train, "
              f"{len(self.X_test)} test samples | classes: {classes_present}")

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        old_weights = get_parameters(self.model)

        is_poison_round = config.get("poison_round", False)

        if self.malicious and is_poison_round:
            rng = np.random.RandomState(seed=config.get("poison_seed", 0))
            y_train_poisoned = rng.randint(0, 10, size=len(self.y_train))
            self.model.partial_fit(self.X_train, y_train_poisoned, classes=self.all_classes)
        else:
            self.model.partial_fit(self.X_train, self.y_train, classes=self.all_classes)

        new_weights = get_parameters(self.model)
        update_norm = np.sqrt(
            sum(np.sum((n - o) ** 2) for n, o in zip(new_weights, old_weights))
        )

        local_acc = self.model.score(self.X_test, self.y_test)

        return (
            self.get_parameters(config),
            len(self.X_train),
            {
                "cid": self.cid,
                "accuracy": local_acc,
                "update_norm": update_norm,
                "poisoned": int(is_poison_round and self.malicious),
            },
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        local_acc = self.model.score(self.X_test, self.y_test)
        return 1.0 - local_acc, len(self.X_test), {"accuracy": local_acc, "cid": self.cid}


def main():
    parser = argparse.ArgumentParser(
        description="Flower client for poisoning experiments."
    )
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--malicious", action="store_true",
                        help="Mark this client as malicious")
    args = parser.parse_args()

    role = "MALICIOUS" if args.malicious else "honest"
    print(f"--- Client {args.cid} [{role}] ---")

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=PoisonClient(args.cid, args.num_clients, args.malicious),
    )


if __name__ == "__main__":
    main()
