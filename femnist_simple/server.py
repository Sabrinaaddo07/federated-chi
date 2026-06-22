import argparse
import csv
import logging
import os
import time
import warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
warnings.filterwarnings("ignore")

from math import ceil
from typing import List, Tuple, Optional, Dict

import numpy as np
import flwr as fl
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, EvaluateIns, FitRes, EvaluateRes, Parameters
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from sklearn.metrics import accuracy_score
from common import (
    create_model, get_parameters, set_parameters,
    load_emnist_subset, NUM_CLASSES,
)

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)

NUM_CLIENTS = 5
NUM_ROUNDS = 30

_, _, SERVER_X_TEST, SERVER_Y_TEST = load_emnist_subset()


class FemnistFedAvg(Strategy):
    def __init__(self, initial_parameters, scheme):
        self.initial_parameters = initial_parameters
        self.scheme = scheme
        self.global_model = None
        self.csv_file = None
        self.csv_writer = None

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        n = client_manager.num_available()
        while n < 1:
            time.sleep(2)
            n = client_manager.num_available()
        clients = client_manager.sample(num_clients=n, min_num_clients=1)
        return [(client, FitIns(parameters, {})) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights_list = [parameters_to_ndarrays(r.parameters) for _, r in results]
        num_examples = [r.num_examples for _, r in results]
        total = sum(num_examples)

        avg_weights = [
            sum(w[i] * n for w, n in zip(weights_list, num_examples)) / total
            for i in range(len(weights_list[0]))
        ]

        self.global_model = create_model()
        set_parameters(self.global_model, avg_weights)

        y_pred = self.global_model.predict(SERVER_X_TEST)
        global_acc = accuracy_score(SERVER_Y_TEST, y_pred)

        print(f"  Round {server_round:2d} — Global test accuracy: {global_acc:.4f}  | {len(results)} clients")

        if self.csv_writer is None:
            self.csv_file = open(f"results_femnist_simple_{self.scheme}.csv", "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["round", "scheme", "global_accuracy"])
        self.csv_writer.writerow([server_round, self.scheme, f"{global_acc:.4f}"])
        self.csv_file.flush()

        return ndarrays_to_parameters(avg_weights), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        n = max(ceil(1.0 * client_manager.num_available()), 1)
        clients = client_manager.sample(num_clients=n, min_num_clients=1)
        return [(client, EvaluateIns(parameters, {})) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        for _, r in results:
            if r.metrics:
                acc = r.metrics["accuracy"]
                print(f"    Client {r.metrics.get('cid', '?')} local acc: {acc:.4f}")
        return None, {}

    def evaluate(self, server_round, parameters):
        return None

    def close(self):
        if self.csv_file:
            self.csv_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", type=str, default="iid", choices=["iid", "non_iid"])
    args = parser.parse_args()

    print(f"\n  FEMNIST SIMPLE — {args.scheme.upper()}, {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds\n")

    initial_model = create_model()
    initial_weights = get_parameters(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    strategy = FemnistFedAvg(
        initial_parameters=initial_parameters,
        scheme=args.scheme,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    strategy.close()


if __name__ == "__main__":
    main()
