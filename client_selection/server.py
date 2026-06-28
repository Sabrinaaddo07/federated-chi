import argparse
import csv
import logging
import os
import time
import warnings

os.environ["GRPC_VERBOSITY"] = "ERROR"
warnings.filterwarnings("ignore")

from typing import List, Optional

import numpy as np
import flwr as fl
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.common import FitIns, EvaluateIns, FitRes, EvaluateRes, Parameters
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from sklearn.metrics import accuracy_score
from common import (
    create_model, get_parameters, set_parameters,
    load_server_test_data, NUM_CLASSES,
)

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)

NUM_CLIENTS = 10
NUM_ROUNDS = 40

SERVER_X_TEST, SERVER_Y_TEST = load_server_test_data()


class LossWeightedClientManager(SimpleClientManager):
    def __init__(self, loss_dict):
        super().__init__()
        self.loss_dict = loss_dict

    def sample(self, num_clients, min_num_clients=None):
        available = list(self.clients.values())
        if len(available) < (min_num_clients or num_clients):
            raise RuntimeError(
                f"Not enough clients: {len(available)} < {min_num_clients or num_clients}"
            )
        cids = [c.cid for c in available]
        losses = np.array([self.loss_dict.get(cid, 1.0) for cid in cids])
        eps = 1e-10
        if losses.sum() < eps:
            probs = np.ones(len(available)) / len(available)
        else:
            probs = losses / losses.sum()
        indices = np.random.choice(
            len(available), size=num_clients, replace=False, p=probs
        )
        return [available[i] for i in indices]


class BaselineFedAvg(Strategy):
    def __init__(self, initial_parameters, num_clients=NUM_CLIENTS,
                 clients_per_round=NUM_CLIENTS):
        self.initial_parameters = initial_parameters
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.global_model = None
        self.csv_file = None
        self.csv_writer = None
        self.client_losses = {str(i): 1.0 for i in range(num_clients)}
        self.model_bytes = sum(
            p.nbytes for p in parameters_to_ndarrays(initial_parameters)
        )
        self.cumulative_overhead_bytes = 0
        self.start_time = time.time()

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        while client_manager.num_available() < self.num_clients:
            time.sleep(2)
            num_avail = client_manager.num_available()
            print(f"  Waiting for clients... {num_avail}/{self.num_clients}")
        k = min(self.clients_per_round, client_manager.num_available())
        clients = client_manager.sample(num_clients=k, min_num_clients=k)
        self.cumulative_overhead_bytes += self.model_bytes * 2 * k
        return [(client, FitIns(parameters, {})) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights_list = [parameters_to_ndarrays(r.parameters) for _, r in results]
        participating_cids = [
            r.metrics.get("cid", "?") for _, r in results if r.metrics
        ]
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

        elapsed = time.time() - self.start_time

        for _, res in results:
            if res.metrics and "train_loss" in res.metrics:
                cid = res.metrics.get("cid", "")
                self.client_losses[str(cid)] = res.metrics["train_loss"]

        print(f"  Round {server_round:2d} — Global test accuracy: {global_acc:.4f}  "
              f"| {len(results)} clients | CIDs: {sorted(participating_cids)}")

        if self.csv_writer is None:
            self.csv_file = open("results_clientsel.csv", "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "round", "global_accuracy", "num_participated", "participating_cids",
                "cumulative_overhead_bytes", "elapsed_time_seconds",
            ])
        self.csv_writer.writerow([
            server_round, f"{global_acc:.4f}", len(results),
            sorted(participating_cids),
            self.cumulative_overhead_bytes, f"{elapsed:.2f}",
        ])
        self.csv_file.flush()

        print("  (pausing 1.5 seconds before next round...)")
        time.sleep(1.5)

        return ndarrays_to_parameters(avg_weights), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        n = client_manager.num_available()
        clients = client_manager.sample(
            num_clients=n, min_num_clients=min(n, 1)
        )
        return [(client, EvaluateIns(parameters, {})) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        for _, r in results:
            if r.metrics:
                acc = r.metrics.get("accuracy", 0.0)
                cid = r.metrics.get("cid", "?")
                print(f"    Client {cid} local acc: {acc:.4f}")
        return None, {}

    def evaluate(self, server_round, parameters):
        return None

    def close(self):
        if self.csv_file:
            self.csv_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, default="10.0.0.1:8080")
    parser.add_argument("--num_clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--num_rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--clients_per_round", type=int, default=NUM_CLIENTS,
                        help="Number of clients sampled per round (loss-weighted)")
    args = parser.parse_args()

    print("=" * 55)
    print("  CLIENT SELECTION — Federated Learning")
    print(f"  {args.num_clients} clients  |  {args.num_rounds} rounds")
    print(f"  {args.clients_per_round} clients/round  |  loss-weighted selection")
    print(f"  Server listening on {args.server_address}")
    print("=" * 55)
    print()

    initial_model = create_model()
    initial_weights = get_parameters(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    strategy = BaselineFedAvg(
        initial_parameters=initial_parameters,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
    )

    client_manager = LossWeightedClientManager(strategy.client_losses)

    try:
        fl.server.start_server(
            server_address=args.server_address,
            config=fl.server.ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
            client_manager=client_manager,
        )
    except KeyboardInterrupt:
        pass
    finally:
        strategy.close()


if __name__ == "__main__":
    main()
