import argparse
import csv
import logging
import os
import time
import warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
warnings.filterwarnings("ignore")

from typing import List, Tuple, Optional, Dict

import flwr as fl
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, EvaluateIns, FitRes, EvaluateRes, Parameters
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from sklearn.metrics import accuracy_score
from common import create_model, get_parameters, set_parameters, load_server_test_data

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)

SERVER_X_TEST, SERVER_Y_TEST = load_server_test_data()


class PoisonPctFedAvg(Strategy):
    def __init__(self, initial_parameters, malicious_pct, run_id,
                 num_rounds=40, min_available=10):
        self.initial_parameters = initial_parameters
        self.malicious_pct = malicious_pct
        self.run_id = run_id
        self.num_rounds = num_rounds
        self.min_available = min_available
        self.global_model = None
        self.csv_file = None
        self.csv_writer = None

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        n = client_manager.num_available()
        while n < self.min_available:
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

        print(f"  Round {server_round:2d} — Global acc: {global_acc:.4f}")

        if self.csv_writer is None:
            csv_name = f"results_pct{self.malicious_pct}_run{self.run_id}.csv"
            self.csv_file = open(csv_name, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["round", "malicious_pct", "run_id", "global_accuracy"])
        self.csv_writer.writerow([server_round, self.malicious_pct, self.run_id, f"{global_acc:.4f}"])
        self.csv_file.flush()

        time.sleep(0.5)
        return ndarrays_to_parameters(avg_weights), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        n = client_manager.num_available()
        while n < self.min_available:
            time.sleep(1)
            n = client_manager.num_available()
        clients = client_manager.sample(num_clients=n, min_num_clients=1)
        return [(client, EvaluateIns(parameters, {})) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None

    def close(self):
        if self.csv_file:
            self.csv_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--malicious_pct", type=float, required=True)
    parser.add_argument("--run_id", type=int, default=0)
    args = parser.parse_args()

    initial_model = create_model()
    initial_weights = get_parameters(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    strategy = PoisonPctFedAvg(
        initial_parameters=initial_parameters,
        malicious_pct=args.malicious_pct,
        run_id=args.run_id,
        min_available=10,
    )

    print(f"\n  Malicious: {args.malicious_pct}%  |  Run: {args.run_id}  |  10 clients, 40 rounds\n")

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=40),
        strategy=strategy,
    )

    strategy.close()


if __name__ == "__main__":
    main()
