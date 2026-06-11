"""
server_non_iid.py - Flower server for non-IID / IID data split experiments.

No dropout — all 8 clients participate every round.
Results are logged to results_non_iid_{scheme}.csv

Usage:
  python3 server_non_iid.py --scheme iid
  python3 server_non_iid.py --scheme a
  python3 server_non_iid.py --scheme b
"""

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


class NonIIDFedAvg(Strategy):
    """
    Federated Averaging for non-IID experiments.
    Simpler than ExplicitFedAvg — no dropout, just weighted averaging + CSV logging.
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        scheme: str = "iid",
        fraction_fit: float = 1.0,
        min_fit_clients: int = 1,
        min_available_clients: int = 1,
        fraction_evaluate: float = 1.0,
        min_evaluate_clients: int = 1,
    ):
        self.initial_parameters = initial_parameters
        self.scheme = scheme
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.fraction_evaluate = fraction_evaluate
        self.min_evaluate_clients = min_evaluate_clients
        self.global_model = None

        self.csv_file = None
        self.csv_writer = None

        self.num_available = 0

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        num_available = client_manager.num_available()
        while num_available < self.min_available_clients:
            time.sleep(2)
            num_available = client_manager.num_available()
            print(f"  Waiting for clients... {num_available}/{self.min_available_clients}")

        n = max(
            ceil(self.fraction_fit * num_available),
            self.min_fit_clients,
        )
        clients = client_manager.sample(
            num_clients=n, min_num_clients=self.min_fit_clients
        )
        self.num_available = len(clients)

        return [(client, FitIns(parameters, {})) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict]:
        if not results:
            return None, {}

        weights_list = [parameters_to_ndarrays(r.parameters) for _, r in results]
        participating_cids = [r.metrics.get("cid", "?") for _, r in results
                              if r.metrics]
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

        print(f"\nRound {server_round} — Global test accuracy: {global_acc:.4f}")
        print(f"    Participants: {len(results)} of {self.num_available} "
              f"| CIDs: {sorted(participating_cids)}")

        if self.csv_writer is None:
            self.csv_file = open(
                f"results_non_iid_{self.scheme}.csv",
                "w", newline="",
            )
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "round", "scheme", "global_accuracy",
                "num_participated", "num_available", "participating_cids",
            ])
        self.csv_writer.writerow([
            server_round,
            self.scheme,
            f"{global_acc:.4f}",
            len(results),
            self.num_available,
            sorted(participating_cids),
        ])
        self.csv_file.flush()

        print("  (pausing 3 seconds before next round...)")
        time.sleep(1.5)

        return ndarrays_to_parameters(avg_weights), {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        n = max(
            ceil(self.fraction_evaluate * client_manager.num_available()),
            self.min_evaluate_clients,
        )
        num_available_eval = client_manager.num_available()
        while num_available_eval < self.min_available_clients:
            time.sleep(2)
            num_available_eval = client_manager.num_available()
        clients = client_manager.sample(
            num_clients=n, min_num_clients=self.min_evaluate_clients
        )
        return [(client, EvaluateIns(parameters, {})) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes]],
    ) -> Tuple[Optional[float], Dict]:
        for _, r in results:
            if r.metrics:
                cid = r.metrics.get("cid", "?")
                acc = r.metrics["accuracy"]
                print(f"    Client {cid} local accuracy: {acc:.4f}")
        return None, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict]]:
        return None

    def close(self):
        if self.csv_file is not None:
            self.csv_file.close()


def main():
    parser = argparse.ArgumentParser(
        description="Federated learning server for non-IID data split experiments."
    )
    parser.add_argument(
        "--scheme", type=str, default="iid",
        choices=["iid", "a", "b"],
        help="Data split scheme: iid (balanced), a (1 class/client, "
             "classes 0-7 only), b (1 class/client but client 7 "
             "gets classes 7-9)",
    )
    args = parser.parse_args()
    scheme = args.scheme

    scheme_descriptions = {
        "iid": "IID (balanced across all 10 classes)",
        "a": "Non-IID A (1 digit class per client, classes 8-9 held out)",
        "b": "Non-IID B (1 class/client except client 7 gets classes 7,8,9)",
    }

    print("=" * 60)
    print(f"  NON-IID EXPERIMENT — Flower + scikit-learn")
    print(f"  Scheme: {scheme_descriptions[scheme]}")
    print("=" * 60)
    print()
    print("  The server holds a Logistic Regression model.")
    print("  All 8 clients participate every round (no dropout).")
    print("  Data NEVER leaves the clients — only weights are shared.")
    print()
    print("  Open new terminals and run (all 8 before round 1):")
    print(f"    python3 client_non_iid.py --cid 0 --num_clients 8 --scheme {scheme}")
    print(f"    python3 client_non_iid.py --cid 1 --num_clients 8 --scheme {scheme}")
    print(f"    python3 client_non_iid.py --cid 2 --num_clients 8 --scheme {scheme}")
    print(f"    python3 client_non_iid.py --cid 3 --num_clients 8 --scheme {scheme}")
    print(f"    python3 client_non_iid.py --cid 4 --num_clients 8 --scheme {scheme}")
    print(f"    python3 client_non_iid.py --cid 5 --num_clients 8 --scheme {scheme}")
    print(f"    python3 client_non_iid.py --cid 6 --num_clients 8 --scheme {scheme}")
    print(f"    python3 client_non_iid.py --cid 7 --num_clients 8 --scheme {scheme}")
    print()
    print("  Ctrl+C any client to remove it; server keeps going.")
    print("  Ctrl+C the server to stop entirely.")
    print()

    initial_model = create_model()
    initial_weights = get_parameters(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    strategy = NonIIDFedAvg(
        initial_parameters=initial_parameters,
        scheme=scheme,
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=8,
        fraction_evaluate=1.0,
        min_evaluate_clients=1,
    )

    print(f"  Server listening on 127.0.0.1:8080")
    print("  Waiting for 8 clients...")
    print()

    try:
        fl.server.start_server(
            server_address="127.0.0.1:8080",
            config=fl.server.ServerConfig(num_rounds=40),
            strategy=strategy,
        )
    except KeyboardInterrupt:
        pass
    finally:
        strategy.close()


if __name__ == "__main__":
    main()
