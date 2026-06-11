"""
server.py - The central orchestrator in the federated learning system.

Supports mid-experiment persistent client dropout:
  --dropout 0            Baseline (no dropout)
  --dropout 2            Drop 2 of 8 clients
  --dropout 4            Drop 4 of 8 clients
  --dropout 6            Drop 6 of 8 clients
  --dropout_round 20     Round at which dropout triggers (default 20)

Results are logged to results_dropout_N.csv
"""

import argparse
import csv
import logging
import os
import random
import time
import warnings

# Suppress Flower's verbose logs
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

# After import, suppress Flower's logger
flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)

# Pre-load the server's held-out test set
SERVER_X_TEST, SERVER_Y_TEST = load_server_test_data()


class ExplicitFedAvg(Strategy):
    """
    Federated Averaging (FedAvg) implemented from the base Strategy class.

    Supports mid-experiment persistent dropout: N clients are permanently
    dropped at a configurable round, simulating devices that disconnect
    mid-training and never return.
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        dropout: int = 0,
        dropout_round: int = 20,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 1,
        min_available_clients: int = 1,
        fraction_evaluate: float = 1.0,
        min_evaluate_clients: int = 1,
    ):
        self.initial_parameters = initial_parameters
        self.dropout = dropout
        self.dropout_round = dropout_round
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.fraction_evaluate = fraction_evaluate
        self.min_evaluate_clients = min_evaluate_clients
        self.global_model = None

        # Persistent dropout tracking
        self.dropped_flower_cids = None
        self.num_available = 0

        # Client data shares (cid -> num_examples), collected during pre-dropout rounds
        self.client_data_shares: Dict[str, int] = {}

        # CSV logging
        self.csv_file = None
        self.csv_writer = None

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

        # Mid-experiment dropout: decide dropped set on dropout_round, persist forever
        if self.dropout > 0 and server_round >= self.dropout_round:
            if self.dropped_flower_cids is None:
                all_cids = [c.cid for c in clients]
                num_drop = min(self.dropout, len(clients) - 1)
                self.dropped_flower_cids = set(
                    random.sample(all_cids, num_drop)
                )

                # Compute data share from the actual dropped clients
                total_data = sum(self.client_data_shares.values()) or 1
                dropped_data = sum(
                    self.client_data_shares.get(c, 0)
                    for c in self.dropped_flower_cids
                )
                dropped_share = dropped_data / total_data
                print(f"\n  *** ROUND {server_round} — DROPPING {num_drop} client(s) ***")
                print(f"  Dropped CIDs: {self.dropped_flower_cids}")
                print(f"  Data lost: {dropped_data} of {total_data} samples "
                      f"({dropped_share:.1%})")
                print(f"  Remaining: {len(clients) - num_drop} of {len(clients)} clients")

            clients = [c for c in clients
                       if c.cid not in self.dropped_flower_cids]

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

        # Track per-client data shares for dropout reporting
        for client_proxy, r in results:
            self.client_data_shares[client_proxy.cid] = r.num_examples

        avg_weights = [
            sum(w[i] * n for w, n in zip(weights_list, num_examples)) / total
            for i in range(len(weights_list[0]))
        ]

        self.global_model = create_model()
        set_parameters(self.global_model, avg_weights)

        y_pred = self.global_model.predict(SERVER_X_TEST)
        global_acc = accuracy_score(SERVER_Y_TEST, y_pred)

        # Compute dropped data share (non-zero only after dropout)
        dropped_data_share = 0.0
        if self.dropped_flower_cids is not None and self.client_data_shares:
            total_data = sum(self.client_data_shares.values()) or 1
            dropped_data = sum(
                self.client_data_shares.get(c, 0) for c in self.dropped_flower_cids
            )
            dropped_data_share = dropped_data / total_data

        print(f"\nRound {server_round} — Global test accuracy: {global_acc:.4f}")
        print(f"    Participants: {len(results)} of {self.num_available} "
              f"| CIDs: {sorted(participating_cids)}")
        if dropped_data_share > 0:
            print(f"    Data lost to dropout: {dropped_data_share:.1%}")

        # CSV logging
        if self.csv_writer is None:
            self.csv_file = open(
                f"results_dropout_{self.dropout}.csv",
                "w", newline="",
            )
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "round", "dropout_count", "dropout_round", "global_accuracy",
                "num_participated", "num_available", "participating_cids",
                "dropped_data_share",
            ])
        self.csv_writer.writerow([
            server_round,
            self.dropout,
            self.dropout_round,
            f"{global_acc:.4f}",
            len(results),
            self.num_available,
            sorted(participating_cids),
            f"{dropped_data_share:.4f}",
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

        if self.dropped_flower_cids is not None:
            clients = [c for c in clients
                       if c.cid not in self.dropped_flower_cids]

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
        description="Federated learning server with mid-experiment dropout."
    )
    parser.add_argument(
        "--dropout", type=int, default=0,
        help="Number of clients to permanently drop (0, 2, 4, 6)",
    )
    parser.add_argument(
        "--dropout_round", type=int, default=20,
        help="Round at which to trigger dropout (default: 20)",
    )
    args = parser.parse_args()

    n_drop = args.dropout
    drop_round = args.dropout_round

    print("=" * 55)
    print(f"  DROPOUT EXPERIMENT — Flower + scikit-learn")
    print(f"  Clients: 8 total  |  Dropout: {n_drop} at round {drop_round}")
    print("=" * 55)
    print()
    print("  The server holds a Logistic Regression model.")
    print("  Clients each have a DIFFERENT slice of the digits dataset.")
    print("  Data NEVER leaves the clients — only weights are shared.")
    print()
    if n_drop > 0:
        print(f"  Rounds 1–{drop_round - 1}: all 8 clients participate")
        print(f"  Round {drop_round}:       {n_drop} client(s) permanently dropped")
        print(f"  Rounds {drop_round + 1}–40: {8 - n_drop} client(s) continue")
        print()
    print("  Open new terminals and run (all 8 before round 1):")
    print("    python3 client.py --cid 0 --num_clients 8")
    print("    python3 client.py --cid 1 --num_clients 8")
    print("    python3 client.py --cid 2 --num_clients 8")
    print("    python3 client.py --cid 3 --num_clients 8")
    print("    python3 client.py --cid 4 --num_clients 8")
    print("    python3 client.py --cid 5 --num_clients 8")
    print("    python3 client.py --cid 6 --num_clients 8")
    print("    python3 client.py --cid 7 --num_clients 8")
    print()
    print("  Ctrl+C any client to remove it; server keeps going.")
    print("  Ctrl+C the server to stop entirely.")
    print()

    initial_model = create_model()
    initial_weights = get_parameters(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    strategy = ExplicitFedAvg(
        initial_parameters=initial_parameters,
        dropout=n_drop,
        dropout_round=drop_round,
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
