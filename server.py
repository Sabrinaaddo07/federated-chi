"""
server.py - The central orchestrator in the federated learning system.

Supports persistent client dropout experiments:
  --dropout 0    Everyone participates every round (baseline)
  --dropout 2    Permanently drop 2 of 8 clients
  --dropout 4    Permanently drop 4 of 8 clients
  --dropout 6    Permanently drop 6 of 8 clients

Results are logged to results_dropout_X.csv
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

    Supports persistent dropout: a fraction of clients are selected on
    round 1 and never participate again, simulating permanently dropped
    devices.
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        dropout: int = 0,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 1,
        min_available_clients: int = 1,
        fraction_evaluate: float = 1.0,
        min_evaluate_clients: int = 1,
    ):
        self.initial_parameters = initial_parameters
        self.dropout = dropout
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.fraction_evaluate = fraction_evaluate
        self.min_evaluate_clients = min_evaluate_clients
        self.global_model = None

        # Persistent dropout tracking
        self.dropped_flower_cids = None  # set of Flower internal CIDs to skip
        self.num_available = 0          # updated each round for reporting

        # CSV logging
        self.csv_file = None
        self.csv_writer = None

    # ------------------------------------------------------------------
    #  Step 0: Give the server the initial model weights
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager):
        """Return the zero-initialised weights to start training from."""
        return self.initial_parameters

    # ------------------------------------------------------------------
    #  Step 1: Pick which clients will train this round
    # ------------------------------------------------------------------

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Sample available clients, then optionally filter out dropped ones.
        The dropped set is decided on round 1 and kept forever.
        """
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

        if self.dropout > 0:
            if self.dropped_flower_cids is None:
                all_cids = [c.cid for c in clients]
                num_drop = min(self.dropout, len(clients) - 1)
                self.dropped_flower_cids = set(
                    random.sample(all_cids, num_drop)
                )
                print(f"\n  Persistent dropout: keeping "
                      f"{len(clients) - num_drop} of {len(clients)} clients")
                print(f"  Dropped Flower CIDs: {self.dropped_flower_cids}")

            clients = [c for c in clients
                       if c.cid not in self.dropped_flower_cids]

        return [(client, FitIns(parameters, {})) for client in clients]

    # ------------------------------------------------------------------
    #  Step 2: Average the weight updates from every client
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict]:
        """
        Combine all client weight updates into one global model.

        This IS Federated Averaging:
          1. Convert each client's Parameters → list of numpy arrays
          2. Weighted average: client with more data gets more influence
          3. Convert back to Parameters and return
        """
        if not results:
            return None, {}

        # Unpack results
        weights_list = [parameters_to_ndarrays(r.parameters) for _, r in results]
        participating_cids = [r.metrics.get("cid", "?") for _, r in results
                              if r.metrics]
        num_examples = [r.num_examples for _, r in results]
        total = sum(num_examples)

        # Weighted average: for each layer (weights, biases), do:
        #   avg = sum(client_weight * client_num_examples) / total_examples
        avg_weights = [
            sum(w[i] * n for w, n in zip(weights_list, num_examples)) / total
            for i in range(len(weights_list[0]))
        ]

        # Save in an sklearn model for server-side evaluation
        self.global_model = create_model()
        set_parameters(self.global_model, avg_weights)

        # Evaluate on the server's held-out test set
        y_pred = self.global_model.predict(SERVER_X_TEST)
        global_acc = accuracy_score(SERVER_Y_TEST, y_pred)

        # --- Reporting ---
        print(f"\nRound {server_round} — Global test accuracy: {global_acc:.4f}")
        print(f"    Participants: {len(results)} of {self.num_available} "
              f"| CIDs: {sorted(participating_cids)}")

        # --- CSV logging ---
        if self.csv_writer is None:
            self.csv_file = open(
                f"results_dropout_{self.dropout}.csv",
                "w", newline="",
            )
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "round", "dropout_count", "global_accuracy",
                "num_participated", "num_available", "participating_cids",
            ])
        self.csv_writer.writerow([
            server_round,
            self.dropout,
            f"{global_acc:.4f}",
            len(results),
            self.num_available,
            sorted(participating_cids),
        ])
        self.csv_file.flush()

        # Pause between rounds
        print("  (pausing 3 seconds before next round...)")
        time.sleep(3)

        return ndarrays_to_parameters(avg_weights), {}

    # ------------------------------------------------------------------
    #  Step 3: Pick which clients evaluate this round
    # ------------------------------------------------------------------

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Decide which clients evaluate this round.
        Apply the same dropout filter as configure_fit.
        """
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

        # Apply the same persistent dropout filter
        if self.dropped_flower_cids is not None:
            clients = [c for c in clients
                       if c.cid not in self.dropped_flower_cids]

        return [(client, EvaluateIns(parameters, {})) for client in clients]

    # ------------------------------------------------------------------
    #  Step 4: Collect and report each client's local accuracy
    # ------------------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes]],
    ) -> Tuple[Optional[float], Dict]:
        """
        Print each client's accuracy on its own local test set.
        These are different test samples for each client.
        """
        for _, r in results:
            if r.metrics:
                cid = r.metrics.get("cid", "?")
                acc = r.metrics["accuracy"]
                print(f"    Client {cid} local accuracy: {acc:.4f}")
        return None, {}

    # ------------------------------------------------------------------
    #  Step 5: Optional server-side evaluation (not used here)
    # ------------------------------------------------------------------

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict]]:
        """Server-side evaluation. We skip this."""
        return None

    def close(self):
        """Close the CSV file if open."""
        if self.csv_file is not None:
            self.csv_file.close()


def main():
    parser = argparse.ArgumentParser(
        description="Federated learning server with persistent dropout."
    )
    parser.add_argument(
        "--dropout", type=int, default=0,
        help="Number of clients to permanently drop (0, 2, 4, 6)",
    )
    args = parser.parse_args()

    n_drop = args.dropout
    print("=" * 55)
    print(f"  DROPOUT EXPERIMENT — Flower + scikit-learn")
    print(f"  Persistent dropout: {n_drop} of 8 clients")
    print("=" * 55)
    print()
    print("  The server holds a Logistic Regression model.")
    print("  Clients each have a DIFFERENT slice of the digits dataset.")
    print("  Data NEVER leaves the clients — only weights are shared.")
    print()
    if n_drop > 0:
        print(f"  {n_drop} client(s) will be permanently dropped on round 1.")
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

    # --- Step 1: Create initial model with zero weights ---
    initial_model = create_model()
    initial_weights = get_parameters(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    # --- Step 2: Configure the strategy ---
    strategy = ExplicitFedAvg(
        initial_parameters=initial_parameters,
        dropout=n_drop,
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=8,
        fraction_evaluate=1.0,
        min_evaluate_clients=1,
    )

    # --- Step 3: Start the server ---
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
