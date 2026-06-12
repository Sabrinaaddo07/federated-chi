"""
server.py - Random per-round client sampling experiment.

Each round picks a random K of 10 clients to participate.
The selection is independent every round (no persistent dropout).

  --clients_per_round 2    Randomly pick 2 of 10 clients each round
  --clients_per_round 4    Randomly pick 4 of 10 clients each round
  --clients_per_round 6    Randomly pick 6 of 10 clients each round

Results are logged to results_sample_N.csv
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


class RandomSampleFedAvg(Strategy):
    """
    Each round: pick a random subset of K clients from the available pool.
    Selection is independent every round — no persistent state.
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        clients_per_round: int = 10,
        min_available_clients: int = 10,
    ):
        self.initial_parameters = initial_parameters
        self.clients_per_round = clients_per_round
        self.min_available_clients = min_available_clients
        self.global_model = None

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

        n = min(self.clients_per_round, num_available)
        clients = client_manager.sample(
            num_clients=n, min_num_clients=n
        )
        picked_cids = sorted(c.cid for c in clients)
        print(f"  Round {server_round}: picked {len(picked_cids)}/{num_available} "
              f"clients — CIDs: {picked_cids}")

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
        print(f"    Participants: {len(results)} | CIDs: {sorted(participating_cids)}")

        if self.csv_writer is None:
            self.csv_file = open(
                f"results_sample_{self.clients_per_round}.csv",
                "w", newline="",
            )
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "round", "clients_per_round", "global_accuracy",
                "num_participated", "total_clients", "participating_cids",
            ])
        self.csv_writer.writerow([
            server_round,
            self.clients_per_round,
            f"{global_acc:.4f}",
            len(results),
            self.min_available_clients,
            sorted(participating_cids),
        ])
        self.csv_file.flush()

        print("  (pausing 1.5 seconds before next round...)")
        time.sleep(1.5)

        return ndarrays_to_parameters(avg_weights), {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        num_available = client_manager.num_available()
        while num_available < self.min_available_clients:
            time.sleep(2)
            num_available = client_manager.num_available()

        n = min(self.clients_per_round, num_available)
        clients = client_manager.sample(
            num_clients=n, min_num_clients=n
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
        description="FL server — random per-round client sampling."
    )
    parser.add_argument(
        "--clients_per_round", type=int, default=10,
        help="Number of clients to randomly pick each round (2, 4, 6, or 10)",
    )
    parser.add_argument(
        "--num_clients", type=int, default=10,
        help="Total number of clients in the system (default: 10)",
    )
    args = parser.parse_args()

    k = args.clients_per_round
    total = args.num_clients

    print("=" * 55)
    print(f"  RANDOM SAMPLING EXPERIMENT")
    print(f"  {total} total clients  |  {k} picked per round")
    print("=" * 55)
    print()
    print("  Each round: a random subset of K clients participates.")
    print("  Selection is independent every round.")
    print()
    print("  Open new terminals and run (all 10 before round 1):")
    for cid in range(total):
        print(f"    python3 client.py --cid {cid} --num_clients {total}")
    print()
    print("  Ctrl+C any client to remove it; server keeps going.")
    print("  Ctrl+C the server to stop entirely.")
    print()

    initial_model = create_model()
    initial_weights = get_parameters(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    strategy = RandomSampleFedAvg(
        initial_parameters=initial_parameters,
        clients_per_round=k,
        min_available_clients=total,
    )

    print(f"  Server listening on 127.0.0.1:8080")
    print(f"  Waiting for {total} clients...")
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
