import argparse
import csv
import json
import logging
import os
import time
import warnings

os.environ["GRPC_VERBOSITY"] = "ERROR"
warnings.filterwarnings("ignore")

from typing import List, Tuple, Optional, Dict

import flwr as fl
import numpy as np
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


class ModelDropoutFedAvg(Strategy):
    def __init__(
        self,
        initial_parameters: Parameters,
        dropout_rate: float = 0.0,
        num_clients: int = 8,
        min_available_clients: int = 8,
        mode: str = "model_dropout",
        clients_per_round: int = 8,
    ):
        self.initial_parameters = initial_parameters
        self.dropout_rate = dropout_rate
        self.num_clients = num_clients
        self.min_available_clients = min_available_clients
        self.mode = mode
        self.clients_per_round = clients_per_round
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

        if self.mode == "client_dropout":
            n = min(self.clients_per_round, num_available)
            clients = client_manager.sample(num_clients=n, min_num_clients=n)
            picked_cids = sorted(c.cid for c in clients)
            print(f"  Round {server_round}: client dropout — picked {len(picked_cids)}/{num_available} CIDs: {picked_cids}")
        else:
            clients = client_manager.sample(
                num_clients=num_available, min_num_clients=num_available
            )

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
        participating_cids = [r.metrics.get("cid", "?") for _, r in results if r.metrics]
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

        per_class_acc = {}
        for digit in range(10):
            mask = SERVER_Y_TEST == digit
            if np.sum(mask) > 0:
                per_class_acc[int(digit)] = float(accuracy_score(SERVER_Y_TEST[mask], y_pred[mask]))
            else:
                per_class_acc[int(digit)] = None

        total_params = 10 * 64 + 10
        if self.mode == "model_dropout":
            params_sent = [
                r.metrics.get("num_params_sent", total_params)
                for _, r in results if r.metrics
            ]
            comm_cost = sum(params_sent)
        else:
            comm_cost = len(results) * total_params

        print(f"\nRound {server_round} — Global test accuracy: {global_acc:.4f}")
        print(f"    Participants: {len(results)} | CIDs: {sorted(participating_cids)}")
        print(f"    Communication cost: {comm_cost}/{self.num_clients * total_params} params")

        if self.csv_writer is None:
            label = (
                f"model_dropout_{self.dropout_rate}"
                if self.mode == "model_dropout"
                else f"client_dropout_{self.clients_per_round}"
            )
            self.csv_file = open(f"results_{label}.csv", "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "round", "mode", "dropout_rate", "clients_per_round",
                "global_accuracy", "per_class_accuracy",
                "num_participated", "num_available", "comm_cost",
                "participating_cids",
            ])
        self.csv_writer.writerow([
            server_round,
            self.mode,
            self.dropout_rate,
            self.clients_per_round,
            f"{global_acc:.4f}",
            json.dumps(per_class_acc),
            len(results),
            self.min_available_clients,
            comm_cost,
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

        clients = client_manager.sample(
            num_clients=num_available, min_num_clients=self.min_available_clients
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

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict]]:
        return None

    def close(self):
        if self.csv_file is not None:
            self.csv_file.close()


def main():
    parser = argparse.ArgumentParser(
        description="FL server — model dropout & resource comparison."
    )
    parser.add_argument(
        "--mode", type=str, default="model_dropout",
        choices=["model_dropout", "client_dropout"],
        help="mode: model_dropout or client_dropout",
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.0,
        help="Fraction of weights to drop per client (0.0–1.0, model_dropout only)",
    )
    parser.add_argument(
        "--clients_per_round", type=int, default=8,
        help="Clients sampled per round (client_dropout only)",
    )
    parser.add_argument(
        "--num_clients", type=int, default=8,
        help="Total number of clients",
    )
    args = parser.parse_args()

    print("=" * 55)
    if args.mode == "model_dropout":
        print(f"  MODEL DROPOUT EXPERIMENT")
        print(f"  {args.num_clients} clients  |  dropout_rate={args.dropout_rate}")
    else:
        print(f"  CLIENT DROPOUT EXPERIMENT")
        print(f"  {args.num_clients} total clients  |  {args.clients_per_round} per round")
    print("=" * 55)
    print()
    print("  Open new terminals and run:")
    for cid in range(args.num_clients):
        dr = args.dropout_rate if args.mode == "model_dropout" else 0.0
        print(f"    python3 client.py --cid {cid} --num_clients {args.num_clients} --dropout_rate {dr}")
    print()

    initial_model = create_model()
    initial_weights = get_parameters(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    strategy = ModelDropoutFedAvg(
        initial_parameters=initial_parameters,
        dropout_rate=args.dropout_rate,
        num_clients=args.num_clients,
        min_available_clients=args.num_clients,
        mode=args.mode,
        clients_per_round=args.clients_per_round,
    )

    print(f"  Server listening on 127.0.0.1:8080")
    print(f"  Waiting for {args.num_clients} clients...")
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
