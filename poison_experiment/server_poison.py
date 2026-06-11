"""
server_poison.py - Server for label-flipping poisoning experiments.

Logs results to results_poison_{scheme}.csv with per-round metrics including:
  - global accuracy and per-class accuracy
  - malicious vs honest local accuracy (for detectability)
  - malicious vs honest update norms (for norm-based detection)

Usage:
  python3 server_poison.py --scheme iid
  python3 server_poison.py --scheme single
  python3 server_poison.py --scheme multi
"""

import argparse
import csv
import json
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
from common import create_model, get_parameters, set_parameters, load_server_test_data

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)

SERVER_X_TEST, SERVER_Y_TEST = load_server_test_data()


class PoisonFedAvg(Strategy):

    def __init__(
        self,
        initial_parameters: Parameters,
        scheme: str = "iid",
        malicious_cid: int = 7,
        num_rounds: int = 40,
        seed: int = 42,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 1,
        min_available_clients: int = 1,
        fraction_evaluate: float = 1.0,
        min_evaluate_clients: int = 1,
    ):
        self.initial_parameters = initial_parameters
        self.scheme = scheme
        self.malicious_cid = str(malicious_cid)
        self.num_rounds = num_rounds
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.fraction_evaluate = fraction_evaluate
        self.min_evaluate_clients = min_evaluate_clients
        self.global_model = None
        self.num_available = 0

        # Randomly select ~half the rounds for poisoning
        rng = np.random.RandomState(seed=seed)
        n_poison = num_rounds // 2
        self.poison_rounds = set(
            rng.choice(range(1, num_rounds + 1), size=n_poison, replace=False)
        )
        print(f"  Poison rounds ({len(self.poison_rounds)} of {num_rounds}): "
              f"{sorted(self.poison_rounds)}")

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

        is_poison = server_round in self.poison_rounds
        fit_config = {
            "poison_round": is_poison,
            "poison_seed": server_round * 100 + 42,
        }

        return [(client, FitIns(parameters, fit_config)) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict]:
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

        # Per-class accuracy on the server test set
        per_class = {}
        for digit in range(10):
            mask = SERVER_Y_TEST == digit
            if mask.sum() > 0:
                per_class[str(digit)] = float(
                    accuracy_score(SERVER_Y_TEST[mask], y_pred[mask])
                )
            else:
                per_class[str(digit)] = 0.0

        # Collect malicious vs honest metrics from fit results
        malicious_local_acc = None
        honest_local_accs = []
        malicious_update_norm = None
        honest_update_norms = []

        for _, r in results:
            if r.metrics:
                cid = r.metrics.get("cid", "?")
                acc = r.metrics.get("accuracy", 0.0)
                norm = r.metrics.get("update_norm", 0.0)
                if str(cid) == self.malicious_cid:
                    malicious_local_acc = acc
                    malicious_update_norm = norm
                else:
                    honest_local_accs.append(acc)
                    honest_update_norms.append(norm)

        honest_avg_acc = np.mean(honest_local_accs) if honest_local_accs else 0.0
        honest_avg_norm = np.mean(honest_update_norms) if honest_update_norms else 0.0

        is_poison = server_round in self.poison_rounds

        print(f"\nRound {server_round} — Global acc: {global_acc:.4f}"
              f" {'[POISON]' if is_poison else ''}")
        print(f"    Participants: {len(results)} | CIDs: {sorted(participating_cids)}")
        if malicious_local_acc is not None:
            print(f"    Malicious local acc: {malicious_local_acc:.4f}"
                  f" | Honest avg: {honest_avg_acc:.4f}")
        if malicious_update_norm is not None:
            print(f"    Malicious norm: {malicious_update_norm:.4f}"
                  f" | Honest avg: {honest_avg_norm:.4f}")

        # CSV logging
        if self.csv_writer is None:
            self.csv_file = open(
                f"results_poison_{self.scheme}.csv",
                "w", newline="",
            )
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "round", "scheme", "global_accuracy", "per_class_accuracy",
                "is_poison_round", "malicious_cid",
                "malicious_local_acc", "honest_avg_local_acc",
                "malicious_update_norm", "honest_avg_update_norm",
            ])
        self.csv_writer.writerow([
            server_round,
            self.scheme,
            f"{global_acc:.4f}",
            json.dumps(per_class),
            int(is_poison),
            self.malicious_cid,
            f"{malicious_local_acc:.4f}" if malicious_local_acc is not None else "",
            f"{honest_avg_acc:.4f}" if honest_local_accs else "",
            f"{malicious_update_norm:.4f}" if malicious_update_norm is not None else "",
            f"{honest_avg_norm:.4f}" if honest_update_norms else "",
        ])
        self.csv_file.flush()

        time.sleep(1.5)

        return ndarrays_to_parameters(avg_weights), {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        n = max(
            ceil(self.fraction_evaluate * client_manager.num_available()),
            self.min_evaluate_clients,
        )
        while client_manager.num_available() < self.min_available_clients:
            time.sleep(2)
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
        description="Federated learning server for poisoning experiments."
    )
    parser.add_argument(
        "--scheme", type=str, default="iid",
        choices=["iid", "single", "multi"],
        help="Data split scheme for the poisoning experiment",
    )
    parser.add_argument("--malicious_cid", type=int, default=7)
    parser.add_argument("--num_rounds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scheme_descriptions = {
        "iid": "IID — all clients have balanced data from all 10 classes",
        "single": "Non-IID — malicious gets class 8, others get classes 0-6",
        "multi": "Non-IID — malicious gets classes 1,2,3, others get one each",
    }

    print("=" * 60)
    print(f"  POISON EXPERIMENT — Flower + scikit-learn")
    print(f"  Scheme: {scheme_descriptions[args.scheme]}")
    print(f"  Malicious cid={args.malicious_cid} | {args.num_rounds} rounds")
    print("=" * 60)
    print()

    initial_model = create_model()
    initial_weights = get_parameters(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    strategy = PoisonFedAvg(
        initial_parameters=initial_parameters,
        scheme=args.scheme,
        malicious_cid=args.malicious_cid,
        num_rounds=args.num_rounds,
        seed=args.seed,
        min_available_clients=8,
    )

    print("  Start 8 clients (all before round 1):")
    for i in range(8):
        mal = " --malicious" if i == args.malicious_cid else ""
        print(f"    python3 client_poison.py --cid {i} --scheme {args.scheme}{mal}")
    print()
    print("  Server listening on 127.0.0.1:8080")
    print()

    try:
        fl.server.start_server(
            server_address="127.0.0.1:8080",
            config=fl.server.ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
        )
    except KeyboardInterrupt:
        pass
    finally:
        strategy.close()


if __name__ == "__main__":
    main()
