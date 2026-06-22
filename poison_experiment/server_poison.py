"""
server_poison.py - Server for label-flipping poisoning experiments.

Usage:
  python3 server_poison.py --baseline
  python3 server_poison.py --mode uniform
  python3 server_poison.py --mode early_only
  python3 server_poison.py --mode late_only
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
        mode: str = "baseline",
        num_malicious: int = 3,
        num_rounds: int = 40,
        seed: int = 42,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 1,
        min_available_clients: int = 1,
        fraction_evaluate: float = 1.0,
        min_evaluate_clients: int = 1,
    ):
        self.initial_parameters = initial_parameters
        self.mode = mode
        self.num_rounds = num_rounds
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.fraction_evaluate = fraction_evaluate
        self.min_evaluate_clients = min_evaluate_clients
        self.global_model = None
        self.num_available = 0

        # Malicious CIDs: last N clients (e.g. 5,6,7 for num_malicious=3)
        self.malicious_cids = set(
            str(c) for c in range(8 - num_malicious, 8)
        )

        rng = np.random.RandomState(seed=seed)

        if mode == "baseline":
            self.poison_rounds = set()
        elif mode == "uniform":
            n_poison = num_rounds // 2
            self.poison_rounds = set(
                rng.choice(range(1, num_rounds + 1), size=n_poison, replace=False)
            )
        elif mode == "early_only":
            first_half = list(range(1, 21))
            n_poison = 10
            self.poison_rounds = set(
                rng.choice(first_half, size=n_poison, replace=False)
            )
        elif mode == "late_only":
            second_half = list(range(21, 41))
            n_poison = 10
            self.poison_rounds = set(
                rng.choice(second_half, size=n_poison, replace=False)
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        n_poison = len(self.poison_rounds)
        if n_poison > 0:
            print(f"  Poison rounds ({n_poison} of {num_rounds}): "
                  f"{sorted(self.poison_rounds)}")
        else:
            print("  Baseline — no poisoning")

        print(f"  Malicious CIDs: {sorted(self.malicious_cids)}")

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

        # Collect metrics: average across all malicious, then all honest
        malicious_accs = []
        malicious_norms = []
        honest_accs = []
        honest_norms = []

        for _, r in results:
            if r.metrics:
                cid = r.metrics.get("cid", "?")
                acc = r.metrics.get("accuracy", 0.0)
                norm = r.metrics.get("update_norm", 0.0)
                if str(cid) in self.malicious_cids:
                    malicious_accs.append(acc)
                    malicious_norms.append(norm)
                else:
                    honest_accs.append(acc)
                    honest_norms.append(norm)

        avg_malicious_acc = np.mean(malicious_accs) if malicious_accs else 0.0
        avg_malicious_norm = np.mean(malicious_norms) if malicious_norms else 0.0
        avg_honest_acc = np.mean(honest_accs) if honest_accs else 0.0
        avg_honest_norm = np.mean(honest_norms) if honest_norms else 0.0

        is_poison = server_round in self.poison_rounds

        print(f"\nRound {server_round} — Global acc: {global_acc:.4f}"
              f" {'[POISON]' if is_poison else ''}")
        print(f"    Participants: {len(results)} | CIDs: {sorted(participating_cids)}")
        print(f"    Avg malicious acc: {avg_malicious_acc:.4f} (n={len(malicious_accs)})"
              f" | Avg honest acc: {avg_honest_acc:.4f} (n={len(honest_accs)})")
        print(f"    Avg malicious norm: {avg_malicious_norm:.4f}"
              f" | Avg honest norm: {avg_honest_norm:.4f}")

        # CSV logging
        if self.csv_writer is None:
            self.csv_file = open(
                f"results_poison_{self.mode}.csv",
                "w", newline="",
            )
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "round", "mode", "global_accuracy", "per_class_accuracy",
                "is_poison_round", "num_malicious",
                "avg_malicious_acc", "avg_honest_acc",
                "avg_malicious_norm", "avg_honest_norm",
            ])
        self.csv_writer.writerow([
            server_round,
            self.mode,
            f"{global_acc:.4f}",
            json.dumps(per_class),
            int(is_poison),
            len(self.malicious_cids),
            f"{avg_malicious_acc:.4f}" if malicious_accs else "",
            f"{avg_honest_acc:.4f}" if honest_accs else "",
            f"{avg_malicious_norm:.4f}" if malicious_norms else "",
            f"{avg_honest_norm:.4f}" if honest_norms else "",
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
        "--mode", type=str, default=None,
        choices=["uniform", "early_only", "late_only"],
        help="Poison schedule mode",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Run clean baseline (no poisoning)",
    )
    parser.add_argument("--num_malicious", type=int, default=3,
                        help="Number of malicious clients (last N CIDs)")
    parser.add_argument("--num_rounds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.baseline and args.mode:
        parser.error("Cannot use both --baseline and --mode")
    if not args.baseline and not args.mode:
        parser.error("Must specify either --baseline or --mode")

    mode = "baseline" if args.baseline else args.mode

    mode_descriptions = {
        "baseline": "Baseline — no malicious clients",
        "uniform": "3 malicious clients poison ~20 coordinated rounds across all 40",
        "early_only": "3 malicious clients poison ~10 rounds in 1-20, then honest in 21-40",
        "late_only": "3 malicious clients honest in 1-20, then poison ~10 rounds in 21-40",
    }

    malicious_cids_str = ",".join(str(c) for c in range(8 - args.num_malicious, 8))

    print("=" * 60)
    print(f"  POISON EXPERIMENT — Flower + scikit-learn")
    print(f"  Mode: {mode_descriptions[mode]}")
    print(f"  Malicious CIDs: {{{malicious_cids_str}}} | {args.num_rounds} rounds")
    print("=" * 60)
    print()

    initial_model = create_model()
    initial_weights = get_parameters(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    strategy = PoisonFedAvg(
        initial_parameters=initial_parameters,
        mode=mode,
        num_malicious=args.num_malicious,
        num_rounds=args.num_rounds,
        seed=args.seed,
        min_available_clients=8,
    )

    print("  Start 8 clients (all before round 1):")
    for i in range(8):
        if args.baseline:
            mal = ""
        else:
            is_mal = i >= 8 - args.num_malicious
            mal = " --malicious" if is_mal else ""
        print(f"    python3 client_poison.py --cid {i}{mal}")
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
