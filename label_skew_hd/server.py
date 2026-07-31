import argparse
import csv
import logging
import os
import random
import signal
import time
import warnings

os.environ["GRPC_VERBOSITY"] = "ERROR"
warnings.filterwarnings("ignore")

import numpy as np
import flwr as fl
from flwr.server.strategy import Strategy
from flwr.common import FitIns, Parameters
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from sklearn.metrics import accuracy_score
from common import (
    create_model, get_parameters, set_parameters,
    get_initial_parameters, load_dataset, DATASET_INFO,
    train_centralized_baseline,
)

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)

NUM_ROUNDS = 150
PATIENCE = 20


class HDExperimentStrategy(Strategy):
    def __init__(self, initial_parameters, dataset, hd_val, seed, num_clients=10):
        self.initial_parameters = initial_parameters
        self.dataset = dataset
        self.hd_val = hd_val
        self.seed = seed
        self.num_clients = num_clients
        self.model = None
        self.rounds_file = None
        self.rounds_writer = None
        self._summary_written = False
        self.model_bytes = sum(p.nbytes for p in parameters_to_ndarrays(initial_parameters))
        self.cumulative_overhead = 0
        self.start_time = time.time()
        self.best_acc = 0.0
        self.no_improve = 0
        self.times_to_target = None

        _, _, X_test, y_test = load_dataset(dataset)
        self.server_X_test = X_test
        self.server_y_test = y_test

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        while client_manager.num_available() < self.num_clients:
            time.sleep(2)
            num_avail = client_manager.num_available()
            print(f"  Waiting for clients... {num_avail}/{self.num_clients}", flush=True)
        clients = client_manager.sample(
            num_clients=self.num_clients, min_num_clients=self.num_clients
        )
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

        self.model = create_model(self.dataset)
        set_parameters(self.model, avg_weights)

        y_pred = self.model.predict(self.server_X_test)
        global_acc = accuracy_score(self.server_y_test, y_pred)

        elapsed = time.time() - self.start_time

        sent_bytes = self.model_bytes * len(results)
        received_bytes = self.model_bytes * len(results)
        round_overhead = sent_bytes + received_bytes
        self.cumulative_overhead += round_overhead

        if global_acc > self.best_acc:
            self.best_acc = global_acc
            self.no_improve = 0
        else:
            self.no_improve += 1

        target = 0.9 * self.best_acc
        if self.times_to_target is None and global_acc >= target:
            self.times_to_target = elapsed

        self._write_round_row(server_round, global_acc, sent_bytes, received_bytes, elapsed)
        print(f"  {server_round:3d}/{NUM_ROUNDS}  acc={global_acc:.4f} best={self.best_acc:.4f}  "
              f"{elapsed:.0f}s  sent={sent_bytes/1e6:.1f}MB recv={received_bytes/1e6:.1f}MB  "
              f"total={self.cumulative_overhead/1e6:.1f}MB")

        if self.no_improve >= PATIENCE and server_round >= 30:
            self._write_summary(elapsed)
            os.kill(os.getpid(), signal.SIGINT)

        time.sleep(0.1)
        return ndarrays_to_parameters(avg_weights), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None

    def _write_round_row(self, rnd, acc, sent_bytes, received_bytes, elapsed):
        if self.rounds_writer is None:
            fname = f"results_{self.dataset}_hd{self.hd_val}_seed{self.seed}_rounds.csv"
            self.rounds_file = open(fname, "w", newline="")
            self.rounds_writer = csv.writer(self.rounds_file)
            self.rounds_writer.writerow([
                "round", "global_accuracy", "sent_bytes", "received_bytes",
                "cumulative_overhead_bytes", "elapsed_time_seconds",
            ])
        self.rounds_writer.writerow([
            rnd, f"{acc:.4f}", sent_bytes, received_bytes,
            self.cumulative_overhead, f"{elapsed:.2f}",
        ])
        self.rounds_file.flush()

    def _count_rounds(self):
        fname = f"results_{self.dataset}_hd{self.hd_val}_seed{self.seed}_rounds.csv"
        if not os.path.exists(fname):
            return 0
        with open(fname) as f:
            return sum(1 for _ in f) - 1

    def _write_summary(self, elapsed):
        if self._summary_written:
            return
        self._summary_written = True

        if self.rounds_file:
            self.rounds_file.close()

        target = 0.9 * self.best_acc
        fname = f"results_{self.dataset}_hd{self.hd_val}_seed{self.seed}_rounds.csv"
        rounds_to_target = None
        with open(fname, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if float(r["global_accuracy"]) >= target:
                    rounds_to_target = int(r["round"])
                    break

        final_round = self._count_rounds()
        fname_summary = f"results_{self.dataset}_hd{self.hd_val}_seed{self.seed}.csv"
        with open(fname_summary, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "dataset", "hd", "seed", "max_accuracy", "target_accuracy",
                "rounds_to_target", "time_to_target_seconds",
                "final_round", "cumulative_overhead_bytes", "elapsed_time_seconds",
            ])
            ttt = f"{self.times_to_target:.2f}" if self.times_to_target is not None else "NA"
            w.writerow([
                self.dataset, self.hd_val, self.seed,
                f"{self.best_acc:.4f}", f"{target:.4f}",
                rounds_to_target if rounds_to_target else "NA",
                ttt,
                final_round, self.cumulative_overhead, f"{elapsed:.2f}",
            ])

        print(f"\n  >>> {self.dataset} HD={self.hd_val} seed={self.seed}: "
              f"max={self.best_acc:.4f} target={target:.4f} "
              f"rounds_to_target={rounds_to_target} "
              f"time_to_target={self.times_to_target:.1f}s")

        self._write_centralized_accuracy()

    def _write_centralized_accuracy(self):
        try:
            cl_acc, _ = train_centralized_baseline(self.dataset)
            fname = f"results_{self.dataset}_hd{self.hd_val}_seed{self.seed}.csv"
            with open(fname, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([f"centralized_baseline_accuracy", f"{cl_acc:.4f}"])
        except Exception as e:
            print(f"  Could not compute centralized baseline: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['mnist', 'cifar10', 'cifar100'])
    parser.add_argument("--hd", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server_address", type=str, default="10.0.0.1:8080")
    parser.add_argument("--num_clients", type=int, default=10)
    args = parser.parse_args()

    print(f"\n=== {args.dataset} HD={args.hd} seed={args.seed} "
          f"on {args.server_address} ===")

    random.seed(args.seed)
    np.random.seed(args.seed)

    init_params = get_initial_parameters(args.dataset)
    init_params_nd = ndarrays_to_parameters(init_params)

    strategy = HDExperimentStrategy(
        initial_parameters=init_params_nd,
        dataset=args.dataset,
        hd_val=args.hd,
        seed=args.seed,
        num_clients=args.num_clients,
    )

    try:
        fl.server.start_server(
            server_address=args.server_address,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
        )
    except KeyboardInterrupt:
        pass

    strategy._write_summary(time.time() - strategy.start_time)


if __name__ == "__main__":
    main()
