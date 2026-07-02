#!/usr/bin/env python3
"""
Federated Learning Server — Quick Draw Sketch Recognition Lab
=============================================================
Instructor-facing script. Handles three things:

  1. Pre-train the global model on a large Quick Draw slice and serve the
     weights file over HTTP so students can download it.
  2. Run FL rounds for Setup 2 (min 1 client — only Student A participates).
  3. Run FL rounds for Setup 3 (min 2 clients — both students participate).

Typical workflow
----------------
  # Step 1: do this once before the lab, distribute global_model.npy
  python server.py --pretrain

  # Step 2: Setup 2 — only Student A connects
  python server.py --min-clients 1 --rounds 3

  # Step 3: Setup 3 — both students connect (restart server between setups)
  python server.py --min-clients 2 --rounds 3

  # Optional flags
  --address  0.0.0.0:8080    Flower gRPC address  (default: 0.0.0.0:8080)
  --http-port 8081            File server port     (default: 8081)
"""

import argparse
import os
import sys
import threading
import warnings
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Dict, List, Optional, Tuple, Union
import urllib.request

import numpy as np
from sklearn.linear_model import LogisticRegression

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Scalar
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────

CLASSES = ["cat", "dog", "sun", "clock", "mountain", "tent", "tree", "bird", "star", "face"]
N_CLASSES = len(CLASSES)
N_FEATURES = 784  # 28 × 28

DATA_DIR = "quickdraw_data"
GLOBAL_MODEL_PATH = "global_model.npy"
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

# Instructor's pretraining slice — kept well above the student slices (0–399)
# so there is no overlap with any client's local data.
PRETRAIN_START = 1000
PRETRAIN_COUNT = 500  # samples per class


# ─── Data utilities ───────────────────────────────────────────────────────────

def download_class(cls: str) -> str:
    """Download one Quick Draw .npy file if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{cls}.npy")
    if not os.path.exists(path):
        url = f"{BASE_URL}{cls}.npy"
        print(f"  Downloading {cls} ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, path)
        print("✓")
    return path


def load_dataset(start: int, count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load `count` samples per class starting at index `start`.
    Returns (X, y) with X normalised to [0, 1] and y as integer class labels.
    """
    X_list, y_list = [], []
    for i, cls in enumerate(CLASSES):
        path = download_class(cls)
        raw = np.load(path, mmap_mode="r")
        chunk = raw[start: start + count].copy()
        X_list.append(chunk)
        y_list.append(np.full(count, i, dtype=np.int64))

    X = np.vstack(X_list).astype(np.float32) / 255.0
    y = np.concatenate(y_list)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ─── Pre-training ─────────────────────────────────────────────────────────────

def pretrain_global_model() -> None:
    """Train a logistic regression model on a large Quick Draw slice and save
    the weights to disk so they can be distributed to students."""
    print("\n── Pre-training Global Model " + "─" * 40)
    print(f"Classes : {CLASSES}")
    print(f"Samples : {PRETRAIN_COUNT}/class  (indices {PRETRAIN_START}–"
          f"{PRETRAIN_START + PRETRAIN_COUNT - 1}, no overlap with student slices)\n")

    X, y = load_dataset(PRETRAIN_START, PRETRAIN_COUNT)

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
        C=1.0,
        random_state=42,
        n_jobs=-1,
    )
    print("Training …", end=" ", flush=True)
    model.fit(X, y)
    train_acc = model.score(X, y)
    print(f"done.  Training accuracy: {train_acc:.1%}\n")

    weights = {"coef": model.coef_, "intercept": model.intercept_}
    np.save(GLOBAL_MODEL_PATH, weights, allow_pickle=True)
    print(f"✓ Weights saved to '{GLOBAL_MODEL_PATH}'")
    print("  Distribute this file to students OR they will download it via the")
    print("  built-in HTTP file server when you run the FL server.\n")


# ─── Flower strategy ──────────────────────────────────────────────────────────

class SaveOnAggregateStrategy(FedAvg):
    """FedAvg that saves aggregated weights to disk after every round.

    Students download these files from the built-in HTTP file server.
    File names: aggregated_round_1.npy, aggregated_round_2.npy, …
    The last file always corresponds to the final FL model.
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:

        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_params is not None:
            arrays = parameters_to_ndarrays(aggregated_params)
            path = f"aggregated_round_{server_round}.npy"
            np.save(
                path,
                {"coef": arrays[0], "intercept": arrays[1]},
                allow_pickle=True,
            )
            n_clients = len(results)
            print(
                f"\n  ✓ Round {server_round}: aggregated {n_clients} client(s) → saved to '{path}'"
            )

        return aggregated_params, metrics


# ─── HTTP file server ─────────────────────────────────────────────────────────

class _QuietHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with suppressed access logs."""
    def log_message(self, format, *args):  # noqa: A002
        pass


def start_file_server(port: int) -> None:
    """Start a simple HTTP file server in a daemon thread.
    Students use this to download global_model.npy and aggregated weights.
    """
    httpd = HTTPServer(("0.0.0.0", port), _QuietHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print(f"  File server  : http://<your-ip>:{port}/")
    print(f"    • global_model.npy")
    print(f"    • aggregated_round_N.npy  (appears after each FL round)")


# ─── FL server ────────────────────────────────────────────────────────────────

def get_initial_parameters() -> fl.common.Parameters:
    if not os.path.exists(GLOBAL_MODEL_PATH):
        print(f"\n✗ '{GLOBAL_MODEL_PATH}' not found.")
        print("  Run:  python server.py --pretrain  first.\n")
        sys.exit(1)
    w = np.load(GLOBAL_MODEL_PATH, allow_pickle=True).item()
    return ndarrays_to_parameters([w["coef"], w["intercept"]])


def run_server(rounds: int, min_clients: int, address: str, http_port: int) -> None:
    print("\n── Federated Learning Server " + "─" * 42)
    print(f"  Flower gRPC  : {address}")
    print(f"  Rounds       : {rounds}")
    print(f"  Min clients  : {min_clients}")
    print()

    start_file_server(http_port)

    strategy = SaveOnAggregateStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=get_initial_parameters(),
    )

    print("\nWaiting for clients to connect …\n")

    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

    print("\n── All rounds complete. ──────────────────────────────────────────")
    print("  Students can now download the final aggregated model:")
    print(f"    aggregated_round_{rounds}.npy\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FL Sketch Lab — Instructor Server",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Download Quick Draw data and pre-train the global model.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        metavar="N",
        help="Number of FL rounds (default: 3).",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=2,
        metavar="N",
        help="Minimum clients before a round starts.\n"
             "  1 → Setup 2 (only Student A)\n"
             "  2 → Setup 3 (both students)  [default]",
    )
    parser.add_argument(
        "--address",
        type=str,
        default="0.0.0.0:8080",
        metavar="HOST:PORT",
        help="Flower gRPC bind address (default: 0.0.0.0:8080).",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8081,
        metavar="PORT",
        help="HTTP file server port for model distribution (default: 8081).",
    )

    args = parser.parse_args()

    if args.pretrain:
        pretrain_global_model()
    else:
        run_server(
            rounds=args.rounds,
            min_clients=args.min_clients,
            address=args.address,
            http_port=args.http_port,
        )


if __name__ == "__main__":
    main()
