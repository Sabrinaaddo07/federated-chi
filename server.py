"""
server.py - The central orchestrator in the federated learning system.

WHAT THIS DOES:
1. Initialises the model with zero / random weights.
2. Waits for at least 1 client to connect.
3. For each round:
   a. Sends the current weights TO every client.
   b. Each client trains on its own private data.
   c. Receives updated weights BACK from every client.
   d. Averages the weights together (FedAvg = Federated Averaging).
4. Evaluates the global model on a held-out test set and reports
   per-client accuracy. Then pauses 10 seconds before the next round.

Run with Ctrl+C to stop.
"""

import logging
import os
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

    Every method is written out explicitly so you can see exactly what
    happens — nothing is hidden inside a parent class.
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 1,
        min_available_clients: int = 1,
        fraction_evaluate: float = 1.0,
        min_evaluate_clients: int = 1,
    ):
        self.initial_parameters = initial_parameters
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.fraction_evaluate = fraction_evaluate
        self.min_evaluate_clients = min_evaluate_clients
        self.global_model = None

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
        Decide which clients train this round.

        Here: a fraction of available clients (fraction_fit = 1.0 means ALL).
        If fewer than min_fit_clients are connected, Flower waits and retries.
        """
        n = max(
            ceil(self.fraction_fit * client_manager.num_available()),
            self.min_fit_clients,
        )
        clients = client_manager.sample(
            num_clients=n, min_num_clients=self.min_fit_clients
        )
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
        print(f"\nRound {server_round} — Global test accuracy: {global_acc:.4f}")

        # Pause so you can add/remove clients between rounds
        print("  (pausing 10 seconds before next round...)")
        time.sleep(10)

        return ndarrays_to_parameters(avg_weights), {}

    # ------------------------------------------------------------------
    #  Step 3: Pick which clients evaluate this round
    # ------------------------------------------------------------------

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Decide which clients evaluate this round.

        Here: a fraction of available clients (fraction_evaluate = 1.0 = ALL).
        """
        n = max(
            ceil(self.fraction_evaluate * client_manager.num_available()),
            self.min_evaluate_clients,
        )
        clients = client_manager.sample(
            num_clients=n, min_num_clients=self.min_evaluate_clients
        )
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
        """
        Server-side evaluation. We skip this — we evaluate in aggregate_fit
        so that accuracy is printed after clients report their results.
        """
        return None


def main():
    print("=" * 55)
    print("  FEDERATED LEARNING DEMO — Flower + scikit-learn")
    print("=" * 55)
    print()
    print("  The server holds a Logistic Regression model.")
    print("  Clients each have a DIFFERENT slice of the digits dataset.")
    print("  Data NEVER leaves the clients — only weights are shared.")
    print("  The server averages (FedAvg) the weights each round.")
    print()
    print("  Open new terminals and run:")
    print("    python3 client.py --cid 0")
    print("    python3 client.py --cid 1")
    print("    ... (up to --cid 9)")
    print()
    print("  Training starts as soon as the first client connects.")
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
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
        fraction_evaluate=1.0,
        min_evaluate_clients=1,
    )

    # --- Step 3: Start the server ---
    print("  Server listening on 127.0.0.1:8080")
    print("  Waiting for at least 1 client...")
    print()

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=99999),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
