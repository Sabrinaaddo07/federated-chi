"""
server.py - The central orchestrator in the federated learning system.

WHAT THIS DOES:
1. Initialises the model with zero / random weights.
2. Waits for clients to connect.
3. For each round:
   a. Sends the current weights TO every client.
   b. Each client trains on its own private data.
   c. Receives updated weights BACK from every client.
   d. Averages the weights together (FedAvg = Federated Averaging).
4. After all rounds, evaluates the final model.

KEY FEDERATED LEARNING CONCEPT:
The server never sees any client's data. It only sees weights.
The server's job is to coordinate and average, not to train.
"""

import logging
import os
import warnings

# Suppress Flower's verbose logs
os.environ["GRPC_VERBOSITY"] = "ERROR"
warnings.filterwarnings("ignore")

import flwr as fl
from flwr.server.strategy import FedAvg
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import numpy as np
from common import create_model, get_parameters, set_parameters

# After import, suppress Flower's logger
flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)


class FedAvgWithHistory(FedAvg):
    """
    FedAvg (Federated Averaging) is the standard FL aggregation algorithm.

    HOW IT WORKS:
    1. Wait for ALL clients to send their updated weights.
    2. Compute a weighted average: client with more data gets more influence.
    3. The averaged weights become the new "global model."

    We extend FedAvg slightly to save the model after every round
    so we can evaluate it at the end.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_model = None   # will hold the aggregated model

    def aggregate_fit(self, rnd, results, failures):
        """
        Called by Flower after clients finish training.
        - results: list of (Client, FitRes) pairs from each client.
        - failures: any clients that crashed this round.

        The parent class (FedAvg) does the actual weighted averaging.
        We just save a copy of the result.
        """
        # Let FedAvg do the aggregation
        aggregated_params, metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_params is not None:
            # Convert Flower's special Parameter format → numpy arrays
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_params)

            # Store in an sklearn model so we can evaluate it later
            self.global_model = create_model()
            set_parameters(self.global_model, ndarrays)

        return aggregated_params, metrics

    def aggregate_evaluate(self, rnd, results, failures):
        """
        Called by Flower after clients finish evaluating.
        We print the average accuracy here so we can watch progress.
        """
        aggregated_loss, metrics = super().aggregate_evaluate(
            rnd, results, failures
        )

        # Collect accuracy from each client and average it
        accuracies = [r.metrics["accuracy"] for _, r in results if r.metrics]
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            print(f"        Average accuracy this round: {avg_accuracy:.4f}")

        return aggregated_loss, metrics


def main():
    print("=" * 55)
    print("  FEDERATED LEARNING DEMO — Flower + scikit-learn")
    print("=" * 55)
    print()
    print("  What is happening?")
    print("  - The server holds a Logistic Regression model.")
    print("  - 2 clients each have a DIFFERENT slice of the digits dataset.")
    print("  - Data NEVER leaves the clients — only weights are shared.")
    print("  - The server averages (FedAvg) the weights each round.")
    print()

    # --- Step 1: Create initial model with zero weights ---
    # This is the starting point. Every round will improve on this.
    initial_model = create_model()
    initial_weights = get_parameters(initial_model)
    initial_parameters = fl.common.ndarrays_to_parameters(initial_weights)

    # --- Step 2: Configure the strategy ---
    # fraction_fit=1.0        → use 100 % of connected clients each round
    # min_fit_clients=2       → wait until BOTH clients are ready
    # min_available_clients=2 → don't start until both are connected
    strategy = FedAvgWithHistory(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        initial_parameters=initial_parameters,
    )

    # --- Step 3: Start the server ---
    # The server listens on port 8080 for gRPC connections.
    # num_rounds=5 means: train → aggregate → repeat 5 times.
    print("  Server listening on 127.0.0.1:8080")
    print("  Waiting for 2 clients to connect...")
    print()

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    # --- Step 4: Final evaluation ---
    print()
    print("=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    print()
    print("  Evaluating the final global model on all test data...")

    # Load the full dataset and split into train/test
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Score the final model
    final_model = strategy.global_model
    if final_model is not None:
        y_pred = final_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Final test accuracy: {acc:.4f} ({acc*100:.1f}%)")
    else:
        print("  No model was saved — did clients connect properly?")

    print()
    print("  Demonstration complete!")
    print("  Notice: the server never saw any raw pixel data —")
    print("  only the aggregated weights from 2 clients.")
    print()


if __name__ == "__main__":
    main()
