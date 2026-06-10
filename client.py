"""
client.py - One participant in the federated learning system.

WHAT THIS DOES:
1. Connects to the central server.
2. Receives the current global model weights.
3. Trains the model on its *own private data* (nobody else sees this data).
4. Sends the updated weights back to the server.
5. Repeats for every round.

KEY FEDERATED LEARNING CONCEPT:
The client NEVER shares its raw data. Only model weights leave this process.
"""

import argparse
import logging
import os
import warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
warnings.filterwarnings("ignore")

import flwr as fl
from common import create_model, get_parameters, set_parameters, load_client_data

# After import, suppress Flower's logger
flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)


class DigitClient(fl.client.NumPyClient):
    """
    A Flower client for the digits dataset.

    NumPyClient is the simplest way to make a Flower client.
    You only need to define 3 methods:
      get_parameters()   — what weights does your model have right now?
      set_parameters()   — overwrite your weights with the server's weights
      fit()              — train on your local data for 1 epoch

    And 1 optional method:
      evaluate()         — how accurate are you on your local test set?
    """

    def __init__(self, cid, num_clients):
        # Load *only* this client's slice of the data.
        self.cid = cid
        self.X_train, self.X_test, self.y_train, self.y_test = \
            load_client_data(cid, num_clients=num_clients)

        self.model = create_model()

        print(f"  Client {cid} loaded {len(self.X_train)} training + "
              f"{len(self.X_test)} test samples")
        print(f"  This client's data is PRIVATE — never sent to the server\n")

    # ------------------------------------------------------------------
    #  get_parameters() and set_parameters() are how weights travel
    #  between the server and this client.
    #
    #  Think of them as packing / unpacking a suitcase:
    #    - get: pack the weights into a list of numpy arrays
    #    - set: unpack and load them into the model
    # ------------------------------------------------------------------

    def get_parameters(self, config):
        """
        Flower calls this when it needs to know our current weights.
        Returns: a list of numpy arrays [weights, biases]
        """
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        """
        Flower calls this to give us the server's latest weights.
        We load them into our model before training.
        """
        set_parameters(self.model, parameters)

    # ------------------------------------------------------------------
    #  fit() and evaluate() are the two things a client does in FL.
    #
    #  fit:      train on my data → return better weights
    #  evaluate: test on my data  → return accuracy
    # ------------------------------------------------------------------

    def fit(self, parameters, config):
        """
        1. Load the server's latest weights.
        2. Train 1 epoch on our local training data.
        3. Return the improved weights to the server.

        The server will average these improvements across all clients.
        """
        # Step 1 — get the latest global weights
        self.set_parameters(parameters)

        # Step 2 — train for 1 epoch (max_iter=1, warm_start=True)
        self.model.fit(self.X_train, self.y_train)

        # Step 3 — send the new weights back
        # Also send len(X_train) so the server knows how important we are
        # (clients with more data get more say in the average).
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        """
        Check how accurate the current global model is on our local test set.
        The server collects this from every client to track overall progress.
        """
        self.set_parameters(parameters)
        accuracy = self.model.score(self.X_test, self.y_test)
        # Use 1 - accuracy as a made-up "loss" (we only care about accuracy)
        return 1.0 - accuracy, len(self.X_test), {"accuracy": accuracy, "cid": self.cid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True,
                        help="Client ID (e.g. 0, 1, 2, ...)")
    parser.add_argument("--num_clients", type=int, default=10,
                        help="Total number of clients splitting the data")
    args = parser.parse_args()

    print(f"--- Client {args.cid} starting ---")

    # Connect to the server and start participating in FL rounds.
    # The server decides when to train, not the client.
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=DigitClient(args.cid, args.num_clients),
    )


if __name__ == "__main__":
    main()
