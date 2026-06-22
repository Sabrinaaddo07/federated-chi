import argparse
import logging
import os
import warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
warnings.filterwarnings("ignore")

import flwr as fl
from common import create_model, get_parameters, set_parameters, load_client_data

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)


class DigitClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients):
        self.cid = cid
        self.X_train, self.X_test, self.y_train, self.y_test = \
            load_client_data(cid, num_clients=num_clients)
        self.model = create_model()
        print(f"  Client {cid} loaded {len(self.X_train)} training + "
              f"{len(self.X_test)} test samples")
        print(f"  This client's data is PRIVATE — never sent to the server\n")

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config), len(self.X_train), {"cid": self.cid}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = self.model.score(self.X_test, self.y_test)
        return 1.0 - accuracy, len(self.X_test), {"accuracy": accuracy, "cid": self.cid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True,
                        help="Client ID (0–9)")
    parser.add_argument("--num_clients", type=int, default=10,
                        help="Total number of clients splitting the data")
    args = parser.parse_args()

    print(f"--- Client {args.cid} starting ---")
    fl.client.start_numpy_client(
        server_address="10.0.0.1:8080",
        client=DigitClient(args.cid, args.num_clients),
    )


if __name__ == "__main__":
    main()
