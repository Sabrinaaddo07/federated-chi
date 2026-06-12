import argparse
import logging
import os
import warnings

os.environ["GRPC_VERBOSITY"] = "ERROR"
warnings.filterwarnings("ignore")

import flwr as fl
import numpy as np
from common import create_model, get_parameters, set_parameters, load_client_data

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)
for h in flwr_logger.handlers:
    h.setLevel(logging.ERROR)


class DropoutClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients, dropout_rate):
        self.cid = cid
        self.dropout_rate = dropout_rate
        self.X_train, self.X_test, self.y_train, self.y_test = \
            load_client_data(cid, num_clients=num_clients)
        self.model = create_model()
        self.rng = np.random.RandomState(seed=42 + cid)

        print(f"  Client {cid} — dropout_rate={dropout_rate}")
        print(f"    {len(self.X_train)} training + {len(self.X_test)} test samples")

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        old_coef = self.model.coef_.copy()
        old_intercept = self.model.intercept_.copy()

        mask_coef = self.rng.binomial(
            1, 1 - self.dropout_rate, old_coef.shape
        ).astype(bool)
        mask_intercept = self.rng.binomial(
            1, 1 - self.dropout_rate, old_intercept.shape
        ).astype(bool)

        self.model.fit(self.X_train, self.y_train)

        new_coef = old_coef.copy()
        new_intercept = old_intercept.copy()
        new_coef[mask_coef] = self.model.coef_[mask_coef]
        new_intercept[mask_intercept] = self.model.intercept_[mask_intercept]

        self.model.coef_ = new_coef
        self.model.intercept_ = new_intercept

        num_params_sent = int(np.sum(mask_coef)) + int(np.sum(mask_intercept))
        num_params_total = old_coef.size + old_intercept.size
        train_acc = float(self.model.score(self.X_train, self.y_train))

        return [new_coef, new_intercept], len(self.X_train), {
            "cid": self.cid,
            "dropout_rate": self.dropout_rate,
            "num_params_sent": num_params_sent,
            "num_params_total": num_params_total,
            "train_accuracy": train_acc,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = self.model.score(self.X_test, self.y_test)
        return 1.0 - accuracy, len(self.X_test), {"accuracy": accuracy, "cid": self.cid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID (0–9)")
    parser.add_argument("--num_clients", type=int, default=10, help="Total number of clients")
    parser.add_argument("--dropout_rate", type=float, default=0.0,
                        help="Fraction of model weights to drop (0.0–1.0)")
    args = parser.parse_args()

    print(f"--- Client {args.cid} starting (dropout_rate={args.dropout_rate}) ---")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=DropoutClient(args.cid, args.num_clients, args.dropout_rate),
    )


if __name__ == "__main__":
    main()
