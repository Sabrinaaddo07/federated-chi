from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np


def create_model():
    return LogisticRegression(
        max_iter=1,
        warm_start=True,
        solver="lbfgs",
    )


def get_parameters(model):
    if not hasattr(model, "coef_") or model.coef_ is None:
        return [
            np.zeros((10, 64), dtype=np.float64),
            np.zeros(10, dtype=np.float64),
        ]
    return [model.coef_, model.intercept_]


def set_parameters(model, parameters):
    model.coef_ = parameters[0]
    model.intercept_ = parameters[1]
    model.classes_ = np.arange(10)


def load_client_data(cid, num_clients):
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0

    total = len(X)
    start = cid * (total // num_clients)
    end = (cid + 1) * (total // num_clients)
    X_client, y_client = X[start:end], y[start:end]

    X_train, X_test, y_train, y_test = train_test_split(
        X_client, y_client, test_size=0.2, random_state=42,
    )
    return X_train, X_test, y_train, y_test


def load_server_test_data():
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    return X_test, y_test
