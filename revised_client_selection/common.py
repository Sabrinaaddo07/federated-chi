import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

NUM_CLASSES = 10
INPUT_DIM = 64


def create_model():
    return SGDClassifier(
        loss="log_loss",
        warm_start=True,
        random_state=42,
    )


def get_parameters(model):
    if not hasattr(model, "coef_") or model.coef_ is None:
        return [
            np.zeros((NUM_CLASSES, INPUT_DIM), dtype=np.float64),
            np.zeros(NUM_CLASSES, dtype=np.float64),
        ]
    return [model.coef_, model.intercept_]


def set_parameters(model, parameters):
    model.coef_ = parameters[0]
    model.intercept_ = parameters[1]
    model.classes_ = np.arange(NUM_CLASSES)


def load_server_test_data():
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test


def load_client_data_iid(cid, num_clients):
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target

    rng = np.random.RandomState(seed=42)
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    mask = np.arange(len(X)) % num_clients == cid
    X_c, y_c = X[mask], y[mask]
    return train_test_split(X_c, y_c, test_size=0.2, random_state=42)
