from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np


# ---------------------------------------------------------------------------
# 1. Model creation
# ---------------------------------------------------------------------------

def create_model():
    return LogisticRegression(
        max_iter=1,
        warm_start=True,
        solver="lbfgs",
    )


def create_sgd_model():
    return SGDClassifier(
        loss="log_loss",
        warm_start=True,
    )


# ---------------------------------------------------------------------------
# 2. Parameter helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. IID data loading — each client gets a balanced mix of all classes
# ---------------------------------------------------------------------------

def load_client_data_iid(cid, num_clients):
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0

    rng = np.random.RandomState(seed=42)
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    mask = np.arange(len(X)) % num_clients == cid
    X_client, y_client = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_client, y_client, test_size=0.2, random_state=42,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 4. Non-IID data loading — each client gets 1 dominant class + shares of 8/9
# ---------------------------------------------------------------------------

def load_client_data_non_iid(cid, num_clients):
    """
    Non-IID split:
      - Client cid gets ALL samples of digit class cid (primary class).
      - Classes 8 and 9 are split evenly among all clients.
    Every client gets roughly the same amount of total data (~224 samples).
    All 10 digit classes are covered across the system.

    Returns: (X_train, X_test, y_train, y_test)
    """
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0

    primary_mask = y == cid
    X_primary, y_primary = X[primary_mask], y[primary_mask]

    shared_mask = (y == 8) | (y == 9)
    X_shared, y_shared = X[shared_mask], y[shared_mask]

    shared_indices = np.arange(len(X))[shared_mask]
    client_shared_mask = np.arange(len(shared_indices)) % num_clients == cid
    X_shared_client = X_shared[client_shared_mask]
    y_shared_client = y_shared[client_shared_mask]

    X_client = np.concatenate([X_primary, X_shared_client])
    y_client = np.concatenate([y_primary, y_shared_client])

    X_train, X_test, y_train, y_test = train_test_split(
        X_client, y_client, test_size=0.2, random_state=42,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 5. Server test data
# ---------------------------------------------------------------------------

def load_server_test_data():
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    return X_test, y_test
