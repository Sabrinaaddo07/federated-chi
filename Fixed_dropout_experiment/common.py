"""
common.py - Shared code used by both the server and the clients.

What this is for:
- Defines the machine learning model (Logistic Regression)
- Defines how to extract / load model weights
- Defines how to load / split the dataset
- Both server.py and client.py import from here
"""

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np


# ---------------------------------------------------------------------------
# 1. Model creation
# ---------------------------------------------------------------------------

def create_model():
    """
    Returns a fresh Logistic Regression model (batch L-BFGS solver).

    Federated learning note:
    - max_iter=1: each time we call .fit(), it does only 1 epoch of training.
      That way we can do many rounds where each round = 1 local epoch.
    - warm_start=True: keeps the existing weights instead of resetting them
      every time. This lets the model keep improving round after round.

    NOTE: This model requires at least 2 classes in the training data.
    Use create_sgd_model() for single-class-per-client scenarios.
    """
    return LogisticRegression(
        max_iter=1,
        warm_start=True,
        solver="lbfgs",
    )


def create_sgd_model():
    """
    Returns a fresh SGD-based Logistic Regression model (online solver).

    Unlike create_model(), this uses stochastic gradient descent and
    supports training on data with only 1 class.
    Use with model.partial_fit(X, y, classes=np.arange(10)).

    Compatible with the same get_parameters / set_parameters helpers.
    """
    return SGDClassifier(
        loss="log_loss",
        warm_start=True,
    )


# ---------------------------------------------------------------------------
# 2. Parameter helpers
# ---------------------------------------------------------------------------
# These two functions are how Flower moves model weights around.
# Flower doesn't know about sklearn — it only knows how to send NumPy arrays.
#
#   get_parameters(model)  →  list of NumPy arrays  (sent to server)
#   set_parameters(model, list of arrays)  ← loaded from server
#
# The "parameters" are just the weights and biases.

def get_parameters(model):
    """
    Return [weights, biases] as a list of two NumPy arrays.
    """
    if not hasattr(model, "coef_") or model.coef_ is None:
        # Model has never been trained → return zeros.
        # load_digits has 10 classes and 64 features (8×8 pixels).
        return [
            np.zeros((10, 64), dtype=np.float64),   # weights
            np.zeros(10, dtype=np.float64),           # biases
        ]
    return [model.coef_, model.intercept_]


def set_parameters(model, parameters):
    """
    Copy the given weights and biases into the model.
    """
    model.coef_ = parameters[0]
    model.intercept_ = parameters[1]
    model.classes_ = np.arange(10)      # tell sklearn there are 10 digits


# ---------------------------------------------------------------------------
# 3. IID data loading — each client gets a balanced mix of all classes
# ---------------------------------------------------------------------------

def load_client_data_iid(cid, num_clients):
    """
    IID split: randomly shuffle all samples, assign round-robin to clients.
    Every client gets roughly the same number of samples from every digit class.

    Returns: (X_train, X_test, y_train, y_test)
    """
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
# 4. Non-IID data loading — each client gets exclusive digit classes
# ---------------------------------------------------------------------------

def load_client_data_non_iid(cid, num_clients, scheme="a"):
    """
    Non-IID split: assign each client data from only 1-2 digit classes.

    Scheme "a": 8 clients, classes 0-7 only (classes 8,9 held out).
      Client cid → digit class cid exclusively.

    Scheme "b": 8 clients, all 10 classes covered.
      Clients 0-6 → digit class cid exclusively.
      Client 7   → digit classes 7, 8, 9 (3× the data).

    Returns: (X_train, X_test, y_train, y_test)
    """
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0

    if scheme == "a":
        mask = y == cid
        X_client, y_client = X[mask], y[mask]
    elif scheme == "b":
        if cid < 7:
            mask = y == cid
            X_client, y_client = X[mask], y[mask]
        else:
            mask = (y == 7) | (y == 8) | (y == 9)
            X_client, y_client = X[mask], y[mask]
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_client, y_client, test_size=0.2, random_state=42,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 5. Data loading (original contiguous split — kept for backward compat)
# ---------------------------------------------------------------------------

def load_client_data(cid, num_clients):
    """
    Give this client its own private slice of the digits dataset.

    Federated learning note:
    Each client ONLY sees its own slice. The slices are different.
    If a client is eavesdropped, the attacker only gets that slice,
    never the full dataset or another client's data.

    Returns: (X_train, X_test, y_train, y_test)
    """

    # Load the built-in digits dataset (1797 images of 8×8 pixels).
    # This is a smaller, faster version of MNIST.
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0     # normalise pixel values
    y = digits.target

    # Work out which rows belong to this client.
    # Client 0 gets the first half, Client 1 gets the second half.
    total = len(X)
    start = cid * (total // num_clients)
    end = (cid + 1) * (total // num_clients)

    X_client = X[start:end]
    y_client = y[start:end]

    # Split the client's slice: 80% train, 20% local test.
    X_train, X_test, y_train, y_test = train_test_split(
        X_client, y_client, test_size=0.2, random_state=42,
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 6. Server test data (unchanged — always uses full dataset)
# ---------------------------------------------------------------------------

def load_server_test_data():
    """
    Return a held-out test set from the full digits dataset.

    The server uses this to evaluate the global model each round.
    No client ever trains on this data, so it gives an unbiased
    accuracy estimate.
    """
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    return X_test, y_test


# ---------------------------------------------------------------------------
# 7. Class-map-based data loading (for poisoning experiments)
# ---------------------------------------------------------------------------

POISON_CLASS_MAPS = {
    "single": {i: [i] for i in range(7)} | {7: [8]},
    "multi": {7: [1, 2, 3], 0: [0], 1: [4], 2: [5], 3: [6], 4: [7], 5: [8], 6: [9]},
}


def load_client_data_from_class_map(cid, class_map):
    """
    Load data for a client given a class map dict.
    class_map: dict mapping cid -> list of digit classes
    """
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0

    classes = class_map[cid]
    mask = np.zeros(len(y), dtype=bool)
    for c in classes:
        mask |= (y == c)
    X_client, y_client = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_client, y_client, test_size=0.2, random_state=42,
    )
    return X_train, X_test, y_train, y_test
