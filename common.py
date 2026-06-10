"""
common.py - Shared code used by both the server and the clients.

What this is for:
- Defines the machine learning model (Logistic Regression)
- Defines how to extract / load model weights
- Defines how to load / split the dataset
- Both server.py and client.py import from here
"""

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np


# ---------------------------------------------------------------------------
# 1. Model creation
# ---------------------------------------------------------------------------

def create_model():
    """
    Returns a fresh Logistic Regression model.

    Federated learning note:
    - max_iter=1: each time we call .fit(), it does only 1 epoch of training.
      That way we can do many rounds where each round = 1 local epoch.
    - warm_start=True: keeps the existing weights instead of resetting them
      every time. This lets the model keep improving round after round.
    """
    return LogisticRegression(
        max_iter=1,
        warm_start=True,
        solver="lbfgs",
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
# 3. Data loading
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
# 4. Server test data
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
