"""
common.py - Shared code for FEMNIST federated learning experiment.

- Creates an MLPClassifier model
- Provides parameter get/set helpers for Flower
- Loads EMNIST data (from local .npz cache via download_data.py)
- Splits data IID (shuffle + round-robin) or non-IID (Dirichlet per-client)
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(DATA_DIR, "emnist_data.npz")


# ---------------------------------------------------------------------------
# 1. EMNIST data loading (with auto-download to local npz)
# ---------------------------------------------------------------------------

def load_emnist_data():
    if not os.path.exists(NPZ_PATH):
        print("  EMNIST data not found locally. Running download_data.py ...")
        import subprocess
        subprocess.check_call(
            ["python3", os.path.join(DATA_DIR, "download_data.py")]
        )
    data = np.load(NPZ_PATH)
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


# ---------------------------------------------------------------------------
# 2. Model creation
# ---------------------------------------------------------------------------

INPUT_DIM = 784


def create_model(num_classes=47):
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="sgd",
        learning_rate_init=0.01,
        max_iter=1,
        warm_start=True,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# 3. Parameter helpers
# ---------------------------------------------------------------------------

def get_parameters(model, num_classes=47):
    if not hasattr(model, "coefs_") or model.coefs_ is None:
        layers = [INPUT_DIM] + list(model.hidden_layer_sizes) + [num_classes]
        n = len(layers) - 1
        return [np.zeros((layers[i], layers[i+1]), dtype=np.float64)
                for i in range(n)] + \
               [np.zeros(layers[i+1], dtype=np.float64)
                for i in range(n)]
    return model.coefs_ + model.intercepts_


def set_parameters(model, parameters):
    n_layers = len(parameters) // 2
    model.coefs_ = list(parameters[:n_layers])
    model.intercepts_ = list(parameters[n_layers:])
    model.n_layers_ = n_layers + 1
    model.n_output_ = model.intercepts_[-1].shape[0]
    model.n_features_in_ = model.coefs_[0].shape[0]
    model._label_binarizer = None
    model._optimizer = None
    model.loss_ = None
    model._no_improvement_count = 0
    model.best_loss_ = None
    model.t_ = 0
    model.n_iter_ = 0


# ---------------------------------------------------------------------------
# 4. IID data loading — each client gets a balanced mix of all classes
# ---------------------------------------------------------------------------

def load_client_data_iid(cid, num_clients, X, y):
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
# 5. Non-IID data loading — Dirichlet distribution per client
# ---------------------------------------------------------------------------

def load_client_data_non_iid(cid, num_clients, X, y, alpha=0.5):
    num_classes = len(np.unique(y))
    rng = np.random.RandomState(seed=42 + cid)

    class_indices = [np.where(y == c)[0] for c in range(num_classes)]

    client_proportions = rng.dirichlet(np.repeat(alpha, num_classes))
    client_proportions = np.maximum(client_proportions, 1e-6)
    client_proportions /= client_proportions.sum()

    client_indices = []
    for c in range(num_classes):
        n_from_class = int(client_proportions[c] * len(X) / num_clients)
        n_from_class = min(n_from_class, len(class_indices[c]))
        if n_from_class > 0:
            chosen = rng.choice(class_indices[c], size=n_from_class, replace=False)
            client_indices.extend(chosen)
            class_indices[c] = np.setdiff1d(class_indices[c], chosen)

    client_indices = np.array(client_indices)
    rng.shuffle(client_indices)

    X_client, y_client = X[client_indices], y[client_indices]

    X_train, X_test, y_train, y_test = train_test_split(
        X_client, y_client, test_size=0.2, random_state=42,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 6. Server test data
# ---------------------------------------------------------------------------

def load_server_test_data(X_test, y_test):
    return X_test, y_test


# ---------------------------------------------------------------------------
# 7. Data analysis helpers
# ---------------------------------------------------------------------------

def client_class_distribution(cid, num_clients, X, y, scheme, alpha=0.5):
    if scheme == "iid":
        _, _, y_train, _ = load_client_data_iid(cid, num_clients, X, y)
    else:
        _, _, y_train, _ = load_client_data_non_iid(cid, num_clients, X, y, alpha=alpha)
    counts = np.bincount(y_train, minlength=len(np.unique(y)))
    return counts / counts.sum()
