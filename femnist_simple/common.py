import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

DATA_NPZ = "emnist_data.npz"
NUM_CLASSES = 47
INPUT_DIM = 784


# ---------------------------------------------------------------------------
# 1. Subsampled EMNIST loader (uses existing .npz in parent directory)
# ---------------------------------------------------------------------------

def load_emnist_subset(n_train=5000, n_test=2000, random_state=42):
    import os
    npz_path = os.path.join(os.path.dirname(__file__), "..", "femnist_experiment", "emnist_data.npz")
    data = np.load(npz_path)
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_te, y_te = data["X_test"], data["y_test"]

    def stratify_sample(X, y, n, rs):
        classes = np.unique(y)
        per_class = n // len(classes)
        idxs = []
        for c in classes:
            c_idxs = np.where(y == c)[0]
            rs.shuffle(c_idxs)
            idxs.extend(c_idxs[:per_class])
        idxs = np.array(idxs)
        rs.shuffle(idxs)
        return X[idxs], y[idxs]

    rng = np.random.RandomState(random_state)
    X_tr, y_tr = stratify_sample(X_tr, y_tr, n_train, rng)
    X_te, y_te = stratify_sample(X_te, y_te, n_test, rng)

    print(f"  Subsampled EMNIST: {len(X_tr)} train, {len(X_te)} test, {len(np.unique(y_tr))} classes")
    return X_tr, y_tr, X_te, y_te


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------

def create_model():
    return SGDClassifier(
        loss="log_loss",
        warm_start=True,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# 3. Weight helpers
# ---------------------------------------------------------------------------

def get_parameters(model):
    if not hasattr(model, "coef_") or model.coef_ is None:
        return [np.zeros((NUM_CLASSES, INPUT_DIM), dtype=np.float64),
                np.zeros(NUM_CLASSES, dtype=np.float64)]
    return [model.coef_, model.intercept_]


def set_parameters(model, parameters):
    model.coef_ = parameters[0]
    model.intercept_ = parameters[1]
    model.classes_ = np.arange(NUM_CLASSES)


# ---------------------------------------------------------------------------
# 4. Data partitioners
# ---------------------------------------------------------------------------

def load_client_data_iid(cid, num_clients, X, y):
    rng = np.random.RandomState(seed=42)
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]
    mask = np.arange(len(X)) % num_clients == cid
    X_c, y_c = X[mask], y[mask]
    return train_test_split(X_c, y_c, test_size=0.2, random_state=42)


def load_client_data_non_iid(cid, num_clients, X, y, alpha=0.1):
    rng = np.random.RandomState(seed=42 + cid)
    class_indices = [np.where(y == c)[0] for c in range(NUM_CLASSES)]
    client_props = rng.dirichlet(np.repeat(alpha, NUM_CLASSES))
    client_props = np.maximum(client_props, 1e-6)
    client_props /= client_props.sum()

    idxs = []
    for c in range(NUM_CLASSES):
        n = int(client_props[c] * len(X) / num_clients)
        n = min(n, len(class_indices[c]))
        if n > 0:
            chosen = rng.choice(class_indices[c], size=n, replace=False)
            idxs.extend(chosen)
            class_indices[c] = np.setdiff1d(class_indices[c], chosen)

    idxs = np.array(idxs)
    rng.shuffle(idxs)
    X_c, y_c = X[idxs], y[idxs]
    return train_test_split(X_c, y_c, test_size=0.2, random_state=42)
