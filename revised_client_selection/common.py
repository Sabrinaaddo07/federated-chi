import numpy as np
import os
import pickle
import ssl
import tarfile
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

NUM_CLASSES = 10
INPUT_DIM = 3072

ssl._create_default_https_context = ssl._create_unverified_context

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CACHE_DIR = "/tmp/cifar10_cache"


def _load_cifar10():
    """Download cached CIFAR-10 once, return (X_train, y_train, X_test, y_test).
    
    If the download is corrupted, removes the bad files and retries.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tarpath = os.path.join(_CACHE_DIR, "cifar-10-python.tar.gz")
    extract_dir = os.path.join(_CACHE_DIR, "cifar-10-batches-py")

    for attempt in range(2):
        if not os.path.isdir(extract_dir):
            if not os.path.exists(tarpath):
                print("Downloading CIFAR-10 (163 MB)...")
                urllib.request.urlretrieve(CIFAR10_URL, tarpath)
            try:
                with tarfile.open(tarpath, "r:gz") as tar:
                    tar.extractall(path=_CACHE_DIR)
            except (EOFError, tarfile.ReadError):
                print("Download corrupted, retrying...")
                os.remove(tarpath)
                for p in os.listdir(_CACHE_DIR):
                    fp = os.path.join(_CACHE_DIR, p)
                    if os.path.isdir(fp):
                        import shutil
                        shutil.rmtree(fp)
                continue

        def _load_batch(path):
            with open(path, "rb") as f:
                d = pickle.load(f, encoding="bytes")
            return d[b"data"], np.array(d[b"labels"], dtype=np.int64)

        try:
            X_train_list, y_train_list = [], []
            for i in range(1, 6):
                Xb, yb = _load_batch(os.path.join(extract_dir, f"data_batch_{i}"))
                X_train_list.append(Xb)
                y_train_list.append(yb)
            X_train = np.concatenate(X_train_list, axis=0).astype(np.float64) / 255.0
            y_train = np.concatenate(y_train_list, axis=0)

            X_test, y_test = _load_batch(os.path.join(extract_dir, "test_batch"))
            X_test = X_test.astype(np.float64) / 255.0
            return X_train, y_train, X_test, y_test
        except Exception:
            if attempt == 0:
                import shutil
                shutil.rmtree(extract_dir)
                if os.path.exists(tarpath):
                    os.remove(tarpath)
                print("CIFAR-10 data corrupted, retrying...")
                continue
            raise


def create_model():
    return SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.00001,
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
    _, _, X_test, y_test = _load_cifar10()
    return X_test, y_test


def load_full_data():
    """Return full CIFAR-10 training data (50K samples) without partitioning."""
    X_train, y_train, _, _ = _load_cifar10()
    rng = np.random.RandomState(seed=42)
    idx = rng.permutation(len(X_train))
    return X_train[idx], y_train[idx]


def load_client_data_iid(cid, num_clients):
    X_train, y_train, _, _ = _load_cifar10()

    rng = np.random.RandomState(seed=42)
    indices = rng.permutation(len(X_train))
    X_train, y_train = X_train[indices], y_train[indices]

    mask = np.arange(len(X_train)) % num_clients == cid
    X_c, y_c = X_train[mask], y_train[mask]
    return train_test_split(X_c, y_c, test_size=0.2, random_state=42)
