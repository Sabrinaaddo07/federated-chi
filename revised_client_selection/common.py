import numpy as np
import os
import pickle
import ssl
import tarfile
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

NUM_CLASSES = 100
INPUT_DIM = 3072

ssl._create_default_https_context = ssl._create_unverified_context

CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
_CACHE_DIR = "/tmp/cifar100_cache"


def _load_cifar100():
    """Download cached CIFAR-100 once, return (X_train, y_train, X_test, y_test).
    
    CIFAR-100 has 50K train + 10K test, 100 classes (fine labels).
    If the download is corrupted, removes the bad files and retries.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tarpath = os.path.join(_CACHE_DIR, "cifar-100-python.tar.gz")
    extract_dir = os.path.join(_CACHE_DIR, "cifar-100-python")

    for attempt in range(2):
        if not os.path.isdir(extract_dir):
            if not os.path.exists(tarpath):
                print("Downloading CIFAR-100 (161 MB)...")
                urllib.request.urlretrieve(CIFAR100_URL, tarpath)
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

        def _load_file(path):
            with open(path, "rb") as f:
                d = pickle.load(f, encoding="bytes")
            return d[b"data"], np.array(d[b"fine_labels"], dtype=np.int64)

        try:
            X_train, y_train = _load_file(os.path.join(extract_dir, "train"))
            X_train = X_train.astype(np.float64) / 255.0

            X_test, y_test = _load_file(os.path.join(extract_dir, "test"))
            X_test = X_test.astype(np.float64) / 255.0

            return X_train, y_train, X_test, y_test
        except Exception:
            if attempt == 0:
                import shutil
                shutil.rmtree(extract_dir)
                if os.path.exists(tarpath):
                    os.remove(tarpath)
                print("CIFAR-100 data corrupted, retrying...")
                continue
            raise


def create_model():
    return SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01,
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
    _, _, X_test, y_test = _load_cifar100()
    return X_test, y_test


def load_full_data():
    """Return full CIFAR-100 training data (50K samples) without partitioning."""
    X_train, y_train, _, _ = _load_cifar100()
    rng = np.random.RandomState(seed=42)
    idx = rng.permutation(len(X_train))
    return X_train[idx], y_train[idx]


def load_partitioned_data(cid, num_clients):
    """Return an exclusive 1/num_clients partition for client cid (no train/test split)."""
    X_train, y_train, _, _ = _load_cifar100()
    rng = np.random.RandomState(seed=42)
    idx = rng.permutation(len(X_train))
    X_shuf, y_shuf = X_train[idx], y_train[idx]
    chunk = len(X_train) // num_clients
    return X_shuf[cid * chunk:(cid + 1) * chunk], y_shuf[cid * chunk:(cid + 1) * chunk]


def load_client_data_iid(cid, num_clients):
    X_train, y_train, _, _ = _load_cifar100()

    rng = np.random.RandomState(seed=42)
    indices = rng.permutation(len(X_train))
    X_train, y_train = X_train[indices], y_train[indices]

    mask = np.arange(len(X_train)) % num_clients == cid
    X_c, y_c = X_train[mask], y_train[mask]
    return train_test_split(X_c, y_c, test_size=0.2, random_state=42)
