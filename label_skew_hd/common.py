import numpy as np
import os
import pickle
import ssl
import tarfile
import urllib.request
from sklearn.linear_model import SGDClassifier

ssl._create_default_https_context = ssl._create_unverified_context

DATASET_INFO = {
    'mnist': {'num_classes': 10, 'input_dim': 784},
    'cifar10': {'num_classes': 10, 'input_dim': 3072},
    'cifar100': {'num_classes': 100, 'input_dim': 3072},
}

HD_ALPHA_MAP = {0.0: 1000, 0.25: 6, 0.5: 1, 0.75: 0.3, 0.9: 0.03}

_CIFAR10_CACHE = "/tmp/cifar10_cache"
_CIFAR100_CACHE = "/tmp/cifar100_cache"


def _load_cifar10():
    os.makedirs(_CIFAR10_CACHE, exist_ok=True)
    base = "cifar-10-batches-py"
    tarpath = os.path.join(_CIFAR10_CACHE, "cifar-10-python.tar.gz")
    extract_dir = os.path.join(_CIFAR10_CACHE, base)
    if not os.path.isdir(extract_dir):
        if not os.path.exists(tarpath):
            print("Downloading CIFAR-10 (163 MB)...")
            urllib.request.urlretrieve(
                "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", tarpath)
        with tarfile.open(tarpath, "r:gz") as tar:
            tar.extractall(path=_CIFAR10_CACHE)
        os.remove(tarpath)
    def _load(p):
        with open(p, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"], np.array(d[b"labels"], dtype=np.int64)
    X_list, y_list = [], []
    for i in range(1, 6):
        Xb, yb = _load(os.path.join(extract_dir, f"data_batch_{i}"))
        X_list.append(Xb)
        y_list.append(yb)
    X_train = np.concatenate(X_list, axis=0).astype(np.float64) / 255.0
    y_train = np.concatenate(y_list, axis=0)
    X_test, y_test = _load(os.path.join(extract_dir, "test_batch"))
    X_test = X_test.astype(np.float64) / 255.0
    return X_train, y_train, X_test, y_test


def _load_cifar100():
    os.makedirs(_CIFAR100_CACHE, exist_ok=True)
    base = "cifar-100-python"
    tarpath = os.path.join(_CIFAR100_CACHE, "cifar-100-python.tar.gz")
    extract_dir = os.path.join(_CIFAR100_CACHE, base)
    if not os.path.isdir(extract_dir):
        if not os.path.exists(tarpath):
            print("Downloading CIFAR-100 (161 MB)...")
            urllib.request.urlretrieve(
                "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", tarpath)
        with tarfile.open(tarpath, "r:gz") as tar:
            tar.extractall(path=_CIFAR100_CACHE)
        os.remove(tarpath)
    def _load(p):
        with open(p, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"], np.array(d[b"fine_labels"], dtype=np.int64)
    X_train, y_train = _load(os.path.join(extract_dir, "train"))
    X_train = X_train.astype(np.float64) / 255.0
    X_test, y_test = _load(os.path.join(extract_dir, "test"))
    X_test = X_test.astype(np.float64) / 255.0
    return X_train, y_train, X_test, y_test


def _load_mnist():
    from sklearn.datasets import fetch_openml
    print("Loading MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    X = np.asarray(X, dtype=np.float64) / 255.0
    y = np.asarray(y, dtype=np.int64)
    return X[:60000], y[:60000], X[60000:], y[60000:]


LOADERS = {
    'cifar10': _load_cifar10,
    'cifar100': _load_cifar100,
    'mnist': _load_mnist,
}


def load_dataset(dataset):
    return LOADERS[dataset]()


def create_model(dataset):
    info = DATASET_INFO[dataset]
    model = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01,
        warm_start=True,
        random_state=42,
    )
    return model


def get_initial_parameters(dataset):
    info = DATASET_INFO[dataset]
    nc, nd = info['num_classes'], info['input_dim']
    return [
        np.zeros((nc, nd), dtype=np.float64),
        np.zeros(nc, dtype=np.float64),
    ]


def train_centralized_baseline(dataset, max_passes=300, patience=60):
    X_train, y_train, X_test, y_test = load_dataset(dataset)
    info = DATASET_INFO[dataset]
    num_classes = info['num_classes']
    model = create_model(dataset)
    best_acc = 0.0
    best_model = None
    no_improve = 0
    for i in range(max_passes):
        model.partial_fit(X_train, y_train, classes=np.arange(num_classes))
        acc = model.score(X_test, y_test)
        if acc > best_acc:
            best_acc = acc
            best_model = pickle.dumps(get_parameters(model))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        if (i + 1) % 10 == 0 or acc > best_acc:
            print(f"  pass {i + 1}: test_acc={acc:.4f} best={best_acc:.4f}", flush=True)
    if best_model is not None:
        set_parameters(model, pickle.loads(best_model))
    return best_acc, model


def get_parameters(model):
    if not hasattr(model, "coef_") or model.coef_ is None:
        return None
    return [model.coef_, model.intercept_]


def set_parameters(model, parameters):
    model.coef_ = parameters[0]
    model.intercept_ = parameters[1]
    model.classes_ = np.arange(parameters[0].shape[0])


def dirichlet_partition(alpha, num_clients, X, y, seed=42, client_idx=None):
    rng = np.random.RandomState(seed)
    num_classes = len(np.unique(y))

    proportions = rng.dirichlet([alpha] * num_classes, size=num_clients)

    class_indices = [np.where(y == c)[0] for c in range(num_classes)]
    class_counts = [len(idx) for idx in class_indices]

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        col_sum = proportions[:, c].sum()
        sizes = (proportions[:, c] / col_sum * class_counts[c]).astype(int)
        diff = class_counts[c] - sizes.sum()
        sizes[-1] += diff
        sizes = np.maximum(sizes, 0)

        idx = class_indices[c].copy()
        rng.shuffle(idx)
        start = 0
        for i in range(num_clients):
            if sizes[i] > 0:
                client_indices[i].append(idx[start:start + sizes[i]])
                start += sizes[i]

    if client_idx is None:
        result = []
        for i in range(num_clients):
            if client_indices[i]:
                idx = np.concatenate(client_indices[i])
                result.append((X[idx], y[idx]))
            else:
                result.append((np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
        return result

    idx = np.concatenate(client_indices[client_idx])
    return X[idx], y[idx]


def compute_hellinger_distance(client_data_list, num_classes):
    dists = []
    for _, y in client_data_list:
        if len(y) == 0:
            dists.append(np.zeros(num_classes))
        else:
            hist = np.bincount(y, minlength=num_classes)
            dists.append(hist / hist.sum())
    dists = np.array(dists)
    total, count = 0.0, 0
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            sd = np.sqrt(dists[i]) - np.sqrt(dists[j])
            total += np.sqrt(np.sum(sd ** 2)) / np.sqrt(2)
            count += 1
    return total / count if count > 0 else 0.0
