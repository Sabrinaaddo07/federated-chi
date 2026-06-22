"""
download_data.py - Download EMNIST dataset for the FEMNIST experiment.

Downloads the EMNIST gzip archive from the NIST biometrics repository,
extracts the IDX-format image/label files for the chosen split,
normalizes pixels to [0,1], and saves as a NumPy .npz file.

Usage:
    python3 download_data.py                     # balanced split (47 classes, default)
    python3 download_data.py --split byclass     # full 62-class split
"""

import argparse
import gzip
import io
import os
import ssl
import struct
import urllib.request
import zipfile

import numpy as np


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(DATA_DIR, "emnist_data.npz")

NIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

SPLIT_FILES = {
    "balanced": {
        "images": "gzip/emnist-balanced-train-images-idx3-ubyte.gz",
        "labels": "gzip/emnist-balanced-train-labels-idx1-ubyte.gz",
        "test_images": "gzip/emnist-balanced-test-images-idx3-ubyte.gz",
        "test_labels": "gzip/emnist-balanced-test-labels-idx1-ubyte.gz",
    },
    "byclass": {
        "images": "gzip/emnist-byclass-train-images-idx3-ubyte.gz",
        "labels": "gzip/emnist-byclass-train-labels-idx1-ubyte.gz",
        "test_images": "gzip/emnist-byclass-test-images-idx3-ubyte.gz",
        "test_labels": "gzip/emnist-byclass-test-labels-idx1-ubyte.gz",
    },
}


def _unverified_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def download_nist_zip(save_dir):
    zip_path = os.path.join(save_dir, "emnist_gzip.zip")
    if os.path.exists(zip_path):
        print(f"  Found existing download: {zip_path}")
        return zip_path

    print(f"  Downloading from NIST (~530 MB)...")
    ctx = _unverified_ctx()
    req = urllib.request.Request(NIST_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as r:
        with open(zip_path, "wb") as f:
            f.write(r.read())
    print(f"  Saved to {zip_path}")
    return zip_path


def parse_idx(byte_data):
    """
    Parse an IDX file, returning (data, shape).
    IDX format:
      magic[0]: 0
      magic[1]: 0
      magic[2]: number of dimensions (1 for labels, 3 for images)
      magic[3]: data type (0x08 = unsigned byte)
      followed by dimension sizes (4 bytes each, big-endian)
      followed by actual data
    """
    buf = io.BytesIO(byte_data)
    magic = struct.unpack(">BBBB", buf.read(4))
    ndim = magic[3]
    dims = struct.unpack(">" + "I" * ndim, buf.read(4 * ndim))
    dtype = np.uint8
    data = np.frombuffer(buf.read(), dtype=dtype).reshape(dims)
    return data


def extract_idx_from_zip(zip_path, member_name):
    with zipfile.ZipFile(zip_path, "r") as zf:
        raw = zf.read(member_name)
    # Files are gzipped inside the zip
    data = gzip.decompress(raw)
    return parse_idx(data)


def load_emnist(split="balanced"):
    save_dir = DATA_DIR
    zip_path = download_nist_zip(save_dir)

    print(f"  Extracting {split} split (IDX format)...")

    images = extract_idx_from_zip(zip_path, SPLIT_FILES[split]["images"])
    labels = extract_idx_from_zip(zip_path, SPLIT_FILES[split]["labels"])
    test_images = extract_idx_from_zip(zip_path, SPLIT_FILES[split]["test_images"])
    test_labels = extract_idx_from_zip(zip_path, SPLIT_FILES[split]["test_labels"])

    # Flatten images and normalize to [0, 1]
    X_train = images.reshape(images.shape[0], -1).astype(np.float64) / 255.0
    y_train = labels.astype(np.int64)
    X_test = test_images.reshape(test_images.shape[0], -1).astype(np.float64) / 255.0
    y_test = test_labels.astype(np.int64)

    # EMNIST balanced uses labels 1-47; shift to 0-46 for contiguous indexing
    unique = np.unique(y_train)
    mapping = {old: new for new, old in enumerate(sorted(unique))}
    y_train = np.array([mapping[l] for l in y_train])
    y_test = np.array([mapping[l] for l in y_test])

    print(f"    Train: {X_train.shape}  |  Test: {X_test.shape}  |  "
          f"Classes: {len(unique)}")
    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser(
        description="Download EMNIST dataset for FEMNIST experiment."
    )
    parser.add_argument(
        "--split", type=str, default="balanced",
        choices=list(SPLIT_FILES.keys()),
        help="EMNIST split: balanced=47 classes, byclass=62 classes",
    )
    args = parser.parse_args()

    split = args.split
    class_names = {"balanced": 47, "byclass": 62}

    print("=" * 60)
    print(f"  Downloading EMNIST ({split}) — {class_names[split]} classes")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_emnist(split=split)

    np.savez_compressed(NPZ_PATH, X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)
    size_mb = os.path.getsize(NPZ_PATH) / 1e6
    print(f"\n  Saved to {NPZ_PATH} ({size_mb:.1f} MB)")
    print(f"    X_train: {X_train.shape}")
    print(f"    X_test:  {X_test.shape}")
    print(f"    Classes: {len(np.unique(y_train))}")


if __name__ == "__main__":
    main()
