"""Image inputs and split indices for the official torchvision datasets."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from runtime.paths import data_root, split_dir


# (torchvision class name, official training size, official test size)
OFFICIAL = {
    "mnist": ("MNIST", 60000, 10000),
    "fashion_mnist": ("FashionMNIST", 60000, 10000),
    "cifar10": ("CIFAR10", 50000, 10000),
}


def official_dataset(dataset: str, train: bool):
    from torchvision import datasets

    name = OFFICIAL[dataset][0]
    return getattr(datasets, name)(root=data_root() / "torchvision", train=train, download=True)


def _images(dataset: str, train: bool, indices: np.ndarray) -> np.ndarray:
    cache = data_root() / f"{dataset}_full.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as arrays:
            return arrays["train_x" if train else "test_x"][indices]
    pixels = np.asarray(official_dataset(dataset, train).data)[indices]
    if pixels.ndim == 3:
        pixels = pixels[:, None, :, :]
    else:
        pixels = pixels.transpose(0, 3, 1, 2)
    return np.ascontiguousarray(pixels, dtype=np.float32) / 255.0


def _training_classes(dataset: str) -> np.ndarray:
    cache = data_root() / f"{dataset}_full.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as arrays:
            return arrays["train_y"]
    return np.asarray(official_dataset(dataset, train=True).targets, dtype=np.int64)


def base_indices(dataset: str, n_train: int, n_test: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = split_dir() / f"{dataset}_seed{seed}_ntr{n_train}_nte{n_test}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            return stored["train_idx"], stored["test_idx"]
    _, total_train, total_test = OFFICIAL[dataset]
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(total_train, size=min(int(n_train), total_train), replace=False).astype(np.int64)
    test_idx = rng.choice(total_test, size=min(int(n_test), total_test), replace=False).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each writer produces the same split. Publish only complete NPZ files.
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        temp_path = Path(tmp.name)
        np.savez(tmp, train_idx=train_idx, test_idx=test_idx)
    temp_path.replace(path)
    return train_idx, test_idx


def training_subset(
    train_idx: np.ndarray,
    train_y: np.ndarray | None,
    sample_frac: float,
    imbalance: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.asarray(train_idx, dtype=np.int64)
    if sample_frac < 1.0:
        size = max(1, int(round(len(idx) * float(sample_frac))))
        idx = rng.choice(idx, size=min(size, len(idx)), replace=False).astype(np.int64)
    if imbalance > 1.0:
        if train_y is None:
            raise ValueError("training classes required to construct the imbalance condition")
        labels = train_y[idx]
        classes, counts = np.unique(labels, return_counts=True)
        majority = classes[int(np.argmax(counts))]
        n_majority = int(counts.max())
        n_minor = max(1, int(round(n_majority / float(imbalance))))
        kept = []
        for cls, count in zip(classes, counts):
            pool = idx[labels == cls]
            take = int(count) if cls == majority else min(len(pool), n_minor)
            kept.append(rng.choice(pool, size=take, replace=False))
        idx = np.concatenate(kept).astype(np.int64)
    rng.shuffle(idx)
    return idx


def apply_outliers(images: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    images = np.array(images, copy=True)
    if fraction <= 0:
        return images
    rng = np.random.default_rng(seed + 17)
    count = int(round(len(images) * float(fraction)))
    if count <= 0:
        return images
    pick = rng.choice(len(images), size=count, replace=False)
    images[pick] = rng.random((count,) + images.shape[1:], dtype=np.float32)
    return images


def add_gaussian_noise(images: np.ndarray, std: float, seed: int) -> np.ndarray:
    if std <= 0:
        return np.array(images, dtype=np.float32, copy=True)
    rng = np.random.default_rng(seed + 101)
    noisy = images.astype(np.float32) + rng.normal(0.0, float(std), size=images.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def _split(dataset: str, protocol: dict, seed: int):
    spec = protocol["splits"][dataset]
    return base_indices(dataset, spec["n_train"], spec["n_test"], seed)


def training_images(dataset: str, protocol: dict, seed: int, sample_frac: float = 1.0,
                    imbalance: float = 1.0, outlier_frac: float = 0.0):
    train_idx, _ = _split(dataset, protocol, seed)
    classes = _training_classes(dataset) if imbalance > 1 else None
    indices = training_subset(train_idx, classes, sample_frac, imbalance, seed)
    images = _images(dataset, True, indices)
    return apply_outliers(images, outlier_frac, seed), indices


def evaluation_images(dataset: str, protocol: dict, seed: int):
    _, indices = _split(dataset, protocol, seed)
    return _images(dataset, False, indices), indices


def materialize(dataset: str, protocol: dict, seed: int, sample_frac: float, imbalance: float, outlier_frac: float):
    train, _ = training_images(dataset, protocol, seed, sample_frac, imbalance, outlier_frac)
    test, indices = evaluation_images(dataset, protocol, seed)
    return train, test, indices
