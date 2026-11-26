"""Dataset splits. Labels leave this module only for subset construction and final evaluation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from runtime.paths import data_root, split_dir


OFFICIAL = {
    "mnist": ("MNIST", 1, 28),
    "fashion_mnist": ("FashionMNIST", 1, 28),
    "cifar10": ("CIFAR10", 3, 32),
}


def load_full(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dataset not in OFFICIAL:
        raise ValueError(dataset)
    cache = data_root() / f"{dataset}_full.npz"
    if cache.exists():
        stored = np.load(cache)
        return stored["train_x"], stored["train_y"], stored["test_x"], stored["test_y"]
    train_x, train_y, test_x, test_y = _download(dataset)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    return train_x, train_y, test_x, test_y


def _download(dataset: str):
    from torchvision import datasets, transforms

    name, channels, _size = OFFICIAL[dataset]
    root = data_root() / "torchvision"
    root.mkdir(parents=True, exist_ok=True)
    transform = transforms.ToTensor()
    cls = getattr(datasets, name)
    train = cls(root=root, train=True, download=True, transform=transform)
    test = cls(root=root, train=False, download=True, transform=transform)
    return _stack(train, channels), _labels(train), _stack(test, channels), _labels(test)


def _stack(dataset, channels: int) -> np.ndarray:
    images = np.empty((len(dataset), channels, dataset[0][0].shape[-2], dataset[0][0].shape[-1]), dtype=np.float32)
    # dataset[0] may download-trigger; use the tensor shape from the first item after a direct loop
    first, _ = dataset[0]
    images = np.empty((len(dataset),) + tuple(first.shape), dtype=np.float32)
    for i in range(len(dataset)):
        image, _label = dataset[i]
        images[i] = image.numpy()
    return images


def _labels(dataset) -> np.ndarray:
    labels = np.empty(len(dataset), dtype=np.int64)
    for i in range(len(dataset)):
        _image, label = dataset[i]
        labels[i] = int(label)
    return labels


def base_indices(dataset: str, n_train: int, n_test: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    _train_x, train_y, _test_x, test_y = load_full(dataset)
    path = split_dir() / f"{dataset}_seed{seed}_ntr{n_train}_nte{n_test}.npz"
    if path.exists():
        stored = np.load(path)
        return stored["train_idx"], stored["test_idx"]
    rng = np.random.default_rng(seed)
    n_train = min(int(n_train), len(train_y))
    n_test = min(int(n_test), len(test_y))
    train_idx = rng.choice(len(train_y), size=n_train, replace=False).astype(np.int64)
    test_idx = rng.choice(len(test_y), size=n_test, replace=False).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_idx=train_idx, test_idx=test_idx)
    return train_idx, test_idx


def training_subset(
    train_idx: np.ndarray,
    train_y: np.ndarray,
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


def materialize(dataset: str, protocol: dict, seed: int, sample_frac: float, imbalance: float, outlier_frac: float):
    spec = protocol["splits"][dataset]
    train_x, train_y, test_x, test_y = load_full(dataset)
    train_idx, test_idx = base_indices(dataset, spec["n_train"], spec["n_test"], seed)
    subset = training_subset(train_idx, train_y, sample_frac, imbalance, seed)
    x_train = apply_outliers(train_x[subset], outlier_frac, seed)
    y_train = train_y[subset]
    x_test = test_x[test_idx]
    y_test = test_y[test_idx]
    return x_train, y_train, x_test, y_test
