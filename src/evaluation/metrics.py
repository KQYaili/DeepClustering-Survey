"""Clustering metrics. This is the only module that reads ground-truth labels."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def evaluation_labels(dataset: str, indices: np.ndarray) -> np.ndarray:
    """Read the official answers for these indices only at final scoring."""
    from runtime.paths import data_root

    cache = data_root() / f"{dataset}_full.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as arrays:
            return arrays["test_y"][indices]
    from datasets.data import official_dataset

    return np.asarray(official_dataset(dataset, train=False).targets, dtype=np.int64)[indices]


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("label and prediction lengths differ")
    if y_true.size == 0:
        return float("nan")
    n_true = int(y_true.max()) + 1 if y_true.size else 0
    n_pred = int(y_pred.max()) + 1 if y_pred.size else 0
    n = max(n_true, n_pred)
    contingency = np.zeros((n, n), dtype=np.int64)
    np.add.at(contingency, (y_pred, y_true), 1)
    row, col = linear_sum_assignment(contingency.max() - contingency)
    return float(contingency[row, col].sum() / y_pred.size)


def normalized_mutual_info(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic"))


def adjusted_rand(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(adjusted_rand_score(y_true, y_pred))


def safe_silhouette(embeddings: np.ndarray, y_pred: np.ndarray, max_points: int, seed: int) -> float | None:
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_pred)) < 2 or len(y_pred) < 3:
        return None
    n = embeddings.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        take = rng.choice(n, size=max_points, replace=False)
        embeddings = embeddings[take]
        y_pred = y_pred[take]
        if len(np.unique(y_pred)) < 2:
            return None
    try:
        value = silhouette_score(embeddings, y_pred, metric="euclidean")
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def cluster_balance(y_pred: np.ndarray, k: int) -> dict:
    counts = np.bincount(np.asarray(y_pred, dtype=np.int64), minlength=k)
    total = max(int(counts.sum()), 1)
    frac = counts / total
    positive = frac[frac > 0]
    entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
    max_entropy = float(np.log(k)) if k > 1 else 0.0
    return {
        "counts": counts.astype(int).tolist(),
        "max_fraction": float(frac.max()) if frac.size else 1.0,
        "n_nonempty": int((counts > 0).sum()),
        "entropy": entropy,
        "entropy_norm": float(entropy / max_entropy) if max_entropy > 0 else 0.0,
    }


def evaluate_partition(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    embeddings: np.ndarray,
    k: int,
    silhouette_points: int,
    seed: int,
) -> dict:
    balance = cluster_balance(y_pred, k)
    return {
        "acc": clustering_accuracy(y_true, y_pred),
        "nmi": normalized_mutual_info(y_true, y_pred),
        "ari": adjusted_rand(y_true, y_pred),
        "silhouette": safe_silhouette(embeddings, y_pred, silhouette_points, seed),
        **balance,
    }
