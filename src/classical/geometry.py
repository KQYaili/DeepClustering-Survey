"""NumPy assignments for saved classical models."""

from __future__ import annotations

import math

import numpy as np


def assign_centers(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    centers64 = np.asarray(centers, dtype=np.float64)
    predictions = np.empty(len(x64), dtype=np.int64)
    x_norm = np.sum(x64 * x64, axis=1)
    center_norm = np.sum(centers64 * centers64, axis=1)
    for start in range(0, len(x64), 4096):
        stop = min(start + 4096, len(x64))
        part = x64[start:stop]
        distance = x_norm[start:stop, None] + center_norm[None, :] - 2.0 * (part @ centers64.T)
        predictions[start:stop] = np.argmin(distance, axis=1)
    return predictions


def pca_transform(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - mean) @ components.T


def gmm_predict(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    means64 = np.asarray(means, dtype=np.float64)
    var = np.clip(np.asarray(variances, dtype=np.float64), 1e-8, None)
    log_weights = np.log(np.clip(weights, 1e-12, None))
    dim = x64.shape[1]
    constant = dim * math.log(2.0 * math.pi)
    log_det = np.sum(np.log(var), axis=1)
    scores = []
    for start in range(0, len(x64), 2048):
        part = x64[start : start + 2048]
        delta = part[:, None, :] - means64[None, :, :]
        mahalanobis = np.sum((delta * delta) / var[None, :, :], axis=2)
        log_prob = log_weights[None, :] - 0.5 * (log_det[None, :] + mahalanobis + constant)
        scores.append(np.argmax(log_prob, axis=1))
    return np.concatenate(scores).astype(np.int64)


def flatten(images: np.ndarray) -> np.ndarray:
    return np.asarray(images, dtype=np.float64).reshape(len(images), -1)
