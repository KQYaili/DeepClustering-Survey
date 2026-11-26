"""K-means, PCA + K-means, and a diagonal Gaussian mixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from classical.geometry import assign_centers, flatten, gmm_predict, pca_transform


class KMeansModel:
    def __init__(self, centers: np.ndarray):
        self.centers = np.asarray(centers, dtype=np.float64)

    def embeddings(self, images: np.ndarray) -> np.ndarray:
        return flatten(images).astype(np.float32)

    def predict(self, images: np.ndarray) -> np.ndarray:
        return assign_centers(flatten(images), self.centers)

    def reconstruction_sse(self, images: np.ndarray) -> None:
        return None

    def n_parameters(self) -> int:
        return int(self.centers.size)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": "kmeans", "centers": self.centers}, path)


class PCAKMeansModel:
    def __init__(self, mean: np.ndarray, components: np.ndarray, centers: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.components = np.asarray(components, dtype=np.float64)
        self.centers = np.asarray(centers, dtype=np.float64)

    def embeddings(self, images: np.ndarray) -> np.ndarray:
        return pca_transform(flatten(images), self.mean, self.components).astype(np.float32)

    def predict(self, images: np.ndarray) -> np.ndarray:
        return assign_centers(self.embeddings(images), self.centers)

    def reconstruction_sse(self, images: np.ndarray) -> None:
        return None

    def n_parameters(self) -> int:
        return int(self.components.size + self.centers.size)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "pca_kmeans",
                "mean": self.mean,
                "components": self.components,
                "centers": self.centers,
            },
            path,
        )


class GMMModel:
    def __init__(self, scaler_mean, scaler_scale, weights, means, variances):
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.variances = np.asarray(variances, dtype=np.float64)

    def _scale(self, images: np.ndarray) -> np.ndarray:
        return (flatten(images) - self.scaler_mean) / np.clip(self.scaler_scale, 1e-8, None)

    def embeddings(self, images: np.ndarray) -> np.ndarray:
        return self._scale(images).astype(np.float32)

    def predict(self, images: np.ndarray) -> np.ndarray:
        return gmm_predict(self._scale(images), self.weights, self.means, self.variances)

    def reconstruction_sse(self, images: np.ndarray) -> None:
        return None

    def n_parameters(self) -> int:
        return int(self.weights.size + self.means.size + self.variances.size)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "gmm",
                "scaler_mean": self.scaler_mean,
                "scaler_scale": self.scaler_scale,
                "weights": self.weights,
                "means": self.means,
                "variances": self.variances,
            },
            path,
        )


def fit_kmeans(images: np.ndarray, k: int, seed: int, n_init: int) -> KMeansModel:
    from sklearn.cluster import KMeans

    if len(images) < k:
        raise ValueError(f"too_few_samples: {len(images)} < k={k}")
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    model.fit(flatten(images))
    return KMeansModel(model.cluster_centers_)


def fit_pca_kmeans(images: np.ndarray, k: int, seed: int, n_init: int, n_components: int) -> PCAKMeansModel:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    flat = flatten(images)
    if len(flat) < k:
        raise ValueError(f"too_few_samples: {len(flat)} < k={k}")
    components = min(n_components, flat.shape[0] - 1, flat.shape[1])
    if components < 1:
        raise ValueError("too_few_samples for PCA")
    pca = PCA(n_components=components, random_state=seed)
    latent = pca.fit_transform(flat)
    kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    kmeans.fit(latent)
    return PCAKMeansModel(pca.mean_, pca.components_, kmeans.cluster_centers_)


def fit_gmm(images: np.ndarray, k: int, seed: int, covariance: str, n_init: int, max_iter: int, reg_covar: float):
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    flat = flatten(images)
    if len(flat) < k:
        raise ValueError(f"too_few_samples: {len(flat)} < k={k}")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(flat)
    mixture = GaussianMixture(
        n_components=k,
        covariance_type=covariance,
        n_init=n_init,
        max_iter=max_iter,
        reg_covar=reg_covar,
        random_state=seed,
    )
    mixture.fit(scaled)
    return GMMModel(
        scaler.mean_,
        scaler.scale_,
        mixture.weights_,
        mixture.means_,
        mixture.covariances_,
    )


def load_classical(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    kind = blob["kind"]
    if kind == "kmeans":
        return KMeansModel(blob["centers"])
    if kind == "pca_kmeans":
        return PCAKMeansModel(blob["mean"], blob["components"], blob["centers"])
    if kind == "gmm":
        return GMMModel(
            blob["scaler_mean"],
            blob["scaler_scale"],
            blob["weights"],
            blob["means"],
            blob["variances"],
        )
    raise ValueError(kind)
