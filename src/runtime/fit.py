"""Fit one configured method. This module does not read ground-truth labels."""

from __future__ import annotations

import time

import numpy as np
import torch

from classical.models import fit_gmm, fit_kmeans, fit_pca_kmeans
from deep.common import torch_device
from pretrained.extract import extract_pair

IMAGE_METHODS = {
    "kmeans",
    "pca_kmeans",
    "gmm",
    "ae_kmeans",
    "dec",
    "idec",
    "vade",
    "simclr_kmeans",
    "cc",
}


def _split_feature_method(method: str):
    for suffix in ("_dec_head", "_gmm", "_kmeans"):
        if method.endswith(suffix):
            return method[: -len(suffix)], suffix[1:]
    raise ValueError(method)


def _peak_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def _finish(model, x_eval, train_seconds: float, info: dict, feature_meta: dict | None):
    started = time.perf_counter()
    prediction = model.predict(x_eval)
    infer_seconds = time.perf_counter() - started
    embeddings = model.embeddings(x_eval)
    recon = model.reconstruction_sse(x_eval)
    payload = {
        "prediction": prediction,
        "embeddings": embeddings,
        "recon_sse": recon,
        "n_parameters": int(model.n_parameters()),
        "time_train_sec": train_seconds,
        "time_infer_sec": infer_seconds,
        "peak_gpu_mb": _peak_mb(),
        "history": info,
        "external_pretrain": False,
        "time_feature_sec": 0.0,
    }
    if feature_meta:
        payload["external_pretrain"] = bool(feature_meta.get("external_pretrain", False))
        payload["time_feature_sec"] = float(feature_meta.get("time_feature_sec", 0.0))
        payload["external_data"] = feature_meta.get("external_data", "none")
        payload["feature_cache_hit"] = bool(feature_meta.get("feature_cache_hit", False))
    return model, payload


def fit_job(job: dict, protocol: dict, x_train: np.ndarray, x_test: np.ndarray):
    dev = torch_device()
    _peak_reset()
    method = job["method"]
    started = time.perf_counter()
    if method in IMAGE_METHODS:
        model, info = _fit_image(job, protocol, x_train, dev)
        train_seconds = time.perf_counter() - started
        model_out, payload = _finish(model, x_test, train_seconds, info, None)
        return model_out, payload
    encoder, clusterer = _split_feature_method(method)
    _train_features, test_features, meta = extract_pair(job, protocol, encoder, x_train, x_test)
    # Reload train features from the returned pair. extract_pair returns both.
    train_features = _train_features
    cluster_started = time.perf_counter()
    model, info = _fit_on_features(job, protocol, train_features, clusterer, dev)
    train_seconds = time.perf_counter() - cluster_started
    model_out, payload = _finish(model, test_features, train_seconds, info, meta)
    payload["time_total_sec"] = time.perf_counter() - started
    return model_out, payload


def _fit_image(job, protocol, x_train, dev):
    method = job["method"]
    if method == "kmeans":
        model = fit_kmeans(x_train, int(job["k"]), int(job["seed"]), int(protocol["kmeans"]["n_init"]))
        return model, {"epochs_ran": 0, "extended": False, "stopped_early": False, "train_loss": []}
    if method == "pca_kmeans":
        model = fit_pca_kmeans(
            x_train,
            int(job["k"]),
            int(job["seed"]),
            int(protocol["kmeans"]["n_init"]),
            int(protocol["pca"]["n_components"]),
        )
        return model, {"epochs_ran": 0, "extended": False, "stopped_early": False, "train_loss": []}
    if method == "gmm":
        cfg = protocol["gmm"]
        model = fit_gmm(
            x_train,
            int(job["k"]),
            int(job["seed"]),
            cfg["covariance"],
            int(cfg["n_init"]),
            int(cfg["max_iter"]),
            float(cfg["reg_covar"]),
        )
        return model, {"epochs_ran": 0, "extended": False, "stopped_early": False, "train_loss": []}
    if method in {"ae_kmeans", "dec", "idec"}:
        from deep.joint import train_joint

        return train_joint(job, protocol, x_train, dev)
    if method == "vade":
        from deep.vade import train_vade

        return train_vade(job, protocol, x_train, dev)
    if method == "simclr_kmeans":
        from contrastive.train import train_simclr

        return train_simclr(job, protocol, x_train, dev)
    if method == "cc":
        from contrastive.train import train_cc

        return train_cc(job, protocol, x_train, dev)
    raise ValueError(method)


def _fit_on_features(job, protocol, features, clusterer, dev):
    if clusterer == "kmeans":
        model = fit_kmeans(features, int(job["k"]), int(job["seed"]), int(protocol["kmeans"]["n_init"]))
        return model, {"epochs_ran": 0, "extended": False, "stopped_early": False, "train_loss": []}
    if clusterer == "gmm":
        cfg = protocol["gmm"]
        model = fit_gmm(
            features,
            int(job["k"]),
            int(job["seed"]),
            cfg["covariance"],
            int(cfg["n_init"]),
            int(cfg["max_iter"]),
            float(cfg["reg_covar"]),
        )
        return model, {"epochs_ran": 0, "extended": False, "stopped_early": False, "train_loss": []}
    if clusterer == "dec_head":
        from pretrained.heads import train_dec_head

        return train_dec_head(features, protocol, int(job["seed"]), int(job["k"]), dev)
    raise ValueError(clusterer)
