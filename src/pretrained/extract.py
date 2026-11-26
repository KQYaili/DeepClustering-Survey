"""Frozen and trained feature extractors. DINOv2 and CLIP stay frozen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from deep.common import SmallCNN, as_tensor, embed_array, input_channels, seed_everything, torch_device
from runtime.paths import project_root, slug_float


EXTERNAL = {"dinov2", "clip"}


def feature_cache_path(job: dict, encoder: str) -> Path:
    name = (
        f"{job['dataset']}__{encoder}__seed{job['seed']}"
        f"__sf{slug_float(job['sample_frac'])}"
        f"__imb{slug_float(job['imbalance'])}"
        f"__out{slug_float(job['outlier_frac'])}.npz"
    )
    return project_root() / "results" / "cache" / "features" / name


def _l2(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norm, 1e-8, None)


def _imagenet_batch(images: np.ndarray, dev: torch.device) -> torch.Tensor:
    batch = as_tensor(images).to(dev)
    if batch.shape[1] == 1:
        batch = batch.repeat(1, 3, 1, 1)
    batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[None, :, None, None]
    return (batch - mean) / std


def _to_pil(image: np.ndarray):
    from PIL import Image

    array = np.transpose(image, (1, 2, 0))
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array.shape[2] == 1:
        return Image.fromarray(array[:, :, 0], mode="L").convert("RGB")
    return Image.fromarray(array, mode="RGB")


@torch.no_grad()
def _dinov2(images: np.ndarray, dev: torch.device) -> np.ndarray:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(dev)
    chunks = []
    for start in range(0, len(images), 32):
        batch = _imagenet_batch(images[start : start + 32], dev)
        chunks.append(model(batch).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


@torch.no_grad()
def _clip(images: np.ndarray, dev: torch.device) -> np.ndarray:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval().to(dev)
    chunks = []
    for start in range(0, len(images), 64):
        tensors = torch.stack([preprocess(_to_pil(image)) for image in images[start : start + 64]]).to(dev)
        chunks.append(model.encode_image(tensors).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def _random_cnn(job, images, dev) -> np.ndarray:
    seed_everything(int(job["seed"]))
    net = SmallCNN(input_channels(job["dataset"]), 128).to(dev)
    net.eval()
    return embed_array(net, images, 256, dev).astype(np.float32)


def _ae_features(job, protocol, images, dev) -> np.ndarray:
    from deep.joint import load_or_pretrain

    model, _loss, _ext, _hit = load_or_pretrain(job, protocol, images if job.get("_fit_images") is None else job["_fit_images"], dev)
    # The caller passes the matrix to embed. Pretraining must see the training images only.
    return embed_array(lambda batch: model.encode(batch), images, 256, dev).astype(np.float32)


def _simclr_features(job, protocol, images, dev) -> np.ndarray:
    from contrastive.train import _backbone_matrix, ensure_simclr

    train_images = job["_fit_images"]
    net, _info = ensure_simclr(job, protocol, train_images, dev)
    return _backbone_matrix(net, images, dev)


def extract_pair(job: dict, protocol: dict, encoder: str, x_train: np.ndarray, x_test: np.ndarray):
    path = feature_cache_path(job, encoder)
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        stored = np.load(path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["feature_cache_hit"] = True
        return stored["train"], stored["test"], meta
    dev = torch_device()
    started = torch.cuda.Event(enable_timing=True) if False else None
    import time

    clock = time.perf_counter()
    payload = dict(job)
    payload["_fit_images"] = x_train
    if encoder == "random_cnn":
        train = _random_cnn(payload, x_train, dev)
        test = _random_cnn(payload, x_test, dev)
    elif encoder == "ae":
        from deep.joint import load_or_pretrain
        from deep.common import embed_array as embed_fn

        model, losses, extended, hit = load_or_pretrain(payload, protocol, x_train, dev)
        train = embed_fn(lambda batch: model.encode(batch), x_train, 256, dev).astype(np.float32)
        test = embed_fn(lambda batch: model.encode(batch), x_test, 256, dev).astype(np.float32)
        meta_extra = {"ae_epochs": len(losses), "ae_cache_hit": hit, "extended": extended}
    elif encoder == "simclr":
        train = _simclr_features(payload, protocol, x_train, dev)
        test = _simclr_features(payload, protocol, x_test, dev)
        meta_extra = {}
    elif encoder == "dinov2":
        train = _dinov2(x_train, dev)
        test = _dinov2(x_test, dev)
        meta_extra = {}
    elif encoder == "clip":
        train = _clip(x_train, dev)
        test = _clip(x_test, dev)
        meta_extra = {}
    else:
        raise ValueError(encoder)
    if encoder in set(protocol["e04"]["l2_normalize_encoders"]):
        train = _l2(train).astype(np.float32)
        test = _l2(test).astype(np.float32)
    elapsed = time.perf_counter() - clock
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, train=train, test=test)
    meta = {
        "time_feature_sec": elapsed,
        "external_pretrain": encoder in EXTERNAL,
        "external_data": {
            "dinov2": "LVD-142M, frozen ViT-S/14",
            "clip": "OpenAI WIT, frozen ViT-B/32",
        }.get(encoder, "none"),
        "l2_normalized": encoder in set(protocol["e04"]["l2_normalize_encoders"]),
        "feature_cache_hit": False,
        "encoder": encoder,
    }
    meta.update(locals().get("meta_extra", {}))
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return train, test, meta
