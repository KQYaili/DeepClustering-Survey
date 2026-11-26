"""Reload a saved clusterer. Feature-space heads are not applied to raw pixels."""

from __future__ import annotations

from pathlib import Path

import torch

from classical.models import load_classical
from contrastive.train import load_contrastive
from deep.joint import load_joint
from deep.vade import load_vade
from pretrained.heads import DecHead, FrozenDecModel


def load_any(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    kind = blob["kind"]
    if kind in {"kmeans", "pca_kmeans", "gmm"}:
        return load_classical(path)
    if kind == "joint":
        return load_joint(path)
    if kind == "vade":
        return load_vade(path)
    if kind in {"simclr", "cc"}:
        return load_contrastive(path)
    if kind == "dec_head":
        module = DecHead(int(blob["in_dim"]), int(blob["k"]))
        module.load_state_dict(blob["state_dict"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module.to(device)
        return FrozenDecModel(module)
    raise ValueError(kind)
