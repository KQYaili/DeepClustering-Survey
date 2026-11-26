"""DEC-style head on frozen features. The backbone is not updated."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from deep.common import (
    as_tensor,
    clip_grads,
    cluster_kl,
    init_centers_from_embeddings,
    make_loader,
    seed_everything,
    soft_assign,
    target_distribution,
)


class DecHead(nn.Module):
    def __init__(self, in_dim: int, n_clusters: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 10))
        self.centers = nn.Parameter(torch.zeros(n_clusters, 10))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


class FrozenDecModel:
    def __init__(self, module: DecHead):
        self.module = module

    def embeddings(self, features: np.ndarray) -> np.ndarray:
        dev = next(self.module.parameters()).device
        self.module.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(features), 1024):
                batch = as_tensor(features[start : start + 1024]).to(dev)
                chunks.append(self.module(batch).cpu().numpy())
        return np.concatenate(chunks).astype(np.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        dev = next(self.module.parameters()).device
        latent = torch.from_numpy(self.embeddings(features)).to(dev)
        with torch.no_grad():
            posterior = soft_assign(latent, self.module.centers, 1.0)
        return posterior.argmax(dim=1).cpu().numpy().astype(np.int64)

    def reconstruction_sse(self, features: np.ndarray) -> None:
        return None

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.module.parameters()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": "dec_head", "state_dict": self.module.state_dict(), "in_dim": self.module.proj[0].in_features, "k": int(self.module.centers.shape[0])}, path)


def train_dec_head(features: np.ndarray, protocol: dict, seed: int, k: int, dev: torch.device) -> tuple[FrozenDecModel, dict]:
    seed_everything(int(seed) + int(protocol["finetune_seed_offset"]) + 2003)
    module = DecHead(features.shape[1], int(k)).to(dev)
    with torch.no_grad():
        initial = module(as_tensor(features).to(dev)).cpu().numpy()
    centers = init_centers_from_embeddings(initial, int(k), int(seed), "kmeans", int(protocol["kmeans"]["n_init"]))
    module.centers.data.copy_(torch.tensor(centers, device=dev))
    optimizer = torch.optim.Adam(module.parameters(), lr=float(protocol["e04"]["dec_head_lr"]))
    planned = int(protocol["e04"]["dec_head_epochs"])
    losses = []
    for epoch in range(planned):
        with torch.no_grad():
            posterior = soft_assign(module(as_tensor(features).to(dev)), module.centers.detach(), 1.0)
            target = target_distribution(posterior).detach()
        module.train()
        total = 0.0
        seen = 0
        for batch, index in make_loader(features.astype(np.float32), int(protocol["batch_size"]), shuffle=True):
            batch = batch.to(dev)
            predicted = soft_assign(module(batch), module.centers, 1.0)
            loss = cluster_kl(target[index], predicted)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grads(module.parameters(), float(protocol["grad_clip"]))
            optimizer.step()
            total += float(loss.detach().item()) * len(batch)
            seen += len(batch)
        losses.append(total / max(seen, 1))
        print(f"dec-head epoch {epoch + 1}/{planned} loss {losses[-1]:.4f}", flush=True)
    return FrozenDecModel(module), {"train_loss": losses, "epochs_ran": planned, "extended": False, "stopped_early": False}
