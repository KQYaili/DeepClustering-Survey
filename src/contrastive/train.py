"""SimCLR and Contrastive Clustering with a small CNN, not the papers' ResNets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from classical.geometry import assign_centers
from deep.common import (
    SmallCNN,
    as_tensor,
    clip_grads,
    count_parameters,
    input_channels,
    loss_still_falling,
    nt_xent,
    seed_everything,
    torch_device,
)


def _augment(image: torch.Tensor, dataset: str) -> torch.Tensor:
    _channels, height, width = image.shape
    scale = float(torch.empty(1).uniform_(0.6, 1.0))
    side = max(1, int(round(height * scale)))
    top = int(torch.randint(0, height - side + 1, ()))
    left = int(torch.randint(0, width - side + 1, ()))
    crop = image[:, top : top + side, left : left + side].unsqueeze(0)
    crop = F.interpolate(crop, size=(height, width), mode="bilinear", align_corners=False)[0]
    if dataset != "mnist" and float(torch.rand(())) < 0.5:
        crop = torch.flip(crop, dims=[2])
    if dataset == "cifar10":
        crop = (crop * float(torch.empty(1).uniform_(0.8, 1.2))).clamp(0, 1)
    return crop.contiguous()


class TwoView(torch.utils.data.Dataset):
    def __init__(self, images: np.ndarray, dataset: str):
        self.images = images
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = torch.from_numpy(np.ascontiguousarray(self.images[index]))
        return _augment(image, self.dataset), _augment(image, self.dataset)


class SimCLRNet(nn.Module):
    def __init__(self, dataset: str, backbone_dim: int, projection_dim: int):
        super().__init__()
        self.dataset = dataset
        self.backbone_dim = int(backbone_dim)
        self.backbone = SmallCNN(input_channels(dataset), self.backbone_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone_dim, int(projection_dim)),
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        return features, self.projector(features)


class CCNet(nn.Module):
    def __init__(self, dataset: str, backbone_dim: int, n_clusters: int):
        super().__init__()
        self.dataset = dataset
        self.backbone_dim = int(backbone_dim)
        self.n_clusters = int(n_clusters)
        self.backbone = SmallCNN(input_channels(dataset), self.backbone_dim)
        self.instance_head = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone_dim, 128),
        )
        self.cluster_head = nn.Linear(self.backbone_dim, self.n_clusters)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        assignments = torch.softmax(self.cluster_head(features), dim=1)
        return features, self.instance_head(features), assignments


def _loader(images: np.ndarray, dataset: str, batch: int):
    view = TwoView(images, dataset)
    batch = min(int(batch), len(images))
    if batch < 2:
        raise ValueError(f"too_few_samples for contrastive batch: {len(images)}")
    return torch.utils.data.DataLoader(
        view,
        batch_size=batch,
        shuffle=True,
        drop_last=len(images) >= batch * 2,
        num_workers=0,
    )


def _loop(net, images, dataset, epochs, optimizer, batch_fn, protocol, dev):
    planned = int(epochs)
    losses = []
    extended = False
    limit = planned
    epoch = 0
    while epoch < limit:
        net.train()
        total = 0.0
        seen = 0
        for view_a, view_b in _loader(images, dataset, int(protocol["batch_size"])):
            view_a = view_a.to(dev)
            view_b = view_b.to(dev)
            loss = batch_fn(view_a, view_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grads(net.parameters(), float(protocol["grad_clip"]))
            optimizer.step()
            total += float(loss.detach().item()) * len(view_a)
            seen += len(view_a)
        losses.append(total / max(seen, 1))
        epoch += 1
        rule = protocol["extension"]
        if (
            epoch == planned
            and not extended
            and rule["enabled"]
            and loss_still_falling(losses, rule["window"], rule["relative_drop"])
        ):
            limit = planned + int(rule["extra_epochs"])
            extended = True
        print(f"contrastive epoch {epoch}/{limit} loss {losses[-1]:.4f}", flush=True)
    return losses, extended, epoch


@torch.no_grad()
def _backbone_matrix(net, images: np.ndarray, dev: torch.device) -> np.ndarray:
    net.eval()
    chunks = []
    for start in range(0, len(images), 256):
        batch = as_tensor(images[start : start + 256]).to(dev)
        features = net.backbone(batch)
        chunks.append(features.cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


class EmbeddingKMeans:
    def __init__(self, net, centers: np.ndarray, kind: str):
        self.net = net
        self.centers = np.asarray(centers, dtype=np.float64)
        self.kind = kind

    def embeddings(self, images: np.ndarray) -> np.ndarray:
        dev = next(self.net.parameters()).device
        return _backbone_matrix(self.net, images, dev)

    def predict(self, images: np.ndarray) -> np.ndarray:
        if self.kind == "cc":
            dev = next(self.net.parameters()).device
            self.net.eval()
            chunks = []
            with torch.no_grad():
                for start in range(0, len(images), 256):
                    batch = as_tensor(images[start : start + 256]).to(dev)
                    _features, _projected, assignments = self.net(batch)
                    chunks.append(assignments.argmax(dim=1).cpu().numpy())
            return np.concatenate(chunks).astype(np.int64)
        return assign_centers(self.embeddings(images), self.centers)

    def reconstruction_sse(self, images: np.ndarray) -> None:
        return None

    def n_parameters(self) -> int:
        return count_parameters(self.net) + int(self.centers.size)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": self.kind,
                "state_dict": self.net.state_dict(),
                "centers": self.centers,
                "dataset": self.net.dataset,
                "backbone_dim": self.net.backbone_dim,
                "n_clusters": int(getattr(self.net, "n_clusters", len(self.centers))),
            },
            path,
        )


def _fit_kmeans(features: np.ndarray, k: int, seed: int, n_init: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    if len(features) < k:
        raise ValueError(f"too_few_samples: {len(features)} < k={k}")
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    model.fit(features)
    return model.cluster_centers_


def simclr_backbone_path(job: dict) -> Path:
    from runtime.paths import project_root, slug_float

    name = (
        f"simclr__{job['dataset']}__seed{job['seed']}"
        f"__sf{slug_float(job['sample_frac'])}"
        f"__imb{slug_float(job['imbalance'])}"
        f"__out{slug_float(job['outlier_frac'])}.pt"
    )
    return project_root() / "results" / "checkpoints" / "backbones" / name


def ensure_simclr(job, protocol, images, dev):
    cfg = protocol["simclr"]
    path = simclr_backbone_path(job)
    seed_everything(int(job["seed"]))
    net = SimCLRNet(job["dataset"], int(cfg["backbone_dim"]), int(cfg["projection_dim"])).to(dev)
    if path.exists():
        blob = torch.load(path, map_location=dev, weights_only=False)
        net.load_state_dict(blob["state_dict"])
        return net, {"train_loss": blob.get("train_loss", []), "extended": blob.get("extended", False), "epochs_ran": blob.get("epochs_ran", 0), "stopped_early": False, "backbone_cache_hit": True}
    optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    def batch_fn(view_a, view_b):
        _feat_a, proj_a = net(view_a)
        _feat_b, proj_b = net(view_b)
        return nt_xent(proj_a, proj_b, float(cfg["temperature"]))

    losses, extended, epochs = _loop(net, images, job["dataset"], cfg["epochs"], optimizer, batch_fn, protocol, dev)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": net.state_dict(), "train_loss": losses, "extended": extended, "epochs_ran": epochs, "dataset": job["dataset"], "backbone_dim": int(cfg["backbone_dim"])},
        path,
    )
    return net, {"train_loss": losses, "extended": extended, "epochs_ran": epochs, "stopped_early": False, "backbone_cache_hit": False}


def train_simclr(job, protocol, images, dev) -> tuple[EmbeddingKMeans, dict]:
    net, info = ensure_simclr(job, protocol, images, dev)
    features = _backbone_matrix(net, images, dev)
    centers = _fit_kmeans(features, int(job["k"]), int(job["seed"]), int(protocol["kmeans"]["n_init"]))
    return EmbeddingKMeans(net, centers, "simclr"), info


def train_cc(job, protocol, images, dev) -> tuple[EmbeddingKMeans, dict]:
    cfg = protocol["cc"]
    seed_everything(int(job["seed"]))
    net = CCNet(job["dataset"], int(cfg["backbone_dim"]), int(job["k"])).to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    def batch_fn(view_a, view_b):
        _fa, inst_a, clus_a = net(view_a)
        _fb, inst_b, clus_b = net(view_b)
        instance = nt_xent(inst_a, inst_b, float(cfg["instance_temperature"]))
        cluster = nt_xent(clus_a.T + 1e-8, clus_b.T + 1e-8, float(cfg["cluster_temperature"]))
        return instance + cluster

    losses, extended, epochs = _loop(net, images, job["dataset"], cfg["epochs"], optimizer, batch_fn, protocol, dev)
    features = _backbone_matrix(net, images, dev)
    centers = _fit_kmeans(features, int(job["k"]), int(job["seed"]), int(protocol["kmeans"]["n_init"]))
    info = {"train_loss": losses, "extended": extended, "epochs_ran": epochs, "stopped_early": False}
    return EmbeddingKMeans(net, centers, "cc"), info


def load_contrastive(path: Path, dev: torch.device | None = None) -> EmbeddingKMeans:
    dev = dev or torch_device()
    blob = torch.load(path, map_location=dev, weights_only=False)
    if blob["kind"] == "simclr":
        net = SimCLRNet(blob["dataset"], int(blob["backbone_dim"]), 128).to(dev)
    elif blob["kind"] == "cc":
        net = CCNet(blob["dataset"], int(blob["backbone_dim"]), int(blob["n_clusters"])).to(dev)
    else:
        raise ValueError(blob["kind"])
    net.load_state_dict(blob["state_dict"])
    return EmbeddingKMeans(net, blob["centers"], blob["kind"])
