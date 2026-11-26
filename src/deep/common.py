"""Networks and unsupervised losses. This module never reads labels."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).float()


def per_image_sse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    diff = (x_hat - x).reshape(x.shape[0], -1)
    return diff.pow(2).sum(dim=1).mean()


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


class MLPAutoencoder(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        self.dims = [int(v) for v in dims]
        self.encoder = _mlp(self.dims)
        self.decoder = _mlp(list(reversed(self.dims)), final_sigmoid=False)
        self.latent_dim = self.dims[-1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.reshape(x.shape[0], -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        flat = x_like_decode(self.decoder(z))
        return flat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z).reshape(x.shape)
        return z, recon


def x_like_decode(decoded: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(decoded)


def _mlp(dims: list[int], final_sigmoid: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU(inplace=True))
    if final_sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class CNNAutoencoder(nn.Module):
    """32x32 convolutional autoencoder. Latent size is independent of the MLP stack."""

    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.in_channels = int(in_channels)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.fc_enc = nn.Linear(128 * 4 * 4, self.latent_dim)
        self.fc_dec = nn.Linear(self.latent_dim, 128 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        return self.fc_enc(h.reshape(x.shape[0], -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).reshape(z.shape[0], 128, 4, 4)
        return torch.sigmoid(self.decoder_conv(h))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z)


def build_autoencoder(dataset: str, latent_dim: int, mlp_dims: list[int]) -> nn.Module:
    if dataset in {"mnist", "fashion_mnist"}:
        dims = list(mlp_dims)
        dims[-1] = int(latent_dim)
        if dims[0] != 784:
            raise ValueError("MLP autoencoder is only for 28x28 grayscale images")
        return MLPAutoencoder(dims)
    if dataset == "cifar10":
        return CNNAutoencoder(3, latent_dim)
    raise ValueError(dataset)


def encoder_parameters(model: nn.Module) -> list[nn.Parameter]:
    if isinstance(model, MLPAutoencoder):
        return list(model.encoder.parameters())
    if isinstance(model, CNNAutoencoder):
        return list(model.encoder_conv.parameters()) + list(model.fc_enc.parameters())
    raise TypeError(type(model))


def decoder_parameters(model: nn.Module) -> list[nn.Parameter]:
    if isinstance(model, MLPAutoencoder):
        return list(model.decoder.parameters())
    if isinstance(model, CNNAutoencoder):
        return list(model.fc_dec.parameters()) + list(model.decoder_conv.parameters())
    raise TypeError(type(model))


def soft_assign(z: torch.Tensor, centers: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    dist = torch.cdist(z, centers).pow(2)
    q = (1.0 + dist / alpha).pow(-(alpha + 1.0) / 2.0)
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    freq = q.sum(dim=0, keepdim=True).clamp_min(1e-12)
    p = q.pow(2) / freq
    return p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)


def cluster_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(dim=1).mean()


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    n = z1.shape[0]
    if n < 2:
        raise ValueError("NT-Xent needs at least 2 samples")
    z = torch.cat([z1, z2], dim=0)
    logits = (z @ z.T) / temperature
    logits = logits.masked_fill(torch.eye(2 * n, device=z.device, dtype=torch.bool), -1e9)
    labels = torch.cat(
        [torch.arange(n, 2 * n, device=z.device), torch.arange(0, n, device=z.device)],
        dim=0,
    )
    return F.cross_entropy(logits, labels)


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.out_dim = int(out_dim)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x).reshape(x.shape[0], -1))


def input_channels(dataset: str) -> int:
    if dataset in {"mnist", "fashion_mnist"}:
        return 1
    if dataset == "cifar10":
        return 3
    raise ValueError(dataset)


@torch.no_grad()
def embed_array(encode, x: np.ndarray, batch: int, dev: torch.device) -> np.ndarray:
    chunks = []
    for start in range(0, len(x), batch):
        batch_x = as_tensor(x[start : start + batch]).to(dev)
        chunks.append(encode(batch_x).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def make_loader(x: np.ndarray, batch: int, shuffle: bool, drop_last: bool = False):
    index = torch.arange(len(x))
    dataset = torch.utils.data.TensorDataset(as_tensor(x), index)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch, max(len(x), 1)),
        shuffle=shuffle,
        drop_last=drop_last and len(x) >= batch * 2,
    )


def loss_still_falling(losses: list[float], window: int, relative_drop: float) -> bool:
    if len(losses) < window + 1:
        return False
    previous = losses[-(window + 1)]
    current = losses[-1]
    if abs(previous) < 1e-12:
        return False
    return (previous - current) / abs(previous) > relative_drop


def clip_grads(parameters, max_norm: float) -> None:
    torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def init_centers_from_embeddings(z: np.ndarray, k: int, seed: int, mode: str) -> np.ndarray:
    if len(z) < k:
        raise ValueError(f"cannot init {k} centers from {len(z)} embeddings")
    if mode == "random":
        rng = np.random.default_rng(seed)
        return z[rng.choice(len(z), size=k, replace=False)].astype(np.float32)
    if mode != "kmeans":
        raise ValueError(mode)
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, n_init=20, random_state=seed)
    model.fit(z)
    return model.cluster_centers_.astype(np.float32)
