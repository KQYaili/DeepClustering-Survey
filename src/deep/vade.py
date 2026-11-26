"""VaDE with a Gaussian-mixture prior. Reconstruction is per-image SSE, not Bernoulli BCE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from deep.common import (
    as_tensor,
    clip_grads,
    count_parameters,
    loss_still_falling,
    make_loader,
    per_image_sse,
    seed_everything,
    torch_device,
)


class VaDENet(nn.Module):
    def __init__(self, dataset: str, latent_dim: int, n_clusters: int, mlp_dims: list[int], logvar_bias: float):
        super().__init__()
        self.dataset = dataset
        self.latent_dim = int(latent_dim)
        self.n_clusters = int(n_clusters)
        self.mlp_dims = list(mlp_dims)
        if dataset in {"mnist", "fashion_mnist"}:
            hidden = list(mlp_dims[:-1])
            layers: list[nn.Module] = []
            for i in range(len(hidden) - 1):
                layers += [nn.Linear(hidden[i], hidden[i + 1]), nn.ReLU(inplace=True)]
            self.encoder = nn.Sequential(*layers)
            self.hidden_dim = hidden[-1]
            self.decoder_hidden = nn.Sequential(
                nn.Linear(self.latent_dim, hidden[-1]),
                nn.ReLU(inplace=True),
                nn.Linear(hidden[-1], hidden[-2]),
                nn.ReLU(inplace=True),
                nn.Linear(hidden[-2], hidden[-3]),
                nn.ReLU(inplace=True),
            )
            self.decoder_out = nn.Linear(hidden[-3], hidden[0])
            self.kind = "mlp"
        elif dataset == "cifar10":
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.ReLU(inplace=True),
            )
            self.hidden_dim = 128 * 4 * 4
            self.decoder_in = nn.Linear(self.latent_dim, self.hidden_dim)
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 3, 4, 2, 1),
            )
            self.kind = "cnn"
        else:
            raise ValueError(dataset)
        self.mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, float(logvar_bias))
        self.pi_logits = nn.Parameter(torch.zeros(self.n_clusters))
        self.mu_c = nn.Parameter(torch.zeros(self.n_clusters, self.latent_dim))
        self.logvar_c = nn.Parameter(torch.zeros(self.n_clusters, self.latent_dim))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "mlp":
            return self.encoder(x.reshape(x.shape[0], -1))
        return self.encoder(x).reshape(x.shape[0], -1)

    def encode_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.features(x)
        return self.mu(hidden), self.logvar(hidden).clamp(-8.0, 8.0)

    def decode(self, z: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if self.kind == "mlp":
            hidden = self.decoder_hidden(z)
            return torch.sigmoid(self.decoder_out(hidden)).reshape(reference.shape)
        hidden = self.decoder_in(z).reshape(z.shape[0], 128, 4, 4)
        return torch.sigmoid(self.decoder_conv(hidden))

    def elbo(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_stats(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.decode(z, x)
        recon_loss = per_image_sse(recon, x)
        logvar_c = self.logvar_c.clamp(-8.0, 8.0)
        var_c = logvar_c.exp()
        log_pi = torch.log_softmax(self.pi_logits, dim=0)
        delta = z.unsqueeze(1) - self.mu_c.unsqueeze(0)
        log_pz_c = -0.5 * (delta.pow(2) / var_c + logvar_c).sum(dim=-1)
        gamma = torch.softmax(log_pz_c + log_pi.unsqueeze(0), dim=1)
        kl_c = (gamma * (gamma.clamp_min(1e-12).log() - log_pi.unsqueeze(0))).sum(dim=1)
        var_z = logvar.exp().unsqueeze(1)
        kl_z = 0.5 * (
            gamma
            * (
                logvar_c
                + (var_z + (mu.unsqueeze(1) - self.mu_c).pow(2)) / var_c
                - 1.0
                - logvar.unsqueeze(1)
            ).sum(dim=-1)
        ).sum(dim=1)
        return recon_loss + (kl_c + kl_z).mean(), gamma

    def pretrain_loss(self, x: torch.Tensor) -> torch.Tensor:
        mu, _logvar = self.encode_stats(x)
        return per_image_sse(self.decode(mu, x), x)

    @torch.no_grad()
    def responsibilities(self, x: torch.Tensor) -> torch.Tensor:
        mu, _logvar = self.encode_stats(x)
        logvar_c = self.logvar_c.clamp(-8.0, 8.0)
        var_c = logvar_c.exp()
        log_pi = torch.log_softmax(self.pi_logits, dim=0)
        delta = mu.unsqueeze(1) - self.mu_c.unsqueeze(0)
        log_pz_c = -0.5 * (delta.pow(2) / var_c + logvar_c).sum(dim=-1)
        return torch.softmax(log_pz_c + log_pi.unsqueeze(0), dim=1)


def _gmm_parameters(net: VaDENet):
    return [net.pi_logits, net.mu_c, net.logvar_c]


def _set_requires(parameters, enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def _run_loop(net, images, epochs, optimizer, loss_fn, protocol, label: str, dev):
    planned = int(epochs)
    losses = []
    extended = False
    stopped_early = False
    limit = planned
    epoch = 0
    previous = None
    stop_rule = protocol["early_stop"]
    while epoch < limit:
        net.train()
        total = 0.0
        seen = 0
        for batch, _index in make_loader(images, int(protocol["batch_size"]), shuffle=True):
            batch = batch.to(dev)
            loss = loss_fn(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grads([p for p in net.parameters() if p.requires_grad], float(protocol["grad_clip"]))
            optimizer.step()
            total += float(loss.detach().item()) * len(batch)
            seen += len(batch)
        losses.append(total / max(seen, 1))
        labels = _predict_numpy(net, images, dev)
        delta = None if previous is None else float(np.mean(labels != previous))
        previous = labels
        epoch += 1
        if (
            label == "elbo"
            and stop_rule["enabled"]
            and epoch >= int(stop_rule["min_epoch"])
            and delta is not None
            and delta < float(stop_rule["assignment_tol"])
        ):
            stopped_early = True
            print(f"vade {label} epoch {epoch}/{limit} loss {losses[-1]:.4f} stop", flush=True)
            break
        rule = protocol["extension"]
        if (
            epoch == planned
            and not extended
            and rule["enabled"]
            and loss_still_falling(losses, rule["window"], rule["relative_drop"])
        ):
            limit = planned + int(rule["extra_epochs"])
            extended = True
        print(f"vade {label} epoch {epoch}/{limit} loss {losses[-1]:.4f}", flush=True)
    return losses, extended, stopped_early, epoch


@torch.no_grad()
def _predict_numpy(net: VaDENet, images: np.ndarray, dev: torch.device) -> np.ndarray:
    net.eval()
    chunks = []
    for start in range(0, len(images), 512):
        batch = as_tensor(images[start : start + 512]).to(dev)
        chunks.append(net.responsibilities(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(chunks).astype(np.int64)


@torch.no_grad()
def _collect_mu(net: VaDENet, images: np.ndarray, dev: torch.device) -> np.ndarray:
    net.eval()
    chunks = []
    for start in range(0, len(images), 512):
        batch = as_tensor(images[start : start + 512]).to(dev)
        mu, _logvar = net.encode_stats(batch)
        chunks.append(mu.cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def _init_mixture(net: VaDENet, images: np.ndarray, seed: int, dev: torch.device, n_init: int) -> None:
    from sklearn.cluster import KMeans

    latent = _collect_mu(net, images, dev)
    if len(latent) < net.n_clusters:
        raise ValueError(f"too_few_samples: {len(latent)} < k={net.n_clusters}")
    kmeans = KMeans(n_clusters=net.n_clusters, n_init=n_init, random_state=seed)
    labels = kmeans.fit_predict(latent)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=dev)
    net.mu_c.data.copy_(centers)
    variances = []
    for cluster in range(net.n_clusters):
        points = latent[labels == cluster]
        if len(points) < 2:
            variances.append(np.ones(net.latent_dim, dtype=np.float32))
        else:
            variances.append(points.var(axis=0).astype(np.float32) + 1e-4)
    net.logvar_c.data.copy_(torch.log(torch.tensor(np.stack(variances), device=dev)))
    counts = np.bincount(labels, minlength=net.n_clusters).astype(np.float32)
    proportions = counts / max(counts.sum(), 1.0)
    net.pi_logits.data.copy_(torch.log(torch.tensor(proportions, device=dev).clamp_min(1e-8)))


class VaDEModel:
    def __init__(self, net: VaDENet):
        self.net = net

    def embeddings(self, images: np.ndarray) -> np.ndarray:
        dev = next(self.net.parameters()).device
        return _collect_mu(self.net, images, dev)

    def predict(self, images: np.ndarray) -> np.ndarray:
        dev = next(self.net.parameters()).device
        return _predict_numpy(self.net, images, dev)

    def reconstruction_sse(self, images: np.ndarray) -> float:
        dev = next(self.net.parameters()).device
        self.net.eval()
        total = 0.0
        seen = 0
        with torch.no_grad():
            for batch, _index in make_loader(images, 512, shuffle=False):
                batch = batch.to(dev)
                mu, _logvar = self.net.encode_stats(batch)
                total += float(per_image_sse(self.net.decode(mu, batch), batch)) * len(batch)
                seen += len(batch)
        return total / max(seen, 1)

    def n_parameters(self) -> int:
        return count_parameters(self.net)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "vade",
                "state_dict": self.net.state_dict(),
                "dataset": self.net.dataset,
                "latent_dim": self.net.latent_dim,
                "n_clusters": self.net.n_clusters,
                "mlp_dims": self.net.mlp_dims,
            },
            path,
        )


def train_vade(job: dict, protocol: dict, images: np.ndarray, dev: torch.device) -> tuple[VaDEModel, dict]:
    cfg = protocol["vade"]
    seed_everything(int(job["seed"]))
    net = VaDENet(
        job["dataset"],
        int(job["latent_dim"]),
        int(job["k"]),
        protocol["ae"]["mlp_dims"],
        float(cfg["logvar_bias"]),
    ).to(dev)
    _set_requires(_gmm_parameters(net), False)
    net.logvar.weight.requires_grad_(False)
    net.logvar.bias.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=float(cfg["lr"]))
    pre_loss, pre_ext, _pre_stop, pre_epochs = _run_loop(
        net, images, cfg["pretrain_epochs"], optimizer, net.pretrain_loss, protocol, "pretrain", dev
    )
    for parameter in net.parameters():
        parameter.requires_grad_(True)
    _init_mixture(net, images, int(job["seed"]), dev, int(protocol["kmeans"]["n_init"]))
    seed_everything(int(job["seed"]) + int(protocol["finetune_seed_offset"]))
    optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg["lr"]))
    elbo_loss, elbo_ext, stopped, elbo_epochs = _run_loop(
        net, images, cfg["elbo_epochs"], optimizer, lambda batch: net.elbo(batch)[0], protocol, "elbo", dev
    )
    info = {
        "train_loss": elbo_loss,
        "pretrain_loss": pre_loss,
        "extended": pre_ext or elbo_ext,
        "stopped_early": stopped,
        "epochs_ran": elbo_epochs,
        "pretrain_epochs_ran": pre_epochs,
    }
    return VaDEModel(net), info


def load_vade(path: Path, dev: torch.device | None = None) -> VaDEModel:
    dev = dev or torch_device()
    blob = torch.load(path, map_location=dev, weights_only=False)
    net = VaDENet(blob["dataset"], blob["latent_dim"], blob["n_clusters"], blob["mlp_dims"], -2.0).to(dev)
    net.load_state_dict(blob["state_dict"])
    return VaDEModel(net)
