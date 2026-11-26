"""AE pretraining and DEC/IDEC finetuning. No ground-truth labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from deep.common import (
    as_tensor,
    build_autoencoder,
    clip_grads,
    cluster_kl,
    count_parameters,
    decoder_parameters,
    embed_array,
    encoder_parameters,
    init_centers_from_embeddings,
    loss_still_falling,
    make_loader,
    per_image_sse,
    seed_everything,
    soft_assign,
    target_distribution,
    torch_device,
)
from runtime.paths import ae_cache_path


def _epoch_plan(planned: int, losses: list[float], extended: bool, stopped_early: bool, epoch: int, protocol: dict):
    if stopped_early or extended or epoch != planned:
        return planned, extended
    rule = protocol["extension"]
    if rule["enabled"] and loss_still_falling(losses, rule["window"], rule["relative_drop"]):
        return planned + int(rule["extra_epochs"]), True
    return planned, extended


def train_reconstruction(model, images, protocol: dict, dev: torch.device):
    cfg = protocol["ae"]
    planned = int(cfg["epochs"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    losses: list[float] = []
    extended = False
    limit = planned
    epoch = 0
    while epoch < limit:
        model.train()
        total = 0.0
        seen = 0
        for batch, _index in make_loader(images, int(protocol["batch_size"]), shuffle=True):
            batch = batch.to(dev)
            _latent, recon = model(batch)
            if recon.shape != batch.shape:
                recon = recon.reshape(batch.shape)
            loss = per_image_sse(recon, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grads(model.parameters(), float(protocol["grad_clip"]))
            optimizer.step()
            total += float(loss.detach().item()) * len(batch)
            seen += len(batch)
        losses.append(total / max(seen, 1))
        epoch += 1
        limit, extended = _epoch_plan(planned, losses, extended, False, epoch, protocol)
        print(f"ae epoch {epoch}/{limit} sse {losses[-1]:.4f}", flush=True)
    return losses, extended, epoch


def load_or_pretrain(job: dict, protocol: dict, images: np.ndarray, dev: torch.device):
    seed_everything(int(job["seed"]))
    model = build_autoencoder(job["dataset"], int(job["latent_dim"]), protocol["ae"]["mlp_dims"]).to(dev)
    if not job["pretrain"]:
        return model, [], False, False
    path = ae_cache_path(job)
    if path.exists():
        blob = torch.load(path, map_location=dev, weights_only=False)
        model.load_state_dict(blob["state_dict"])
        return model, list(blob.get("train_loss", [])), bool(blob.get("extended", False)), True
    losses, extended, epochs_ran = train_reconstruction(model, images, protocol, dev)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "train_loss": losses,
            "extended": extended,
            "epochs_ran": epochs_ran,
            "dataset": job["dataset"],
            "latent_dim": int(job["latent_dim"]),
        },
        path,
    )
    return model, losses, extended, False


class JointModel:
    def __init__(
        self,
        model,
        centers: torch.Tensor,
        alpha: float,
        dataset: str,
        latent_dim: int,
        mlp_dims: list[int],
        assignment: str = "student",
    ):
        self.model = model
        self.centers = centers
        self.alpha = float(alpha)
        self.dataset = dataset
        self.latent_dim = int(latent_dim)
        self.mlp_dims = list(mlp_dims)
        self.assignment = assignment

    def _encode(self, images: np.ndarray) -> np.ndarray:
        dev = next(self.model.parameters()).device
        self.model.eval()
        return embed_array(lambda batch: self.model.encode(batch), images, 512, dev)

    def embeddings(self, images: np.ndarray) -> np.ndarray:
        return self._encode(images).astype(np.float32)

    def predict(self, images: np.ndarray) -> np.ndarray:
        dev = next(self.model.parameters()).device
        latent_np = self._encode(images)
        if self.assignment == "euclidean":
            from classical.geometry import assign_centers

            return assign_centers(latent_np, self.centers.detach().cpu().numpy())
        latent = torch.from_numpy(latent_np).to(dev)
        with torch.no_grad():
            posterior = soft_assign(latent, self.centers, self.alpha)
        return posterior.argmax(dim=1).cpu().numpy().astype(np.int64)

    def reconstruction_sse(self, images: np.ndarray) -> float:
        dev = next(self.model.parameters()).device
        self.model.eval()
        total = 0.0
        seen = 0
        with torch.no_grad():
            for batch, _index in make_loader(images, 512, shuffle=False):
                batch = batch.to(dev)
                _latent, recon = self.model(batch)
                if recon.shape != batch.shape:
                    recon = recon.reshape(batch.shape)
                total += float(per_image_sse(recon, batch)) * len(batch)
                seen += len(batch)
        return total / max(seen, 1)

    def n_parameters(self) -> int:
        return count_parameters(self.model) + int(self.centers.numel())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "joint",
                "state_dict": self.model.state_dict(),
                "centers": self.centers.detach().cpu(),
                "alpha": self.alpha,
                "dataset": self.dataset,
                "latent_dim": self.latent_dim,
                "mlp_dims": self.mlp_dims,
                "assignment": self.assignment,
            },
            path,
        )


def train_joint(job: dict, protocol: dict, images: np.ndarray, dev: torch.device) -> tuple[JointModel, dict]:
    model, pretrain_loss, pretrain_extended, cache_hit = load_or_pretrain(job, protocol, images, dev)
    lambda_recon = float(job["lambda_recon"]) if job["method"] == "idec" else 0.0
    if job["method"] == "ae_kmeans":
        from sklearn.cluster import KMeans

        seed_everything(int(job["seed"]) + int(protocol["finetune_seed_offset"]))
        latent = embed_array(lambda batch: model.encode(batch), images, int(protocol["batch_size"]), dev)
        if len(latent) < int(job["k"]):
            raise ValueError(f"too_few_samples: {len(latent)} < k={job['k']}")
        kmeans = KMeans(n_clusters=int(job["k"]), n_init=int(protocol["kmeans"]["n_init"]), random_state=int(job["seed"]))
        kmeans.fit(latent)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=dev)
        fitted = JointModel(
            model,
            centers,
            float(protocol["dec"]["alpha"]),
            job["dataset"],
            int(job["latent_dim"]),
            protocol["ae"]["mlp_dims"],
            assignment="euclidean",
        )
        info = {
            "train_loss": pretrain_loss,
            "extended": pretrain_extended,
            "ae_cache_hit": cache_hit,
            "epochs_ran": len(pretrain_loss),
            "stopped_early": False,
        }
        return fitted, info

    cfg = protocol["dec"] if job["method"] == "dec" else protocol["idec"]
    seed_everything(int(job["seed"]) + int(protocol["finetune_seed_offset"]))
    with torch.no_grad():
        initial = embed_array(lambda batch: model.encode(batch), images, int(protocol["batch_size"]), dev)
    centers_np = init_centers_from_embeddings(initial, int(job["k"]), int(job["seed"]), job["center_init"])
    centers = torch.nn.Parameter(torch.tensor(centers_np, dtype=torch.float32, device=dev))
    for parameter in encoder_parameters(model):
        parameter.requires_grad_(not bool(job["freeze_encoder"]))
    parameters = [centers]
    if not job["freeze_encoder"]:
        parameters.extend(encoder_parameters(model))
    if lambda_recon > 0:
        parameters.extend(decoder_parameters(model))
    optimizer = torch.optim.Adam(parameters, lr=float(cfg["lr"]))
    alpha = float(protocol["dec"]["alpha"])
    planned = int(cfg["epochs"])
    losses: list[float] = []
    extended = False
    stopped_early = False
    limit = planned
    epoch = 0
    previous = None
    stop_rule = protocol["early_stop"]

    def refresh():
        model.eval()
        with torch.no_grad():
            latent = embed_array(lambda batch: model.encode(batch), images, int(protocol["batch_size"]), dev)
            latent_t = torch.from_numpy(latent).to(dev)
            posterior = soft_assign(latent_t, centers.detach(), alpha)
            target = target_distribution(posterior).detach()
            labels = posterior.argmax(dim=1).cpu().numpy()
        return target, labels

    while epoch < limit:
        target, _labels = refresh()
        model.train()
        total = 0.0
        seen = 0
        for batch, index in make_loader(images, int(protocol["batch_size"]), shuffle=True):
            batch = batch.to(dev)
            latent = model.encode(batch)
            posterior = soft_assign(latent, centers, alpha)
            loss = cluster_kl(target[index], posterior)
            if lambda_recon > 0:
                recon = model.decode(latent)
                if recon.shape != batch.shape:
                    recon = recon.reshape(batch.shape)
                loss = loss + lambda_recon * per_image_sse(recon, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grads(parameters, float(protocol["grad_clip"]))
            optimizer.step()
            total += float(loss.detach().item()) * len(batch)
            seen += len(batch)
        losses.append(total / max(seen, 1))
        _target, labels = refresh()
        delta = None
        if previous is not None:
            delta = float(np.mean(labels != previous))
        previous = labels
        epoch += 1
        if (
            stop_rule["enabled"]
            and epoch >= int(stop_rule["min_epoch"])
            and delta is not None
            and delta < float(stop_rule["assignment_tol"])
        ):
            stopped_early = True
            print(f"joint epoch {epoch}/{limit} loss {losses[-1]:.4f} delta {delta:.5f} stop", flush=True)
            break
        limit, extended = _epoch_plan(planned, losses, extended, False, epoch, protocol)
        delta_text = "na" if delta is None else f"{delta:.5f}"
        print(f"joint epoch {epoch}/{limit} loss {losses[-1]:.4f} delta {delta_text}", flush=True)
    fitted = JointModel(
        model,
        centers.detach(),
        alpha,
        job["dataset"],
        int(job["latent_dim"]),
        protocol["ae"]["mlp_dims"],
        assignment="student",
    )
    info = {
        "train_loss": losses,
        "pretrain_loss": pretrain_loss,
        "extended": extended or pretrain_extended,
        "ae_cache_hit": cache_hit,
        "epochs_ran": epoch,
        "stopped_early": stopped_early,
        "lambda_recon": lambda_recon,
    }
    return fitted, info


def load_joint(path: Path, dev: torch.device | None = None) -> JointModel:
    dev = dev or torch_device()
    blob = torch.load(path, map_location=dev, weights_only=False)
    model = build_autoencoder(blob["dataset"], int(blob["latent_dim"]), blob["mlp_dims"]).to(dev)
    model.load_state_dict(blob["state_dict"])
    centers = blob["centers"].to(dev)
    return JointModel(
        model,
        centers,
        float(blob["alpha"]),
        blob["dataset"],
        int(blob["latent_dim"]),
        blob["mlp_dims"],
        assignment=blob.get("assignment", "student"),
    )
