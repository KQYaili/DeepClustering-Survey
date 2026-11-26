"""Output locations. Checkpoints stay local; metric JSON is the evidence record."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return ROOT


def data_root() -> Path:
    override = os.environ.get("DATA_ROOT")
    path = Path(override) if override else Path.home() / ".cache" / "dc-survey" / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_dir() -> Path:
    path = project_root() / "results" / "splits"
    path.mkdir(parents=True, exist_ok=True)
    return path


def slug_float(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace(".", "p").replace("-", "m")


def job_id(job: dict) -> str:
    return (
        f"{job['dataset']}__{job['method']}"
        f"__seed{job['seed']}"
        f"__k{job['k']}"
        f"__lat{job['latent_dim']}"
        f"__lam{slug_float(job['lambda_recon'])}"
        f"__pre{int(bool(job['pretrain']))}"
        f"__init{job['center_init']}"
        f"__fr{int(bool(job['freeze_encoder']))}"
        f"__sf{slug_float(job['sample_frac'])}"
        f"__imb{slug_float(job['imbalance'])}"
        f"__out{slug_float(job['outlier_frac'])}"
    )


def result_path(job: dict) -> Path:
    return project_root() / "results" / "raw" / job["experiment"] / f"{job_id(job)}.json"


def checkpoint_path(job: dict) -> Path:
    return project_root() / "results" / "checkpoints" / job["experiment"] / f"{job_id(job)}.pt"


def ae_cache_path(job: dict) -> Path:
    # Shared by AE, DEC and IDEC when the training subset matches.
    # The cache key intentionally ignores method, k, lambda, freeze and center init.
    name = (
        f"{job['dataset']}__ae"
        f"__seed{job['seed']}"
        f"__lat{job['latent_dim']}"
        f"__sf{slug_float(job['sample_frac'])}"
        f"__imb{slug_float(job['imbalance'])}"
        f"__out{slug_float(job['outlier_frac'])}.pt"
    )
    return project_root() / "results" / "checkpoints" / "ae_pretrain" / name


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.floating):
        return json_ready(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            json.dump(json_ready(payload), temp, indent=2, ensure_ascii=False)
        except BaseException:
            temp.close()
            temp_path.unlink(missing_ok=True)
            raise
    try:
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def retryable_infrastructure(error: str) -> bool:
    return any(message in error for message in (
        "File not found or corrupted.", "CUDA out of memory", "OutOfMemoryError",
        "ConnectionError:", "ConnectionResetError:", "TimeoutError:", "URLError:",
        "Temporary failure in name resolution", "No space left on device",
    ))


def pending_result(path: Path) -> bool:
    if not path.exists():
        return True
    record = read_json(path)
    return (record.get("status") == "failed" and int(record.get("attempts", 1)) < 2
            and retryable_infrastructure(record.get("error", "")))


def record_failure(destination: Path, job: dict, error: str) -> None:
    attempts = int(read_json(destination).get("attempts", 1)) + 1 if destination.exists() else 1
    record = {"status": "failed", "job": job, "error": error, "attempts": attempts}
    archive = destination.parents[2] / "attempts" / job["experiment"] / destination.stem
    write_json(archive / f"attempt-{attempts}.json", record)
    write_json(destination, record)
