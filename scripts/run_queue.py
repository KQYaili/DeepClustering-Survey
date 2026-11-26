"""Serial experiment queue. A finished job is not repeated. Failures are kept."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runtime.paths import job_id, project_root, read_json, result_path  # noqa: E402


def load_protocol() -> dict:
    return yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text(encoding="utf-8"))


def _base(**kwargs) -> dict:
    job = {
        "experiment": "E01",
        "dataset": "mnist",
        "method": "kmeans",
        "seed": 0,
        "k": 10,
        "latent_dim": 10,
        "lambda_recon": 1.0,
        "pretrain": True,
        "center_init": "kmeans",
        "freeze_encoder": False,
        "sample_frac": 1.0,
        "imbalance": 1.0,
        "outlier_frac": 0.0,
        "required": True,
        "priority": 50,
    }
    job.update(kwargs)
    if job["method"] == "dec":
        job["lambda_recon"] = 0.0
    return job


def build_jobs(protocol: dict) -> list[dict]:
    jobs = []
    seeds = [int(s) for s in protocol["seeds"]]
    image_methods = [
        "kmeans",
        "gmm",
        "pca_kmeans",
        "ae_kmeans",
        "dec",
        "idec",
        "vade",
        "simclr_kmeans",
        "cc",
    ]
    fast = {"kmeans", "gmm", "pca_kmeans", "ae_kmeans", "dec", "idec"}
    for dataset in ("mnist", "fashion_mnist"):
        for method in image_methods:
            for seed in seeds:
                jobs.append(
                    _base(
                        experiment="E01",
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        priority=10 if method in fast else 20,
                    )
                )
    for method in image_methods:
        for seed in seeds:
            jobs.append(
                _base(
                    experiment="E01",
                    dataset="cifar10",
                    method=method,
                    seed=seed,
                    priority=15 if method in {"kmeans", "pca_kmeans", "gmm"} else 40,
                )
            )
    fashion = protocol["e02"]
    for value in fashion["lambdas"]:
        jobs.append(
            _base(
                experiment="E02",
                dataset=fashion["dataset"],
                method="idec",
                seed=int(fashion["seed"]),
                lambda_recon=float(value),
                priority=30,
            )
        )
    for dim in fashion["latent_dims"]:
        jobs.append(
            _base(
                experiment="E02",
                dataset=fashion["dataset"],
                method="idec",
                seed=int(fashion["seed"]),
                latent_dim=int(dim),
                lambda_recon=1.0,
                priority=30,
            )
        )
    for name, extra in (
        ("no_pretrain", {"pretrain": False}),
        ("random_init", {"center_init": "random"}),
        ("freeze", {"freeze_encoder": True}),
    ):
        jobs.append(
            _base(
                experiment="E02",
                dataset=fashion["dataset"],
                method="idec",
                seed=int(fashion["seed"]),
                lambda_recon=1.0,
                priority=32,
                **extra,
            )
        )
    robust = protocol["e03"]
    for method in robust["methods"]:
        for ratio in robust["imbalance"]:
            if float(ratio) == 1:
                continue
            jobs.append(
                _base(
                    experiment="E03",
                    dataset=robust["dataset"],
                    method=method,
                    seed=int(robust["seed"]),
                    imbalance=float(ratio),
                    priority=60,
                )
            )
        for frac in robust["sample_frac"]:
            if float(frac) == 1:
                continue
            jobs.append(
                _base(
                    experiment="E03",
                    dataset=robust["dataset"],
                    method=method,
                    seed=int(robust["seed"]),
                    sample_frac=float(frac),
                    priority=61,
                )
            )
        for k in robust["k_values"]:
            if int(k) == int(protocol["k_default"]):
                continue
            jobs.append(
                _base(
                    experiment="E03",
                    dataset=robust["dataset"],
                    method=method,
                    seed=int(robust["seed"]),
                    k=int(k),
                    priority=62,
                )
            )
        for frac in robust["outlier_frac"]:
            if float(frac) == 0:
                continue
            jobs.append(
                _base(
                    experiment="E03",
                    dataset=robust["dataset"],
                    method=method,
                    seed=int(robust["seed"]),
                    outlier_frac=float(frac),
                    priority=63,
                )
            )
    e04 = protocol["e04"]
    for encoder in e04["encoders"]:
        for clusterer in e04["clusterers"]:
            method = f"{encoder}_{clusterer}"
            if method in {"ae_kmeans", "simclr_kmeans", "kmeans", "gmm", "pca_kmeans"}:
                continue
            seed_list = seeds if method in {"dinov2_kmeans", "clip_kmeans"} else [int(e04["seed"])]
            for seed in seed_list:
                priority = 25 if encoder in {"dinov2", "clip", "random_cnn"} else 45
                jobs.append(
                    _base(
                        experiment="E04",
                        dataset=e04["dataset"],
                        method=method,
                        seed=seed,
                        priority=priority,
                    )
                )
    dedup = {}
    for job in jobs:
        dedup[job_id(job) + job["experiment"]] = job
    return sorted(dedup.values(), key=lambda job: (job["priority"], job["dataset"], job["method"], job["seed"]))


def _should_skip(path: Path) -> bool:
    if not path.exists():
        return False
    record = read_json(path)
    if record.get("status") == "ok":
        return True
    error = record.get("error", "")
    if "too_few_samples" in error:
        return True
    return int(record.get("attempts", 1)) >= 2


def main() -> None:
    protocol = load_protocol()
    jobs = build_jobs(protocol)
    if "--dry-run" in sys.argv:
        print(f"{len(jobs)} jobs")
        for job in jobs:
            print(f"{job['priority']:02d} {job['experiment']} {job['dataset']} {job['method']} seed={job['seed']}")
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    log_path = project_root() / "results" / "queue.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if not _should_skip(result_path(job))]
    print(f"queue {len(pending)} pending / {len(jobs)} total", flush=True)
    for index, job in enumerate(pending, start=1):
        slot = project_root() / "results" / "queue"
        slot.mkdir(parents=True, exist_ok=True)
        spec = slot / f"{job['experiment']}__{job_id(job)}.json"
        spec.write_text(json.dumps(job), encoding="utf-8")
        print(f"[{index}/{len(pending)}] {job['experiment']} {job['dataset']} {job['method']} seed {job['seed']}", flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== {spec.name} =====\n")
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "run_one.py"), "--job-file", str(spec)],
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
    subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_noise.py")], cwd=ROOT, env=env, check=False)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze.py")], cwd=ROOT, env=env, check=False)


if __name__ == "__main__":
    main()
