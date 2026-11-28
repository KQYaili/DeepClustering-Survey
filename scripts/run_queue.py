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

from runtime.paths import job_id, pending_result, project_root, read_json, record_failure, result_path  # noqa: E402


def load_protocol() -> dict:
    return yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text(encoding="utf-8"))


def _base(protocol: dict, **kwargs) -> dict:
    job = {
        "experiment": "E01",
        "dataset": "mnist",
        "method": "kmeans",
        "seed": 0,
        "k": int(protocol["k_default"]),
        "latent_dim": int(protocol["ae"]["mlp_dims"][-1]),
        "lambda_recon": float(protocol["idec"]["lambda_recon"]),
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
                        protocol,
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
                    protocol,
                    experiment="E01",
                    dataset="cifar10",
                    method=method,
                    seed=seed,
                    priority=15 if method in {"kmeans", "pca_kmeans", "gmm"} else 40,
                )
            )
    fashion = protocol["e02"]
    for seed in fashion["seeds"]:
        for value in fashion["lambdas"]:
            jobs.append(_base(protocol, experiment="E02", dataset=fashion["dataset"], method="idec",
                              seed=int(seed), lambda_recon=float(value), priority=30))
        for dim in fashion["latent_dims"]:
            jobs.append(_base(protocol, experiment="E02", dataset=fashion["dataset"], method="idec",
                              seed=int(seed), latent_dim=int(dim), priority=30))
        for extra in ({"pretrain": False}, {"center_init": "random"}, {"freeze_encoder": True}):
            jobs.append(_base(protocol, experiment="E02", dataset=fashion["dataset"], method="idec",
                              seed=int(seed), priority=32, **extra))
    fixed = protocol["e02_fixed"]
    for seed in fixed["seeds"]:
        for value in fixed["lambdas"]:
            jobs.append(_base(protocol, experiment="E02_fixed", dataset=fixed["dataset"], method="idec",
                              seed=int(seed), lambda_recon=float(value), fixed_epochs=int(fixed["epochs"]),
                              priority=31))
    robust = protocol["e03"]
    k_default = int(protocol["k_default"])
    for seed in robust["seeds"]:
        first = int(seed) == int(robust["seeds"][0])
        methods = robust["methods"] if first else robust["replicate_methods"]
        for method in methods:
            variants = (
                [("imbalance", float(v), 60) for v in robust["imbalance"] if float(v) != 1]
                + [("sample_frac", float(v), 61) for v in robust["sample_frac"] if float(v) != 1]
                + [("k", int(v), 62) for v in robust["k_values"] if int(v) != k_default]
                + [("outlier_frac", float(v), 63) for v in robust["outlier_frac"] if float(v) != 0]
            )
            if not first:
                variants = [v for v in variants if v[0] in robust["replicate_factors"]]
            for name, value, priority in variants:
                jobs.append(_base(protocol, experiment="E03", dataset=robust["dataset"], method=method,
                                  seed=int(seed), priority=priority, **{name: value}))
    e04 = protocol["e04"]
    for dataset in e04["datasets"]:
        for encoder in e04["encoders"]:
            for clusterer in e04["clusterers"]:
                method = f"{encoder}_{clusterer}"
                if method in {"ae_kmeans", "simclr_kmeans"}:
                    continue  # the same fits as E01
                for seed in e04["seeds"]:
                    priority = 25 if encoder in {"dinov2", "clip", "random_cnn"} else 45
                    jobs.append(_base(protocol, experiment="E04", dataset=dataset, method=method,
                                      seed=int(seed), priority=priority))
    dedup = {}
    for job in jobs:
        dedup[job_id(job) + job["experiment"]] = job
    return sorted(dedup.values(), key=lambda job: (job["priority"], job["dataset"], job["method"], job["seed"]))


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
    pending = [job for job in jobs if pending_result(result_path(job))]
    print(f"queue {len(pending)} pending / {len(jobs)} total", flush=True)
    for index, job in enumerate(pending, start=1):
        slot = project_root() / "results" / "queue"
        slot.mkdir(parents=True, exist_ok=True)
        spec = slot / f"{job['experiment']}__{job_id(job)}.json"
        spec.write_text(json.dumps(job), encoding="utf-8")
        print(f"[{index}/{len(pending)}] {job['experiment']} {job['dataset']} {job['method']} seed {job['seed']}", flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== {spec.name} =====\n")
            process = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "run_one.py"), "--job-file", str(spec)],
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode and not result_path(job).exists():
            record_failure(result_path(job), job, f"Worker exited with code {process.returncode}; see results/queue.log")
    for script in ("eval_noise.py", "export_embeddings.py", "eval_train_stability.py", "analyze.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, env=env, check=True)
    if any(read_json(result_path(job)).get("status") != "ok" for job in jobs):
        raise SystemExit("Some training jobs failed. See results/raw and results/attempts.")


if __name__ == "__main__":
    main()
