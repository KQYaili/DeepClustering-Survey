"""Evaluate saved image models under Gaussian noise without refitting."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.data import add_gaussian_noise, evaluation_images
from evaluation.metrics import evaluate_partition, evaluation_labels
from runtime.load_model import load_any
from runtime.paths import checkpoint_path, pending_result, read_json, record_failure, result_path, write_json


def noise_destination(job: dict, std: float):
    spec = dict(job, experiment="E03_noise", noise_std=float(std))
    base = result_path(spec)
    suffix = str(float(std)).replace(".", "p")
    return spec, base.with_name(f"{base.stem}__std{suffix}.json")


def main() -> None:
    protocol = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text(encoding="utf-8"))
    raw = ROOT / "results/raw"
    failed = []
    for experiment in ("E01", "E03"):
        for source in sorted((raw / experiment).glob("*.json")):
            record = read_json(source)
            if record["status"] != "ok":
                continue
            job = record["job"]
            conditions = [noise_destination(job, std) for std in protocol["e03"]["noise_std"]]
            pending = [(spec, path) for spec, path in conditions if pending_result(path)]
            failed.extend(str(path) for spec, path in conditions if path.exists()
                          and not pending_result(path) and read_json(path)["status"] != "ok")
            if not pending:
                continue
            try:
                model = load_any(checkpoint_path(job))
                images, indices = evaluation_images(job["dataset"], protocol, int(job["seed"]))
            except Exception:
                error = traceback.format_exc()
                for spec, path in pending:
                    record_failure(path, spec, error)
                    failed.append(str(path))
                continue
            for spec, path in pending:
                std = spec["noise_std"]
                try:
                    noisy = add_gaussian_noise(images, std, int(job["seed"]))
                    prediction = model.predict(noisy)
                    embeddings = model.embeddings(noisy)
                    metrics = evaluate_partition(
                        evaluation_labels(job["dataset"], indices), prediction, embeddings,
                        k=int(job["k"]), silhouette_points=int(protocol["silhouette"]["max_points"]),
                        seed=int(job["seed"]),
                    )
                    attempts = int(read_json(path).get("attempts", 1)) + 1 if path.exists() else 1
                    write_json(path, {
                        "status": "ok", "job": spec, "noise_std": std, "metrics": metrics,
                        "n_test": len(images), "attempts": attempts,
                    })
                    print(f"noise {job['dataset']} {job['method']} seed {job['seed']} "
                          f"std {std} ARI {metrics['ari']:.4f}", flush=True)
                except Exception:
                    record_failure(path, spec, traceback.format_exc())
                    failed.append(str(path))
    if failed:
        raise SystemExit(f"{len(failed)} noise evaluations failed. See results/raw/E03_noise and results/attempts.")


if __name__ == "__main__":
    main()
