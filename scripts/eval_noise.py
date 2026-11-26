"""Test-time Gaussian noise. Models are not retrained and labels are not used to build the noise."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.data import add_gaussian_noise, materialize  # noqa: E402
from evaluation.metrics import evaluate_partition  # noqa: E402
from runtime.fit import IMAGE_METHODS  # noqa: E402
from runtime.load_model import load_any  # noqa: E402
from runtime.paths import checkpoint_path, project_root, read_json, result_path, write_json  # noqa: E402


def main() -> None:
    protocol = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text(encoding="utf-8"))
    raw = project_root() / "results" / "raw"
    if not raw.exists():
        print("no results yet", flush=True)
        return
    records = []
    for path in raw.glob("*/*.json"):
        record = read_json(path)
        if record.get("status") != "ok":
            continue
        job = record["job"]
        if job["method"] not in IMAGE_METHODS:
            continue
        if job["experiment"] not in {"E01", "E03"}:
            continue
        records.append(job)
    print(f"noise eval on {len(records)} checkpoints", flush=True)
    for job in records:
        checkpoint = checkpoint_path(job)
        if not checkpoint.exists():
            continue
        model = load_any(checkpoint)
        _x_train, _y_train, x_test, y_test = materialize(
            job["dataset"],
            protocol,
            int(job["seed"]),
            float(job["sample_frac"]),
            float(job["imbalance"]),
            float(job["outlier_frac"]),
        )
        del _x_train, _y_train
        for std in protocol["e03"]["noise_std"]:
            noise_job = dict(job)
            noise_job["experiment"] = "E03_noise"
            noise_job["noise_std"] = float(std)
            destination = result_path(noise_job).with_name(result_path(noise_job).name.replace(".json", f"__std{str(std).replace('.', 'p')}.json"))
            if destination.exists():
                previous = read_json(destination)
                if previous.get("status") == "ok":
                    continue
            noisy = add_gaussian_noise(x_test, float(std), int(job["seed"]))
            prediction = model.predict(noisy)
            embeddings = model.embeddings(noisy)
            metrics = evaluate_partition(
                y_test,
                prediction,
                embeddings,
                k=int(job["k"]),
                silhouette_points=int(protocol["silhouette"]["max_points"]),
                seed=int(job["seed"]),
            )
            write_json(
                destination,
                {
                    "status": "ok",
                    "job": noise_job,
                    "noise_std": float(std),
                    "metrics": metrics,
                    "n_test": int(len(x_test)),
                },
            )
            print(
                f"noise {job['dataset']} {job['method']} seed {job['seed']} std {std} ARI {metrics['ari']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
