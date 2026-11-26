"""Run one job. Ground-truth labels are used only after fit returns."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.data import materialize  # noqa: E402
from evaluation.metrics import evaluate_partition  # noqa: E402
from runtime.fit import fit_job  # noqa: E402
from runtime.paths import checkpoint_path, read_json, result_path, write_json  # noqa: E402


def load_protocol() -> dict:
    return yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text(encoding="utf-8"))


def execute(job: dict) -> dict:
    protocol = load_protocol()
    destination = result_path(job)
    x_train, _y_train, x_test, y_test = materialize(
        job["dataset"],
        protocol,
        int(job["seed"]),
        float(job["sample_frac"]),
        float(job["imbalance"]),
        float(job["outlier_frac"]),
    )
    # y_train is intentionally unused. Imbalance already consumed labels inside materialize.
    del _y_train
    started = time.perf_counter()
    model, payload = fit_job(job, protocol, x_train, x_test)
    prediction = payload.pop("prediction")
    embeddings = payload.pop("embeddings")
    metrics = evaluate_partition(
        y_test,
        prediction,
        embeddings,
        k=int(job["k"]),
        silhouette_points=int(protocol["silhouette"]["max_points"]),
        seed=int(job["seed"]),
    )
    model.save(checkpoint_path(job))
    record = {
        "status": "ok",
        "job": job,
        "metrics": metrics,
        "recon_sse": payload["recon_sse"],
        "n_parameters": payload["n_parameters"],
        "time_train_sec": payload["time_train_sec"],
        "time_infer_sec": payload["time_infer_sec"],
        "time_feature_sec": payload["time_feature_sec"],
        "peak_gpu_mb": payload["peak_gpu_mb"],
        "external_pretrain": payload["external_pretrain"],
        "external_data": payload.get("external_data", "none"),
        "history": {
            "epochs_ran": payload["history"].get("epochs_ran"),
            "extended": payload["history"].get("extended"),
            "stopped_early": payload["history"].get("stopped_early"),
            "train_loss": payload["history"].get("train_loss", []),
            "pretrain_loss": payload["history"].get("pretrain_loss", []),
            "ae_cache_hit": payload["history"].get("ae_cache_hit"),
            "backbone_cache_hit": payload["history"].get("backbone_cache_hit"),
        },
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "wall_sec": time.perf_counter() - started,
        "attempts": 1,
    }
    if destination.exists():
        previous = read_json(destination)
        record["attempts"] = int(previous.get("attempts", 0)) + 1
    write_json(destination, record)
    print(
        f"OK {job['experiment']} {job['dataset']} {job['method']} seed {job['seed']} "
        f"ACC {metrics['acc']:.4f} NMI {metrics['nmi']:.4f} ARI {metrics['ari']:.4f}",
        flush=True,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job_file).read_text(encoding="utf-8"))
    try:
        execute(job)
    except Exception:
        destination = result_path(job)
        attempts = 1
        if destination.exists():
            try:
                attempts = int(read_json(destination).get("attempts", 0)) + 1
            except Exception:
                attempts = 1
        write_json(
            destination,
            {
                "status": "failed",
                "job": job,
                "error": traceback.format_exc(),
                "attempts": attempts,
            },
        )
        print(f"FAILED {job['method']} {job['dataset']} seed {job['seed']}", flush=True)
        raise


if __name__ == "__main__":
    main()
