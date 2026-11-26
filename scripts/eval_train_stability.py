"""Save paired cluster assignments on fitted training images, without class labels."""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from datasets.data import add_gaussian_noise, training_images
from run_queue import build_jobs, load_protocol
from runtime.load_model import load_any
from runtime.paths import checkpoint_path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", help="Only process the named E01 candidates")
    args = parser.parse_args()
    cfg_path = ROOT / "configs/selection_diagnostic.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    protocol = load_protocol()
    jobs = [job for job in build_jobs(protocol) if job["experiment"] == cfg["candidate_experiment"]
            and job["seed"] == cfg["seed"] and (not args.methods or job["method"] in args.methods)]
    output_dir = ROOT / "results/raw/E05_train"
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in protocol["splits"]:
        images, train_ids = training_images(dataset, protocol, cfg["seed"])
        positions = np.random.default_rng(cfg["seed"]).choice(
            len(train_ids), min(cfg["sample_count"], len(train_ids)), replace=False)
        indices, clean = train_ids[positions], images[positions]
        del images
        noisy = add_gaussian_noise(clean, cfg["noise_std"], cfg["seed"])
        for job in (j for j in jobs if j["dataset"] == dataset):
            checkpoint = checkpoint_path(job)
            path = output_dir / f"{dataset}__{job['method']}__seed{cfg['seed']}.npz"
            fingerprint = digest(checkpoint)
            settings = digest(cfg_path)
            if path.exists():
                with np.load(path) as previous:
                    if (str(previous["checkpoint_sha256"]) == fingerprint
                            and str(previous["settings_sha256"]) == settings
                            and "runtime" in previous.files
                            and np.array_equal(previous["sample_ids"], indices)):
                        continue
            model = load_any(checkpoint)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            original = model.predict(clean)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            predict_sec = time.perf_counter() - start
            shifted = model.predict(noisy)
            temp = path.with_suffix(".tmp")
            with temp.open("wb") as stream:
                np.savez_compressed(stream, dataset=dataset, method=job["method"], seed=cfg["seed"],
                                    sample_ids=indices, clean=original, noisy=shifted, k=job["k"],
                                    split=cfg["split"], noise_std=cfg["noise_std"], predict_sec=predict_sec,
                                    checkpoint=checkpoint.relative_to(ROOT).as_posix(),
                                    device=("cpu" if job["method"] in {"pca_kmeans", "gmm", "kmeans"}
                                            or not torch.cuda.is_available() else "cuda"),
                                    runtime=(torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"),
                                    checkpoint_sha256=fingerprint, settings_sha256=settings)
            os.replace(temp, path)
            print(f"Wrote {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
