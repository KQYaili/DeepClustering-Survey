"""Export label-free Fashion-MNIST representations for the paired test-set plot."""

from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.data import evaluation_images
from runtime.load_model import load_any
from runtime.paths import checkpoint_path, read_json


def main() -> None:
    protocol = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text(encoding="utf-8"))
    dataset = protocol["e02"]["dataset"]
    seed = int(protocol["e02"]["seeds"][0])
    images, sample_ids = evaluation_images(dataset, protocol, seed)
    raw = ROOT / "results/raw"
    output = raw / "embeddings"
    output.mkdir(parents=True, exist_ok=True)
    for method in ("pca_kmeans", "ae_kmeans", "dec", "idec"):
        source, = (raw / "E01").glob(f"{dataset}__{method}__seed{seed}__*.json")
        record = read_json(source)
        if record["status"] != "ok":
            raise ValueError(f"Unfinished model: {source}")
        model = load_any(checkpoint_path(record["job"]))
        predictions = model.predict(images)
        if np.bincount(predictions, minlength=record["job"]["k"]).tolist() != record["metrics"]["counts"]:
            raise ValueError(f"Checkpoint predictions do not reproduce the cluster counts: {source}")
        destination = output / f"{dataset}__{method}__seed{seed}.npz"
        np.savez_compressed(destination, sample_ids=sample_ids, predictions=predictions,
                            embeddings=model.embeddings(images), source=np.array(str(source.relative_to(raw))))
        print(f"{destination.name}: {len(sample_ids)} aligned test images")


if __name__ == "__main__":
    main()
