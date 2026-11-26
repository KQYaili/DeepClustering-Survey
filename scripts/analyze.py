"""Build tables and figures from measured JSON only. Missing cells stay blank."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.selection import choose, pareto_ari_time  # noqa: E402
from runtime.paths import project_root, read_json, write_json  # noqa: E402
from visualization.style import COLORS, apply_style  # noqa: E402


def load_records() -> list[dict]:
    raw = project_root() / "results" / "raw"
    if not raw.exists():
        return []
    return [read_json(path) for path in raw.glob("*/*.json")]


def default_factors(job: dict, protocol: dict) -> bool:
    if int(job.get("k", -1)) != int(protocol["k_default"]):
        return False
    if int(job.get("latent_dim", -1)) != 10:
        return False
    if not bool(job.get("pretrain", True)):
        return False
    if job.get("center_init") != "kmeans":
        return False
    if bool(job.get("freeze_encoder", False)):
        return False
    if float(job.get("sample_frac", 1)) != 1.0 or float(job.get("imbalance", 1)) != 1.0:
        return False
    if float(job.get("outlier_frac", 0)) != 0.0:
        return False
    if job["method"] == "idec" and float(job.get("lambda_recon", 1)) != 1.0:
        return False
    if job["method"] == "dec" and float(job.get("lambda_recon", 0)) != 0.0:
        return False
    return True


def _mean_std(values: list[float]) -> tuple[float | None, float | None, int]:
    if not values:
        return None, None, 0
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return float(array.mean()), std, int(len(array))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict], protocol: dict) -> None:
    tables = project_root() / "results" / "tables"
    ok = [record for record in records if record.get("status") == "ok" and "metrics" in record]
    failed = [record for record in records if record.get("status") == "failed"]
    write_csv(
        tables / "failures.csv",
        [
            {
                "experiment": record["job"]["experiment"],
                "dataset": record["job"]["dataset"],
                "method": record["job"]["method"],
                "seed": record["job"]["seed"],
                "attempts": record.get("attempts", ""),
                "error_head": (record.get("error") or "").splitlines()[-1][:300],
            }
            for record in failed
        ],
    )
    flat = []
    for record in ok:
        job = record["job"]
        metrics = record["metrics"]
        flat.append(
            {
                "experiment": job["experiment"],
                "dataset": job["dataset"],
                "method": job["method"],
                "seed": job["seed"],
                "k": job["k"],
                "latent_dim": job["latent_dim"],
                "lambda_recon": job["lambda_recon"],
                "sample_frac": job["sample_frac"],
                "imbalance": job["imbalance"],
                "outlier_frac": job["outlier_frac"],
                "noise_std": record.get("noise_std", ""),
                "acc": metrics["acc"],
                "nmi": metrics["nmi"],
                "ari": metrics["ari"],
                "silhouette": "" if metrics["silhouette"] is None else metrics["silhouette"],
                "max_fraction": metrics["max_fraction"],
                "n_nonempty": metrics["n_nonempty"],
                "recon_sse": record.get("recon_sse", ""),
                "time_train_sec": record.get("time_train_sec", ""),
                "time_feature_sec": record.get("time_feature_sec", ""),
                "time_infer_sec": record.get("time_infer_sec", ""),
                "peak_gpu_mb": record.get("peak_gpu_mb", ""),
                "n_parameters": record.get("n_parameters", ""),
                "external_pretrain": record.get("external_pretrain", False),
                "epochs_ran": record.get("history", {}).get("epochs_ran", ""),
                "extended": record.get("history", {}).get("extended", ""),
                "stopped_early": record.get("history", {}).get("stopped_early", ""),
            }
        )
    write_csv(tables / "all_runs.csv", flat)

    grouped = defaultdict(list)
    for record in ok:
        job = record["job"]
        if job["experiment"] != "E01" or not default_factors(job, protocol):
            continue
        grouped[(job["dataset"], job["method"])].append(record)
    summary_rows = []
    for (dataset, method), items in sorted(grouped.items()):
        acc, acc_std, n = _mean_std([item["metrics"]["acc"] for item in items])
        nmi, nmi_std, _ = _mean_std([item["metrics"]["nmi"] for item in items])
        ari, ari_std, _ = _mean_std([item["metrics"]["ari"] for item in items])
        train, train_std, _ = _mean_std([float(item.get("time_train_sec") or 0) for item in items])
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n": n,
                "acc_mean": acc,
                "acc_std": acc_std,
                "nmi_mean": nmi,
                "nmi_std": nmi_std,
                "ari_mean": ari,
                "ari_std": ari_std,
                "time_train_mean_sec": train,
                "time_train_std_sec": train_std,
            }
        )
    write_csv(tables / "e01_summary.csv", summary_rows)
    _figures(ok, summary_rows, protocol)
    _selection(summary_rows, ok, protocol)


def _metric(record: dict, name: str) -> float:
    value = record["metrics"][name]
    return float(value)


def _figures(ok: list[dict], summary_rows: list[dict], protocol: dict) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    figures = project_root() / "results" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        datasets = sorted({row["dataset"] for row in summary_rows})
        methods = []
        for row in summary_rows:
            if row["method"] not in methods:
                methods.append(row["method"])
        grid = np.full((len(methods), len(datasets)), np.nan)
        lookup = {(row["dataset"], row["method"]): row["ari_mean"] for row in summary_rows}
        for i, method in enumerate(methods):
            for j, dataset in enumerate(datasets):
                if (dataset, method) in lookup:
                    grid[i, j] = lookup[(dataset, method)]
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        image = ax.imshow(grid, cmap="YlGnBu", aspect="auto", vmin=np.nanmin(grid), vmax=np.nanmax(grid))
        ax.set_xticks(range(len(datasets)), datasets, rotation=20, ha="right")
        ax.set_yticks(range(len(methods)), methods)
        for i in range(len(methods)):
            for j in range(len(datasets)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="#0F172A", fontsize=8)
        ax.set_title("E01 mean ARI")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(figures / "fig_e01_ari.png")
        plt.close(fig)

    e02 = [
        record
        for record in ok
        if record["job"]["experiment"] == "E02"
        and record["job"]["method"] == "idec"
        and int(record["job"]["latent_dim"]) == 10
        and bool(record["job"]["pretrain"])
        and record["job"]["center_init"] == "kmeans"
        and not bool(record["job"]["freeze_encoder"])
    ]
    if e02:
        e02 = sorted(e02, key=lambda record: float(record["job"]["lambda_recon"]))
        fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
        xs = [float(record["job"]["lambda_recon"]) for record in e02]
        ax1.plot(xs, [_metric(record, "ari") for record in e02], marker="o", color=COLORS[0], label="ARI")
        ax1.set_xscale("symlog", linthresh=0.01)
        ax1.set_xlabel("lambda on per-image reconstruction SSE")
        ax1.set_ylabel("ARI")
        ax2 = ax1.twinx()
        ax2.plot(xs, [record.get("recon_sse") for record in e02], marker="s", color=COLORS[1], label="recon SSE")
        ax2.set_ylabel("test reconstruction SSE")
        ax1.set_title("E02 Fashion-MNIST, latent 10, seed 0")
        fig.savefig(figures / "fig_e02_lambda.png")
        plt.close(fig)

    noise = [record for record in ok if record["job"]["experiment"] == "E03_noise"]
    if noise:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        series = defaultdict(list)
        for record in noise:
            if record["job"]["dataset"] != "fashion_mnist" or int(record["job"]["seed"]) != 0:
                continue
            if not default_factors(record["job"], protocol):
                continue
            series[record["job"]["method"]].append(record)
        for index, (method, items) in enumerate(sorted(series.items())):
            items = sorted(items, key=lambda record: float(record["noise_std"]))
            ax.plot(
                [float(record["noise_std"]) for record in items],
                [_metric(record, "ari") for record in items],
                marker="o",
                color=COLORS[index % len(COLORS)],
                label=method,
            )
        ax.set_xlabel("Gaussian noise std on [0, 1] pixels")
        ax.set_ylabel("ARI")
        ax.set_title("E03 test-time noise, Fashion-MNIST seed 0")
        ax.legend(fontsize=8)
        fig.savefig(figures / "fig_e03_noise.png")
        plt.close(fig)

    if summary_rows:
        points = [
            {"method": row["method"], "ari": row["ari_mean"], "time_train_sec": row["time_train_mean_sec"]}
            for row in summary_rows
            if row["dataset"] == "fashion_mnist" and row["time_train_mean_sec"] is not None
        ]
        if points:
            front = {row["method"] for row in pareto_ari_time(points)}
            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            for row in points:
                color = COLORS[1] if row["method"] in front else COLORS[8]
                ax.scatter(row["time_train_sec"], row["ari"], color=color, s=40)
                ax.annotate(row["method"], (row["time_train_sec"], row["ari"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
            ax.set_xlabel("mean train seconds")
            ax.set_ylabel("mean ARI")
            ax.set_title("Fashion-MNIST ARI–time frontier")
            fig.savefig(figures / "fig_e05_pareto.png")
            plt.close(fig)


def _selection(summary_rows: list[dict], ok: list[dict], protocol: dict) -> None:
    rules = protocol["selection"]
    by_dataset = defaultdict(dict)
    for row in summary_rows:
        by_dataset[row["dataset"]][f"{row['method']}_ari"] = row["ari_mean"]
        by_dataset[row["dataset"]][f"{row['method']}_time_sec"] = row["time_train_mean_sec"]
    decisions = {}
    for dataset, summary in by_dataset.items():
        noise = [
            record
            for record in ok
            if record["job"]["experiment"] == "E03_noise"
            and record["job"]["dataset"] == dataset
            and record["job"]["method"] == "idec"
            and int(record["job"]["seed"]) == 0
            and default_factors(record["job"], protocol)
        ]
        clean = [record for record in noise if float(record["noise_std"]) == 0.0]
        noisy = [record for record in noise if float(record["noise_std"]) == 0.3]
        if clean and noisy and _metric(clean[0], "ari") > 0.01:
            summary["idec_retention"] = _metric(noisy[0], "ari") / _metric(clean[0], "ari")
        e04_groups = defaultdict(list)
        for record in ok:
            job = record["job"]
            if job["experiment"] == "E04" and job["dataset"] == dataset and default_factors(job, protocol):
                e04_groups[job["method"]].append(record["metrics"]["ari"])
        for method, values in e04_groups.items():
            mean, _std, _n = _mean_std(values)
            summary[f"{method}_ari"] = mean
        pretrained = []
        for method in ("dinov2_kmeans", "clip_kmeans"):
            key = f"{method}_ari"
            if summary.get(key) is not None:
                pretrained.append((method.split("_")[0], summary[key], method))
        if pretrained:
            best = max(pretrained, key=lambda item: item[1])
            summary["pretrained_kmeans_ari"] = best[1]
            summary["best_pretrained_encoder"] = best[0]
            second_candidates = []
            for method, value in list(summary.items()):
                if method.startswith(best[0]) and method.endswith("_ari") and "kmeans" not in method:
                    second_candidates.append((method[: -len("_ari")], value))
            if second_candidates:
                name, value = max(second_candidates, key=lambda item: item[1])
                summary["pretrained_best_second_ari"] = value
                summary["pretrained_second_name"] = name
        decisions[dataset] = choose(summary, rules)
    write_json(project_root() / "results" / "tables" / "selection.json", decisions)


def main() -> None:
    protocol = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text(encoding="utf-8"))
    records = load_records()
    summarize(records, protocol)
    print(f"analyzed {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
