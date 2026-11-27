"""Read the active raw records and build the survey's measured comparisons."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.selection import choose, choose_train_candidate, stability_evidence
from run_queue import build_jobs, load_protocol
from runtime.paths import job_id, read_json, write_json
from visualization.benchmark import render_figures

RAW = ROOT / "results/raw"
TABLES = ROOT / "results/tables"
GROUPS = ("E01", "E02", "E03", "E03_noise", "E04")
FACTORS = ("k", "latent_dim", "lambda_recon", "pretrain", "center_init", "freeze_encoder",
           "sample_frac", "imbalance", "outlier_frac")
MEASURES = ("acc", "nmi", "ari", "silhouette", "n_nonempty", "max_fraction", "entropy_norm",
            "recon_sse", "n_train", "n_test", "n_parameters", "time_train_sec",
            "time_feature_sec", "time_infer_sec", "peak_gpu_mb")
FIELDS = ("source", "experiment", "dataset", "method", "seed", *FACTORS, "noise_std", *MEASURES,
          "external_pretrain", "epochs_ran", "extended", "stopped_early", "ae_cache_hit", "backbone_cache_hit")


def expected_paths(jobs, protocol):
    for job in jobs:
        stem = job_id(job)
        yield RAW / job["experiment"] / f"{stem}.json"
        if job["experiment"] in {"E01", "E03"}:
            for std in protocol["e03"]["noise_std"]:
                tag = str(std).replace(".", "p")
                yield RAW / "E03_noise" / f"{stem}__std{tag}.json"


def load_records(jobs, protocol):
    expected = set(expected_paths(jobs, protocol))
    observed = {path for group in GROUPS for path in (RAW / group).glob("*.json")}
    rows, coverage, errors = [], [], []
    for path in sorted(expected | observed):
        source = path.relative_to(ROOT).as_posix()
        record = read_json(path) if path in observed else {}
        status = record.get("status", "missing")
        coverage.append(dict(source=source, experiment=path.parent.name,
                             planned=path in expected, status=status))
        if status == "failed":
            errors.append(dict(source=source, experiment=path.parent.name,
                               attempts=record.get("attempts", 1), error=record.get("error")))
        if status != "ok":
            continue
        if path not in expected:
            raise ValueError(f"Unscheduled result in active directory: {source}")
        job = record["job"]
        if job["experiment"] != path.parent.name:
            raise ValueError(f"Experiment mismatch: {source}")
        row = {**job, **record, **record.get("history", {}), **record["metrics"], "source": source}
        rows.append({k: row.get(k) for k in FIELDS})
    # Separate attempt receipts are excluded from the analysis denominator.
    for path in sorted((ROOT / "results/attempts").rglob("*.json")):
        attempt = read_json(path)
        if attempt.get("status") == "failed":
            errors.append(dict(source=path.relative_to(ROOT).as_posix(), experiment=attempt["job"]["experiment"],
                               attempts=attempt["attempts"], error=attempt.get("error")))
    return (pd.DataFrame(rows, columns=FIELDS), pd.DataFrame(coverage),
            pd.DataFrame(errors, columns=["source", "experiment", "attempts", "error"]))


def aggregate(frame, keys, metrics):
    columns = [*keys, "n", "seeds", "sources"]
    output = []
    for names, items in frame.groupby(keys, sort=True):
        if items.seed.duplicated().any():
            raise ValueError(f"Multiple runs in a seed for {names}")
        items = items.sort_values("seed")
        row = dict(zip(keys, names if isinstance(names, tuple) else (names,)))
        row.update(n=len(items), seeds=",".join(map(str, items.seed)),
                   sources="|".join(items.source))
        for metric in metrics:
            values = pd.to_numeric(items[metric], errors="raise").dropna()
            row[f"{metric}_mean"] = values.mean() if len(values) == len(items) else None
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) == len(items) and len(values) > 1 else None
        output.append(row)
    return pd.DataFrame(output, columns=[*columns, *(f"{m}_{s}" for m in metrics for s in ("mean", "std"))])


def e01_summary(frame):
    metrics = ["acc", "nmi", "ari", "time_train_sec", "time_infer_sec", "peak_gpu_mb",
               "n_parameters", "silhouette"]
    result = aggregate(frame, ["dataset", "method"], metrics)
    # Preserve the original public CSV column names.
    return result.rename(columns={f"time_{step}_sec_{stat}": f"time_{step}_{stat}_sec"
                                  for step in ("train", "infer") for stat in ("mean", "std")})


def single_factor(frame, defaults, factors):
    parts = []
    for factor in factors:
        matches = pd.Series(True, index=frame.index, dtype=bool)
        for name, value in defaults.items():
            if name != factor:
                matches &= frame[name].eq(value)
        selected = frame.loc[matches].copy()
        selected.insert(0, "factor", factor)
        selected.insert(1, "value", selected[factor])
        parts.append(selected.sort_values(["method", factor]))
    return pd.concat(parts, ignore_index=True)


def experiment_tables(runs, protocol):
    e01 = runs[runs.experiment.eq("E01")]
    baseline = dict(latent_dim=protocol["ae"]["mlp_dims"][-1], lambda_recon=protocol["idec"]["lambda_recon"],
                    pretrain=True, center_init="kmeans", freeze_encoder=False)
    ablation = single_factor(runs[runs.experiment.eq("E02")], baseline, baseline)
    robust = protocol["e03"]
    anchors = e01[e01.dataset.eq(robust["dataset"]) & e01.seed.eq(robust["seed"])
                  & e01.method.isin(robust["methods"])]
    variants = pd.concat([anchors, runs[runs.experiment.eq("E03")]])
    defaults = dict(sample_frac=1.0, imbalance=1.0, outlier_frac=0.0, k=protocol["k_default"])
    sensitivity = single_factor(variants, defaults, defaults)

    # Only compare noise added to standard E01 fits; noise on E03 variants remains in all_runs.
    noise = runs[runs.experiment.eq("E03_noise")]
    for name, value in defaults.items():
        noise = noise.loc[noise[name].eq(value)]
    noise = aggregate(noise, ["dataset", "method", "noise_std"], ["ari", "nmi", "acc", "n_nonempty"])
    cfg = protocol["e04"]
    paired_rows = []
    for encoder in cfg["encoders"]:
        for clusterer in cfg["clusterers"]:
            source = "E01" if encoder in {"ae", "simclr"} and clusterer == "kmeans" else "E04"
            chosen = runs.loc[runs.experiment.eq(source) & runs.dataset.eq(cfg["dataset"])
                              & runs.seed.eq(cfg["seed"]) & runs.method.eq(f"{encoder}_{clusterer}")].copy()
            chosen["source_experiment"], chosen["encoder"], chosen["clusterer"] = source, encoder, clusterer
            paired_rows.append(chosen)
    paired = pd.concat(paired_rows, ignore_index=True)
    ext = runs[runs.experiment.eq("E04") & runs.external_pretrain.eq(True)
               & runs.method.isin(["clip_kmeans", "dinov2_kmeans"])].copy()
    ext["encoder"] = ext.method.str.removesuffix("_kmeans")
    ext["clusterer"] = "kmeans"
    pretrained = aggregate(ext, ["dataset", "encoder", "clusterer"],
                           ["ari", "acc", "nmi", "time_feature_sec", "time_train_sec", "time_infer_sec"])
    return e01_summary(e01), ablation, sensitivity, noise, paired, pretrained


def cost_frontiers(summary, noise, protocol):
    sigma = max(protocol["e03"]["noise_std"])
    cols = ["dataset", "method", "seeds", "ari_mean"]
    noisy = noise.loc[noise.noise_std.eq(sigma), cols].rename(columns={"ari_mean": "noisy_ari_mean"})
    costs = summary.merge(noisy, on=["dataset", "method", "seeds"], how="left")
    costs["noise_std"] = sigma
    objectives = {"ari_mean": 1, "time_train_mean_sec": -1, "noisy_ari_mean": 1, "peak_gpu_mb_mean": -1}
    for label, count in [("front_ari_time", 2), ("front_four_measures", 4)]:
        costs[label] = pd.Series(pd.NA, index=costs.index, dtype="boolean")
        for _, group in costs.groupby(["dataset", "seeds"]):
            if set(map(int, group.seeds.iloc[0].split(","))) != set(protocol["seeds"]):
                continue
            columns = list(objectives)[:count]
            valid = group.dropna(subset=columns)
            points = valid[columns].to_numpy(float) * [objectives[k] for k in columns]
            costs.loc[valid.index, label] = [
                not np.any(np.all(points >= point, axis=1) & np.any(points > point, axis=1)) for point in points
            ]
    return costs


def training_selections(jobs, protocol):
    cfg_path = ROOT / "configs/selection_diagnostic.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    settings_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    by_dataset = {}
    for dataset in protocol["splits"]:
        methods = {j["method"] for j in jobs if j["dataset"] == dataset and
                   j["experiment"] == cfg["candidate_experiment"] and j["seed"] == cfg["seed"]}
        observations = {}
        for method in sorted(methods):
            path = RAW / "E05_train" / f"{dataset}__{method}__seed{cfg['seed']}.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as data:
                if (str(data["dataset"]) != dataset or str(data["method"]) != method
                        or str(data["settings_sha256"]) != settings_sha
                        or str(data["split"]) != cfg["split"] or int(data["seed"]) != cfg["seed"]
                        or float(data["noise_std"]) != cfg["noise_std"] or "runtime" not in data.files):
                    continue
                ids = data["sample_ids"]
                if len(np.unique(ids)) != len(ids) or len(ids) != len(data["clean"]):
                    raise ValueError(f"Unpaired training assignments in {path.name}")
                evidence = stability_evidence(data["clean"], data["noisy"], int(data["k"]))
                for key in ("k", "noise_std", "seed", "predict_sec", "runtime", "device", "split",
                            "settings_sha256", "checkpoint_sha256"):
                    evidence[key] = data[key].item()
                observations[method] = dict(**evidence, n_samples=len(ids),
                                            sample_sha256=hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
                                            source=path.relative_to(ROOT).as_posix())
        by_dataset[dataset] = choose_train_candidate(observations, methods)
    return by_dataset


def benchmark_interpretation(summary, noise, paired, pretrained, protocol):
    decisions = {}
    for dataset in protocol["splits"]:
        rows = summary[summary.dataset.eq(dataset) & summary.n.eq(len(protocol["seeds"]))]
        inputs = {}
        for r in rows.itertuples():
            inputs[f"{r.method}_ari"] = r.ari_mean
            inputs[f"{r.method}_time_sec"] = r.time_train_mean_sec
        idec_noise = noise[noise.dataset.eq(dataset) & noise.method.eq("idec")]
        clean = idec_noise[idec_noise.noise_std.eq(0)]
        noisy = idec_noise[idec_noise.noise_std.eq(max(protocol["e03"]["noise_std"]))]
        if (len(clean) == len(noisy) == 1 and clean.iloc[0].seeds == noisy.iloc[0].seeds
                and clean.iloc[0].n == len(protocol["seeds"]) and clean.iloc[0].ari_mean > 0):
            inputs["idec_retention"] = noisy.iloc[0].ari_mean / clean.iloc[0].ari_mean
        ext = pretrained[pretrained.dataset.eq(dataset) & pretrained.n.eq(len(protocol["seeds"]))]
        if len(ext):
            best = ext.loc[ext.ari_mean.idxmax()]
            inputs.update(best_pretrained_encoder=best.encoder, pretrained_kmeans_ari=best.ari_mean)
            enc = paired[paired.dataset.eq(dataset) & paired.encoder.eq(best.encoder)]
            reference = enc[enc.clusterer.eq("kmeans")]
            heads = enc[~enc.clusterer.eq("kmeans")]
            if len(reference) == 1 and len(heads):
                head = heads.loc[heads.ari.idxmax()]
                inputs.update(pretrained_paired_kmeans_ari=reference.iloc[0].ari,
                              pretrained_best_second_ari=head.ari, pretrained_second_name=head.method)
        decisions[dataset] = {**choose(inputs, protocol["selection"]), "inputs": inputs,
                              "use": "Retrospective only. The IDEC noise gate applies to IDEC only."}
    return decisions


def main():
    protocol = load_protocol()
    jobs = build_jobs(protocol)
    runs, coverage, failures = load_records(jobs, protocol)
    summary, ablation, sensitivity, noise, paired, pretrained = experiment_tables(runs, protocol)
    costs = cost_frontiers(summary, noise, protocol)
    TABLES.mkdir(parents=True, exist_ok=True)
    outputs = dict(all_runs=runs, coverage=coverage, failures=failures, e01_summary=summary,
                   e02_ablation=ablation, e03_sensitivity=sensitivity, e03_noise=noise,
                   e04_paired=paired, e04_pretrained=pretrained, e05_cost_frontiers=costs)
    for name, table in outputs.items():
        table.to_csv(TABLES / f"{name}.csv", index=False, lineterminator="\n")
    train_sel = training_selections(jobs, protocol)
    write_json(TABLES / "selection.json", train_sel)
    write_json(TABLES / "benchmark_interpretation.json",
               benchmark_interpretation(summary, noise, paired, pretrained, protocol))
    render_figures(ROOT, protocol, outputs, selection=train_sel)
    print(f"{len(runs)} successful records; {coverage.status.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
