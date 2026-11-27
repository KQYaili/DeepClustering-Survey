"""Train-only stability selection and retrospective interpretation of a benchmark."""

from __future__ import annotations

import numpy as np


def stability_evidence(clean, noisy, k: int) -> dict:
    """Within each predicted cluster, measure membership preserved after perturbation."""
    a, b = np.asarray(clean), np.asarray(noisy)
    if (a.ndim != 1 or b.shape != a.shape or not a.size or k < 2 or
            any(v.dtype.kind not in "iu" or v.min() < 0 or v.max() >= k for v in (a, b))):
        raise ValueError("invalid paired cluster assignments")
    clean_size = np.bincount(a, minlength=k)
    noisy_size = np.bincount(b, minlength=k)
    intersection = np.bincount(a[a == b], minlength=k)
    union = clean_size + noisy_size - intersection
    scores = np.divide(intersection, union, out=np.zeros(k), where=union > 0)
    return {"macro_cluster_jaccard": float(scores.mean()), "clean_nonempty": int((clean_size > 0).sum()),
            "noisy_nonempty": int((noisy_size > 0).sum()), "clean_max_fraction": float(clean_size.max() / a.size),
            "cluster_jaccard": scores.tolist()}


def choose_train_candidate(observations: dict, candidates, *, max_predict_sec=None) -> dict:
    """Select only from a matched, complete train-side panel. Budget is optional seconds per batch."""
    fields = ("macro_cluster_jaccard", "clean_nonempty", "noisy_nonempty", "clean_max_fraction",
              "cluster_jaccard", "predict_sec", "source", "sample_sha256", "settings_sha256",
              "n_samples", "noise_std", "k", "split", "seed", "device", "runtime", "checkpoint_sha256")
    candidates = sorted(candidates)
    evidence = {name: {field: observations[name].get(field) for field in fields}
                for name in candidates if name in observations}
    result = {"choice": None, "reason": "insufficient_evidence", "split": "fitted_training_subset",
              "score": "macro_cluster_jaccard", "max_predict_sec": max_predict_sec,
              "speed_used_in_selection": max_predict_sec is not None,
              "purpose": "train_stability_and_collapse_diagnostic_only",
              "candidate_methods": candidates, "missing": sorted(set(candidates) - evidence.keys()),
              "eligibility": "All k clusters must be present in both predictions.",
              "tie_break": "Alphabetical method name for exactly tied scores.",
              "hardware_note": "predict_sec spans CPU (classical) and CUDA (neural) devices; reported as a diagnostic only unless a uniform device budget is configured.",
              "warning": "Perturbation stability does not measure semantic correctness; high stability can coexist with low semantic ARI.",
              "evidence": evidence,
              "scope": "Post-benchmark diagnostic on already-fitted E01 from-scratch training images (excludes external pretrained routes); one split and noise level."}
    if not evidence or result["missing"]:
        return result
    conditions = {(r["sample_sha256"], r["settings_sha256"], r["n_samples"], r["noise_std"], r["k"],
                   r["split"], r["seed"], r["runtime"]) for r in evidence.values()}
    if len(conditions) != 1 or any(r["split"] != result["split"] or r["macro_cluster_jaccard"] is None
                                   or r["sample_sha256"] is None for r in evidence.values()):
        return result
    eligible = {m: r for m, r in evidence.items() if r["clean_nonempty"] == r["k"]
                and r["noisy_nonempty"] == r["k"] and (max_predict_sec is None
                or (r["predict_sec"] is not None and r["predict_sec"] <= max_predict_sec))}
    if eligible:
        result.update(choice=max(eligible, key=lambda m: eligible[m]["macro_cluster_jaccard"]),
                      reason="highest_train_macro_cluster_jaccard")
    return result


def pareto_ari_time(rows: list[dict]) -> list[dict]:
    """Non-dominated rows when ARI is maximized and train time is minimized."""

    kept = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            better_or_equal = other["ari"] >= row["ari"] and other["time_train_sec"] <= row["time_train_sec"]
            strictly = other["ari"] > row["ari"] or other["time_train_sec"] < row["time_train_sec"]
            if better_or_equal and strictly:
                dominated = True
                break
        if not dominated:
            kept.append(row)
    return kept


def _present_max(summary: dict, keys: list[str]):
    values = [(key, summary[key]) for key in keys if summary.get(key) is not None]
    if not values:
        return None
    return max(values, key=lambda item: item[1])


def choose(summary: dict, rules: dict) -> dict:
    """Legacy, label-based interpretation of finished benchmark scores. Never deploy as a selector."""
    notes = []
    pretrained = summary.get("pretrained_kmeans_ari")
    deep = _present_max(
        summary,
        ["ae_kmeans_ari", "dec_ari", "idec_ari", "vade_ari", "simclr_kmeans_ari", "cc_ari"],
    )
    if pretrained is not None and deep is not None and pretrained >= deep[1] + float(rules["pretrain_margin"]):
        second = summary.get("pretrained_best_second_ari")
        paired_kmeans = summary.get("pretrained_paired_kmeans_ari", pretrained)
        if second is not None and paired_kmeans is not None and second >= paired_kmeans + float(rules["second_stage_margin"]):
            return {
                "choice": summary.get("pretrained_second_name"),
                "reason": "pretrained_embedding_plus_second_stage",
                "evidence": {
                    "pretrained_kmeans_ari": pretrained,
                    "best_from_scratch": {"method": deep[0], "ari": deep[1]},
                    "second_stage_ari": second,
                    "matched_kmeans_ari": paired_kmeans,
                },
                "notes": notes,
            }
        return {
            "choice": f"{summary.get('best_pretrained_encoder')}+kmeans",
            "reason": "pretrained_embedding_kmeans",
            "evidence": {
                "pretrained_kmeans_ari": pretrained,
                "best_from_scratch": {"method": deep[0], "ari": deep[1]},
            },
            "notes": notes,
        }

    idec = summary.get("idec_ari")
    ae = summary.get("ae_kmeans_ari")
    retention = summary.get("idec_retention")
    if idec is not None and ae is not None and idec >= ae + float(rules["ari_margin"]):
        if retention is None:
            notes.append("idec_rule_blocked_missing_noise")
        elif retention >= float(rules["noise_retention_floor"]):
            return {
                "choice": "idec",
                "reason": "joint_optimization_keeps_reconstruction",
                "evidence": {"idec_ari": idec, "ae_kmeans_ari": ae, "idec_retention": retention},
                "notes": notes,
            }
        else:
            notes.append("idec_rejected_on_noise_retention")

    pca = summary.get("pca_kmeans_ari")
    if ae is not None and pca is not None and ae >= pca + float(rules["ari_margin"]):
        return {
            "choice": "ae_kmeans",
            "reason": "nonlinear_representation_before_kmeans",
            "evidence": {"ae_kmeans_ari": ae, "pca_kmeans_ari": pca},
            "notes": notes,
        }

    classical = []
    if summary.get("kmeans_ari") is not None:
        classical.append(("kmeans", summary["kmeans_ari"], summary.get("kmeans_time_sec")))
    if summary.get("pca_kmeans_ari") is not None:
        classical.append(("pca_kmeans", summary["pca_kmeans_ari"], summary.get("pca_kmeans_time_sec")))
    if not classical:
        return {"choice": None, "reason": "insufficient_evidence", "evidence": {}, "notes": notes}
    classical.sort(key=lambda item: item[1], reverse=True)
    best_name, best_ari, best_time = classical[0]
    if len(classical) == 2 and abs(classical[0][1] - classical[1][1]) <= float(rules["ari_margin"]) / 2:
        ordered = sorted(classical, key=lambda item: (item[2] is None, item[2] if item[2] is not None else 0))
        best_name, best_ari, best_time = ordered[0]
        notes.append("classical_tie_broken_by_time")
    return {
        "choice": best_name,
        "reason": "classical_baseline",
        "evidence": {"ari": best_ari, "time_sec": best_time},
        "notes": notes,
    }
