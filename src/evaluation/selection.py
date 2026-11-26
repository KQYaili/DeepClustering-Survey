"""Pre-registered selection rules.

Thresholds live in configs/protocol.yaml. This function only compares measured
summaries. Missing measurements block the rule that needs them.
"""

from __future__ import annotations


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
    notes = []
    pretrained = summary.get("pretrained_kmeans_ari")
    deep = _present_max(
        summary,
        ["ae_kmeans_ari", "dec_ari", "idec_ari", "vade_ari", "simclr_kmeans_ari", "cc_ari"],
    )
    if pretrained is not None and deep is not None and pretrained >= deep[1] + float(rules["pretrain_margin"]):
        second = summary.get("pretrained_best_second_ari")
        if second is not None and second >= pretrained + float(rules["second_stage_margin"]):
            return {
                "choice": summary.get("pretrained_second_name"),
                "reason": "pretrained_embedding_plus_second_stage",
                "evidence": {
                    "pretrained_kmeans_ari": pretrained,
                    "best_from_scratch": {"method": deep[0], "ari": deep[1]},
                    "second_stage_ari": second,
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
