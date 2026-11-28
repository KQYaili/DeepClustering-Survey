"""Regression tests for Issues #1-#5: label isolation, E05 scope, E2E cost, path sanitization, and PPT binding."""

from __future__ import annotations

import csv
import inspect
import json
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze  # noqa: E402
import build_pptx  # noqa: E402
from evaluation.selection import choose_train_candidate  # noqa: E402


def _make_obs(jacc: float, predict_sec: float = 0.05, nonempty: int = 10, **extra) -> dict:
    obs = {
        "macro_cluster_jaccard": jacc,
        "clean_nonempty": nonempty,
        "noisy_nonempty": nonempty,
        "clean_max_fraction": 0.15,
        "cluster_jaccard": [jacc] * 10,
        "predict_sec": predict_sec,
        "source": "results/raw/E05_train/mock.npz",
        "sample_sha256": "abc123",
        "settings_sha256": "def456",
        "n_samples": 2000,
        "noise_std": 0.1,
        "k": 10,
        "split": "fitted_training_subset",
        "seed": 0,
        "device": "cpu",
        "runtime": "test-host",
        "checkpoint_sha256": "ckpt0",
    }
    obs.update(extra)
    return obs


def _extract_slide_text(prs: Presentation, slide_index: int) -> str:
    slide = prs.slides[slide_index]
    parts = []
    for shp in slide.shapes:
        if shp.has_text_frame:
            parts.append(shp.text_frame.text)
        if shp.has_table:
            for row in shp.table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
    return "\n".join(parts)


class Issue2And3SelectionAuditTests(unittest.TestCase):
    def test_e05_selector_ignores_injected_test_label_metrics(self):
        """Issue #2: injecting or altering test ACC/NMI/ARI must never change E05 choice."""
        candidates = ["kmeans", "dec", "idec"]
        base_obs = {
            "kmeans": _make_obs(0.95, predict_sec=0.01, acc=0.20, nmi=0.10, ari=0.05),
            "dec": _make_obs(0.85, predict_sec=0.02, acc=0.99, nmi=0.99, ari=0.99),
            "idec": _make_obs(0.80, predict_sec=0.02, acc=0.98, nmi=0.98, ari=0.98),
        }
        res_base = choose_train_candidate(base_obs, candidates)
        self.assertEqual(res_base["choice"], "kmeans")
        self.assertEqual(res_base["reason"], "highest_train_macro_cluster_jaccard")
        self.assertFalse(res_base["speed_used_in_selection"])
        self.assertEqual(res_base["purpose"], "train_stability_and_collapse_diagnostic_only")
        for ev in res_base["evidence"].values():
            for forbidden in ("acc", "nmi", "ari", "y_test"):
                self.assertNotIn(forbidden, ev)

        # Flip test metrics completely; train-only choice must stay identical.
        mutated_obs = {
            "kmeans": _make_obs(0.95, predict_sec=0.01, acc=0.99, nmi=0.99, ari=0.99),
            "dec": _make_obs(0.85, predict_sec=0.02, acc=0.01, nmi=0.01, ari=0.00),
            "idec": _make_obs(0.80, predict_sec=0.02, acc=0.01, nmi=0.01, ari=0.00),
        }
        res_mut = choose_train_candidate(mutated_obs, candidates)
        self.assertEqual(res_mut["choice"], res_base["choice"])

    def test_e05_missing_or_mismatched_evidence_returns_insufficient_evidence(self):
        """Issue #2: incomplete or mismatched train observations yield choice=None."""
        res_missing = choose_train_candidate({"kmeans": _make_obs(0.95)}, ["kmeans", "dec"])
        self.assertIsNone(res_missing["choice"])
        self.assertEqual(res_missing["reason"], "insufficient_evidence")

        res_mismatch = choose_train_candidate(
            {
                "kmeans": _make_obs(0.95, sample_sha256="hash_a"),
                "dec": _make_obs(0.90, sample_sha256="hash_b"),
            },
            ["kmeans", "dec"],
        )
        self.assertIsNone(res_mismatch["choice"])
        self.assertEqual(res_mismatch["reason"], "insufficient_evidence")

    def test_e05_speed_is_unused_when_max_predict_sec_is_none(self):
        """Issue #3: mutating predict_sec does not change choice when no budget is configured."""
        candidates = ["kmeans", "dec"]
        obs_fast = {"kmeans": _make_obs(0.95, predict_sec=0.01), "dec": _make_obs(0.85, predict_sec=0.02)}
        obs_slow = {"kmeans": _make_obs(0.95, predict_sec=999.0), "dec": _make_obs(0.85, predict_sec=0.01)}
        r1 = choose_train_candidate(obs_fast, candidates, max_predict_sec=None)
        r2 = choose_train_candidate(obs_slow, candidates, max_predict_sec=None)
        self.assertEqual(r1["choice"], "kmeans")
        self.assertEqual(r2["choice"], "kmeans")
        self.assertFalse(r1["speed_used_in_selection"])

        # When a budget IS passed, slow candidates are filtered out and flag flips True.
        r_budget = choose_train_candidate(obs_slow, candidates, max_predict_sec=1.0)
        self.assertEqual(r_budget["choice"], "dec")
        self.assertTrue(r_budget["speed_used_in_selection"])

    def test_analyze_training_selections_has_no_test_label_references(self):
        """Issue #2: static audit that training_selections never touches test metrics or y_test."""
        src = inspect.getsource(analyze.training_selections)
        for token in ("y_test", "ari_mean", "acc_mean", "nmi_mean", "e01_summary", "e03_noise"):
            self.assertNotIn(token, src)

    def test_published_selection_json_matches_contract(self):
        """Issue #2 & #3: published selection.json has train-only metadata and speed_used=False."""
        sel = json.loads((ROOT / "results/tables/selection.json").read_text(encoding="utf-8"))
        for ds in ("mnist", "fashion_mnist", "cifar10"):
            entry = sel[ds]
            self.assertEqual(entry["split"], "fitted_training_subset")
            self.assertFalse(entry["speed_used_in_selection"])
            self.assertIsNone(entry["max_predict_sec"])
            self.assertEqual(entry["purpose"], "train_stability_and_collapse_diagnostic_only")
            self.assertIn("warning", entry)


class Issue3And5CostAndHygieneTests(unittest.TestCase):
    def test_end_to_end_cost_accounting_adds_matched_ae_and_nan_if_unmatched(self):
        """Issue #5 P1.5: cached DEC/IDEC adds exact matched AE pretrain time; unmatched stays NaN."""
        df = pd.DataFrame([
            {"dataset": "fashion_mnist", "method": "ae_kmeans", "seed": 0, "k": 10, "latent_dim": 10,
             "ae_cache_hit": False, "time_train_sec": 40.0},
            {"dataset": "fashion_mnist", "method": "dec", "seed": 0, "k": 10, "latent_dim": 10,
             "ae_cache_hit": True, "time_train_sec": 20.0},
            {"dataset": "fashion_mnist", "method": "idec", "seed": 1, "k": 10, "latent_dim": 10,
             "ae_cache_hit": True, "time_train_sec": 30.0},
        ])
        out = analyze.attach_end_to_end_train_time(df).set_index(["method", "seed"])
        self.assertAlmostEqual(out.loc[("ae_kmeans", 0), "time_train_e2e_sec"], 40.0)
        self.assertAlmostEqual(out.loc[("dec", 0), "time_train_e2e_sec"], 60.0)
        self.assertAlmostEqual(out.loc[("dec", 0), "time_train_amortized_sec"], 20.0 + 40.0 / 3.0)
        self.assertTrue(np.isnan(out.loc[("idec", 1), "time_train_e2e_sec"]))

    def test_retrospective_pareto_table_and_legacy_files_removed(self):
        """Issue #3 & #5 P1.5: benchmark_cost_frontiers replaces e05_cost_frontiers with E2E frontier."""
        self.assertFalse((ROOT / "results/tables/e05_cost_frontiers.csv").exists())
        self.assertFalse((ROOT / "results/figures/fig_e05_pareto.png").exists())
        self.assertTrue((ROOT / "results/figures/cost_frontier.png").exists())

        df = pd.read_csv(ROOT / "results/tables/benchmark_cost_frontiers.csv")
        self.assertTrue(df["uses_test_labels"].all())
        self.assertTrue((df["analysis_role"] == "retrospective_labeled_benchmark_only").all())
        fm = df[df["dataset"] == "fashion_mnist"]
        e2e_front = set(fm[fm["front_ari_time_e2e"]]["method"])
        self.assertEqual(e2e_front, {"pca_kmeans", "kmeans", "ae_kmeans", "dec"})

    def test_failure_logs_have_no_personal_machine_paths(self):
        """Issue #5 P2.8: sanitized failures.csv and attempt receipts contain no user home paths."""
        forbidden = ("/mnt/c/Users/", "C:\\Users\\", "/home/lgd")
        targets = [ROOT / "results/tables/failures.csv", *(ROOT / "results/attempts").rglob("*.json")]
        for path in targets:
            text = path.read_text(encoding="utf-8")
            for pat in forbidden:
                self.assertNotIn(pat, text, f"Found unsanitized path {pat!r} in {path}")


class Issue1And4And5PptBindingTests(unittest.TestCase):
    VERDICT, GUIDE, STABILITY = 28, 29, 27  # zero-based slide indices (31 main + 6 appendix)

    def test_pptx_updates_dynamically_from_tables_and_handles_missing_selection(self):
        """Issue #1 & #4: mutating tables changes rendered PPT text; missing/null selection renders fallback."""
        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            tables = tmp_dir / "tables"
            shutil.copytree(ROOT / "results/tables", tables)

            # Mutate MNIST DEC ARI to a unique sentinel value (0.812) in e01_summary.csv
            e01_path = tables / "e01_summary.csv"
            with e01_path.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            original = next(r["ari_mean"] for r in rows if r["dataset"] == "mnist" and r["method"] == "dec")
            for r in rows:
                if r["dataset"] == "mnist" and r["method"] == "dec":
                    r["ari_mean"] = "0.8120"
            with e01_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            # Remove selection.json to verify missing-file fallback
            (tables / "selection.json").unlink()
            out_pptx = tmp_dir / "mutated.pptx"
            build_pptx.build(tables_dir=tables, out_path=out_pptx)

            prs = Presentation(out_pptx)
            self.assertEqual(len(prs.slides), 37)
            for index in (self.VERDICT, self.GUIDE):
                text = _extract_slide_text(prs, index)
                self.assertIn("0.812", text)
                self.assertNotIn(f"{float(original):.3f}", text)
            self.assertIn("证据不足", _extract_slide_text(prs, self.STABILITY))
            notes = prs.slides[self.VERDICT].notes_slide.notes_text_frame.text
            self.assertIn("0.812", notes)

            # Also verify choice=null in selection.json renders fallback
            (tables / "selection.json").write_text(
                json.dumps({ds: {"choice": None, "reason": "insufficient_evidence"} for ds in ("mnist", "fashion_mnist", "cifar10")}),
                encoding="utf-8",
            )
            out_null = tmp_dir / "null_sel.pptx"
            build_pptx.build(tables_dir=tables, out_path=out_null)
            prs_null = Presentation(out_null)
            self.assertIn("证据不足", _extract_slide_text(prs_null, self.STABILITY))

    def test_speaker_notes_strict_slide_id_binding_fails_on_mismatch(self):
        """Issue #5 P0.2: missing or reordered slide IDs in speaker_notes.md must raise ValueError."""
        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            orig_notes = (ROOT / "presentation/speaker_notes.md").read_text(encoding="utf-8")

            # 1) Missing one slide ID -> ValueError
            bad_missing = orig_notes.replace("## [S17_noise]", "## [S17_WRONG_ID]")
            bad_path = tmp_dir / "bad_notes.md"
            bad_path.write_text(bad_missing, encoding="utf-8")
            with self.assertRaises(ValueError):
                build_pptx.build(out_path=tmp_dir / "fail1.pptx", notes_path=bad_path)

            # 2) Valid build embeds [SlideID] tag into all 37 slides' notes and resolves every {fact}
            ok_pptx = tmp_dir / "ok.pptx"
            build_pptx.build(out_path=ok_pptx)
            prs = Presentation(ok_pptx)
            self.assertEqual(len(prs.slides), 37)
            for idx, slide in enumerate(prs.slides):
                note_txt = slide.notes_slide.notes_text_frame.text.strip()
                self.assertTrue(note_txt.startswith("["), f"Slide {idx + 1} note missing [SlideID] header")
                self.assertNotRegex(note_txt, r"\{[A-Za-z_][^}]*\}", f"Slide {idx + 1} has an unresolved fact")


if __name__ == "__main__":
    unittest.main()
