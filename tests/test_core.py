"""CPU tests for metrics, losses, and the pre-registered selection rule."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deep.common import (  # noqa: E402
    build_autoencoder,
    cluster_kl,
    nt_xent,
    per_image_sse,
    soft_assign,
    target_distribution,
)
from deep.vade import VaDENet  # noqa: E402
from evaluation.metrics import clustering_accuracy  # noqa: E402
from evaluation.selection import choose, pareto_ari_time  # noqa: E402
from runtime.paths import job_id  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_accuracy_is_permutation_invariant(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([2, 2, 0, 0, 1, 1])
        self.assertAlmostEqual(clustering_accuracy(y_true, y_pred), 1.0)

    def test_accuracy_majority(self):
        y_true = np.array([0, 0, 0, 1])
        y_pred = np.array([4, 4, 4, 4])
        self.assertAlmostEqual(clustering_accuracy(y_true, y_pred), 0.75)


class LossTests(unittest.TestCase):
    def test_soft_assignment_and_kl(self):
        z = torch.randn(16, 8, requires_grad=True)
        centers = torch.randn(4, 8, requires_grad=True)
        posterior = soft_assign(z, centers)
        self.assertTrue(torch.allclose(posterior.sum(dim=1), torch.ones(16), atol=1e-5))
        target = target_distribution(posterior.detach())
        self.assertTrue(torch.allclose(target.sum(dim=1), torch.ones(16), atol=1e-5))
        loss = cluster_kl(target, posterior)
        loss.backward()
        self.assertIsNotNone(z.grad)
        self.assertGreater(float(z.grad.norm()), 0.0)

    def test_reconstruction_sse_matches_definition(self):
        x = torch.zeros(2, 1, 2, 2)
        x_hat = torch.ones(2, 1, 2, 2)
        # Per image, four pixels of error 1, so SSE is 4. Mean over the batch is 4.
        self.assertAlmostEqual(float(per_image_sse(x_hat, x)), 4.0)

    def test_autoencoder_backward(self):
        model = build_autoencoder("mnist", 10, [784, 32, 10])
        batch = torch.rand(4, 1, 28, 28)
        _z, recon = model(batch)
        loss = per_image_sse(recon, batch)
        loss.backward()
        self.assertEqual(tuple(recon.shape), (4, 1, 28, 28))

    def test_cifar_autoencoder_shape(self):
        model = build_autoencoder("cifar10", 10, [784, 500, 500, 2000, 10])
        batch = torch.rand(2, 3, 32, 32)
        _z, recon = model(batch)
        self.assertEqual(tuple(recon.shape), (2, 3, 32, 32))

    def test_contrastive_and_vade_backward(self):
        left = torch.randn(6, 16, requires_grad=True)
        right = torch.randn(6, 16, requires_grad=True)
        loss = nt_xent(left, right, 0.5)
        loss.backward()
        self.assertGreater(float(left.grad.norm()), 0.0)
        net = VaDENet("mnist", 4, 3, [784, 32, 16, 8, 4], -2.0)
        images = torch.rand(4, 1, 28, 28)
        elbo, gamma = net.elbo(images)
        elbo.backward()
        self.assertEqual(tuple(gamma.shape), (4, 3))
        self.assertTrue(torch.allclose(gamma.sum(dim=1), torch.ones(4), atol=1e-4))
        cifar = VaDENet("cifar10", 4, 3, [784, 32, 16, 8, 4], -2.0)
        recon = cifar.decode(torch.randn(2, 4), torch.rand(2, 3, 32, 32))
        self.assertEqual(tuple(recon.shape), (2, 3, 32, 32))


class SelectionTests(unittest.TestCase):
    def test_pretrained_margin_prefers_kmeans_until_second_stage_pays(self):
        rules = {
            "ari_margin": 0.02,
            "pretrain_margin": 0.05,
            "second_stage_margin": 0.02,
            "noise_retention_floor": 0.90,
        }
        summary = {
            "pretrained_kmeans_ari": 0.50,
            "best_pretrained_encoder": "dinov2",
            "ae_kmeans_ari": 0.30,
            "idec_ari": 0.32,
            "pretrained_best_second_ari": 0.51,
            "pretrained_second_name": "dinov2+dec_head",
        }
        chosen = choose(summary, rules)
        self.assertEqual(chosen["choice"], "dinov2+kmeans")
        summary["pretrained_best_second_ari"] = 0.54
        chosen = choose(summary, rules)
        self.assertEqual(chosen["choice"], "dinov2+dec_head")

    def test_idec_requires_measured_retention(self):
        rules = {
            "ari_margin": 0.02,
            "pretrain_margin": 0.05,
            "second_stage_margin": 0.02,
            "noise_retention_floor": 0.90,
        }
        summary = {"idec_ari": 0.40, "ae_kmeans_ari": 0.30, "pca_kmeans_ari": 0.20, "kmeans_ari": 0.10}
        chosen = choose(summary, rules)
        self.assertEqual(chosen["choice"], "ae_kmeans")
        self.assertIn("idec_rule_blocked_missing_noise", chosen["notes"])
        summary["idec_retention"] = 0.95
        chosen = choose(summary, rules)
        self.assertEqual(chosen["choice"], "idec")

    def test_pareto(self):
        rows = [
            {"method": "a", "ari": 0.2, "time_train_sec": 1.0},
            {"method": "b", "ari": 0.5, "time_train_sec": 2.0},
            {"method": "c", "ari": 0.4, "time_train_sec": 3.0},
        ]
        kept = {row["method"] for row in pareto_ari_time(rows)}
        self.assertEqual(kept, {"a", "b"})

    def test_fit_module_does_not_read_labels(self):
        source = (ROOT / "src" / "runtime" / "fit.py").read_text(encoding="utf-8")
        self.assertNotIn("y_test", source)
        self.assertNotIn("y_train", source)

    def test_protocol_is_explicit(self):
        protocol = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text(encoding="utf-8"))
        self.assertTrue(protocol["registered_before_test_metrics"])
        self.assertEqual(protocol["losses"]["reconstruction"], "mean_per_image_sum_of_squares")
        job = {
            "dataset": "fashion_mnist",
            "method": "idec",
            "seed": 0,
            "k": 10,
            "latent_dim": 10,
            "lambda_recon": 0.01,
            "pretrain": True,
            "center_init": "kmeans",
            "freeze_encoder": False,
            "sample_frac": 1.0,
            "imbalance": 1.0,
            "outlier_frac": 0.0,
        }
        self.assertIn("lam0p01", job_id(job))


if __name__ == "__main__":
    unittest.main()
