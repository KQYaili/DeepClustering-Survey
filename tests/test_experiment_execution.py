"""Regressions for the course training schedule, seeding and failed-attempt retention."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contrastive.train import ensure_simclr
from deep.joint import _epoch_plan
from runtime.paths import pending_result, read_json, record_failure


def protocol():
    return yaml.safe_load((ROOT / "configs/protocol.yaml").read_text(encoding="utf-8"))


class ExecutionTests(unittest.TestCase):
    def test_extension_runs_full_registered_window_once(self):
        cfg = protocol()
        planned = cfg["ae"]["epochs"]
        extra = cfg["extension"]["extra_epochs"]
        limit = planned
        epoch = 0
        extended = False
        losses = []
        while epoch < limit:
            epoch += 1
            losses.append(0.9 ** epoch)
            limit, extended = _epoch_plan(planned, losses, extended, False, epoch, cfg)
        self.assertTrue(extended)
        self.assertEqual(epoch, planned + extra)
        flat = [1.0] * planned
        self.assertEqual(_epoch_plan(planned, flat, False, False, planned, cfg), (planned, False))

    def test_simclr_cold_start_uses_job_seed(self):
        cfg = protocol()
        cfg["simclr"]["epochs"] = 0  # Inspect the initial weights alone.
        job = {"dataset": "mnist", "seed": 2}
        with TemporaryDirectory() as temp:
            states = []
            for unrelated_seed in (12, 83):
                torch.manual_seed(unrelated_seed)
                path = Path(temp) / f"{unrelated_seed}.pt"
                with patch("contrastive.train.simclr_backbone_path", return_value=path):
                    net, _ = ensure_simclr(job, cfg, [], torch.device("cpu"))
                states.append(net.state_dict())
            for name, first in states[0].items():
                self.assertTrue(torch.equal(first, states[1][name]), name)

    def test_infrastructure_has_one_retry_with_both_tracebacks_retained(self):
        with TemporaryDirectory() as temp:
            output = Path(temp) / "raw/E01/run.json"
            self.assertTrue(pending_result(output))
            job = {"experiment": "E01"}
            for attempt in (1, 2):
                error = f"Traceback attempt {attempt}: ConnectionResetError: reset"
                record_failure(output, job, error)
                archived = Path(temp) / f"attempts/E01/run/attempt-{attempt}.json"
                self.assertEqual(read_json(archived)["error"], error)
                self.assertEqual(pending_result(output), attempt == 1)
            other = output.with_name("model.json")
            record_failure(other, job, "ValueError: invalid inputs")
            self.assertFalse(pending_result(other))


if __name__ == "__main__":
    unittest.main()
