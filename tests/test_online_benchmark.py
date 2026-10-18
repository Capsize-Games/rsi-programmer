"""Pins and preflight checks for the online pilot runner."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi_programmer import online_benchmark as runner
from rsi_programmer.online_spiking import OnlineSpikingPolicy


class OnlineBenchmarkTests(unittest.TestCase):
    def test_real_policy_configuration_matches_published_protocol(self
                                                               ) -> None:
        declared = runner.load_corpus()["protocol"]["policy"]
        actual = OnlineSpikingPolicy(seed=0).snapshot()["config"]
        for key, field in (("input_features", "feature_count"),
                           ("hidden_units", "hidden_count"),
                           ("learning_rate", "lr"),
                           ("steps", "steps"),
                           ("eligibility_decay", "eligibility_decay"),
                           ("reward_baseline", "baseline_rate"),
                           ("membrane_beta", "beta"),
                           ("threshold", "threshold"),
                           ("weight_clip", "max_weight"),
                           ("update_clip", "max_update")):
            self.assertEqual(declared[key], actual[field], key)

    def test_modified_corpus_is_rejected_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_bytes(runner.CORPUS.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                runner.load_corpus(path)

    def test_existing_bundle_cannot_be_overwritten_or_execute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "_policy") as factory:
                with self.assertRaises(FileExistsError):
                    runner.run_benchmark(Path(directory))
                factory.assert_not_called()

    def test_source_manifest_covers_new_runtime_tests_and_workflow(self
                                                               ) -> None:
        files = runner.source_manifest()
        for name in ("rsi_programmer/online_spiking.py",
                     "rsi_programmer/online_graph.py",
                     "rsi_programmer/online_benchmark.py",
                     "scripts/audit_online_graph.py",
                     "tests/test_online_benchmark.py",
                     "fixtures/online_graph_v1.json",
                     "requirements.txt", "justfile",
                     ".github/workflows/ci.yml"):
            self.assertIn(name, files)


if __name__ == "__main__":
    unittest.main()
