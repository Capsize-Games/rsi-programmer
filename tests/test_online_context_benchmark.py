"""Frozen protocol and control boundaries for observed-example context."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi_programmer import online_benchmark as base
from rsi_programmer import online_counterbalanced as common
from rsi_programmer.online_context_benchmark import (
    PROTOCOL,
    load_protocol,
    policy_for,
    run,
)
from rsi_programmer.online_observed_context import ObservedContextPolicy


class ObservedContextBenchmarkTests(unittest.TestCase):
    def test_frozen_controls_matrices_and_original_encoder(self) -> None:
        protocol = load_protocol()
        self.assertEqual(protocol["planned_episode_rows"], 8640)
        self.assertEqual(protocol["orders"], common.load_schedule()["orders"])
        for seed in base.SEEDS:
            original = base._policy("frozen_snn", seed)
            self.assertIsNotNone(original)
            for arm in base.ARMS:
                policy = policy_for(arm, seed)
                if arm == "grammar_streaming":
                    self.assertIsNone(policy)
                    continue
                self.assertIsInstance(policy, ObservedContextPolicy)
                self.assertEqual(policy._encoder, original._encoder)
                self.assertEqual(policy._rng.getstate(),
                                 original._rng.getstate())
                self.assertEqual(policy._context_projection,
                                 protocol["context"]["projection_matrices"]
                                 [str(seed)])
                self.assertEqual(policy._context_projection_hash,
                                 protocol["context"]
                                 ["projection_matrix_sha256"][str(seed)])
                self.assertEqual(policy._learning, arm != "frozen_snn")
                self.assertEqual(policy._spiking, arm != "online_nonspiking")
        with self.assertRaises(ValueError):
            policy_for("unknown", 0)

    def test_changed_protocol_rejects_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_bytes(PROTOCOL.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_protocol(path)

    def test_existing_result_directory_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            sentinel = path / "sentinel.json"
            sentinel.write_text(json.dumps({"preserve": True}))
            with patch("rsi_programmer.online_context_benchmark.policy_for",
                       side_effect=AssertionError("must not execute")):
                with self.assertRaises(FileExistsError):
                    run(path)
            self.assertEqual(json.loads(sentinel.read_text()),
                             {"preserve": True})


if __name__ == "__main__":
    unittest.main()
