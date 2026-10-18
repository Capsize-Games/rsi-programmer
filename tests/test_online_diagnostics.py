"""Tests for non-invasive cold-state online-policy diagnostics."""

from __future__ import annotations

import math
import unittest

from rsi_programmer.online_diagnostics import inspect_policy
from rsi_programmer.online_spiking import OnlineSpikingPolicy


class OnlineDiagnosticsTests(unittest.TestCase):
    """Check diagnostics on controls and adaptive policies."""

    def test_streaming_control_has_deterministic_first_lane(self) -> None:
        report = inspect_policy(None)
        self.assertIsNone(report["logits"])
        self.assertEqual(report["probabilities"], [1.0, 0.0, 0.0])
        self.assertEqual(report["entropy_nats"], 0.0)
        self.assertEqual(report["max_probability"], 1.0)
        self.assertEqual(report["readout_l2"], 0.0)
        self.assertEqual(report["clipped_weights"], 0)
        self.assertIsNone(report["baseline"])

    def test_initial_policy_is_uniform_and_diagnostic_is_read_only(
        self,
    ) -> None:
        policy = OnlineSpikingPolicy(seed=41)
        snapshot_before = policy.snapshot()
        metrics_before = policy.metrics()
        rng_before = policy._rng.getstate()
        report = inspect_policy(policy)
        self.assertEqual(
            report["probabilities"], [1.0 / 3.0] * 3
        )
        self.assertEqual(report["logits"], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(report["entropy_nats"], math.log(3.0))
        self.assertAlmostEqual(report["max_probability"], 1.0 / 3.0)
        self.assertEqual(report["readout_l2"], 0.0)
        self.assertEqual(report["clipped_weights"], 0)
        self.assertEqual(report["baseline"], 0.0)
        self.assertEqual(policy.snapshot(), snapshot_before)
        self.assertEqual(policy.metrics(), metrics_before)
        self.assertEqual(policy._rng.getstate(), rng_before)

    def test_inspection_preserves_the_next_seeded_choice(self) -> None:
        inspected = OnlineSpikingPolicy(seed=47)
        untouched = OnlineSpikingPolicy(seed=47)
        inspect_policy(inspected)
        for policy in (inspected, untouched):
            policy.start_episode("same-history")
        features = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        action_inspected = inspected.choose(
            features, ("add", "sub", "mul")
        )
        action_untouched = untouched.choose(
            features, ("add", "sub", "mul")
        )
        self.assertEqual(action_inspected, action_untouched)
        self.assertEqual(inspected._rng.getstate(), untouched._rng.getstate())
        self.assertEqual(inspected.metrics(), untouched.metrics())
        inspected.observe(0.5)
        untouched.observe(0.5)
        for policy in (inspected, untouched):
            policy.end_episode()
        self.assertEqual(inspected.snapshot(), untouched.snapshot())

    def test_adaptive_policy_reports_readout_statistics_without_weights(
        self,
    ) -> None:
        policy = OnlineSpikingPolicy(seed=43)
        policy.start_episode("diagnostic-update")
        policy.choose(
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            ("add", "sub", "mul"),
        )
        policy.observe(1.0)
        policy.end_episode()
        report = inspect_policy(policy)
        self.assertGreater(report["readout_l2"], 0.0)
        self.assertEqual(report["clipped_weights"], 0)
        self.assertGreaterEqual(report["baseline"], 0.0)
        self.assertAlmostEqual(
            sum(report["probabilities"]), 1.0, places=12
        )
        self.assertTrue(all(0.0 <= value <= 1.0
                            for value in report["probabilities"]))
        self.assertNotIn("weights", report)
        self.assertNotIn("bias", report)


if __name__ == "__main__":
    unittest.main()
