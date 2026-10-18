"""Contract tests for causal sensing prediction baselines."""
from __future__ import annotations

import copy
import math
import unittest

from rsi_programmer.sensing_baseline import EMA_TAU_SECONDS
from rsi_programmer.sensing_baseline import INITIAL_NLMS_WEIGHTS
from rsi_programmer.sensing_baseline import NLMS_EPSILON
from rsi_programmer.sensing_baseline import NLMS_LEARNING_RATE
from rsi_programmer.sensing_baseline import PARAMETER_LIMIT
from rsi_programmer.sensing_baseline import UPDATE_LIMIT
from rsi_programmer.sensing_baseline import SensingBaseline


class SensingBaselineTests(unittest.TestCase):
    """Check causal forecasts, NLMS updates, and boundary persistence."""

    def test_persistence_forecasts_current_value(self) -> None:
        baseline = SensingBaseline("persistence")
        self.assertEqual(baseline.start_session(0.0, 0.3), 0.3)
        self.assertEqual(baseline.advance(0.2, 0.8), 0.8)
        metrics = baseline.metrics()
        self.assertEqual(metrics["predictions"], 2)
        self.assertEqual(metrics["scored_targets"], 1)
        self.assertEqual(metrics["updates"], 0)
        baseline.end_session()

    def test_ema_uses_exact_elapsed_time_update(self) -> None:
        baseline = SensingBaseline("ema", learning=False)
        self.assertEqual(baseline.start_session(1.0, 0.2), 0.2)
        expected = (math.exp(-0.4 / EMA_TAU_SECONDS) * 0.2
                    + (1.0 - math.exp(-0.4 / EMA_TAU_SECONDS)) * 0.8)
        self.assertAlmostEqual(baseline.advance(1.4, 0.8), expected)
        self.assertEqual(baseline.metrics()["filter_updates"], 1)
        self.assertEqual(baseline.metrics()["updates"], 0)
        baseline.end_session()

    def test_ema_filter_state_resets_at_session_boundary(self) -> None:
        baseline = SensingBaseline("ema")
        baseline.start_session(0.0, 0.1)
        baseline.advance(0.2, 0.9)
        baseline.end_session()
        self.assertIsNone(baseline._ema)
        self.assertEqual(baseline.start_session(5.0, 0.3), 0.3)
        baseline.end_session()

    def test_nlms_update_uses_saved_features_and_simultaneous_deltas(
            self) -> None:
        baseline = SensingBaseline("nlms")
        prediction = baseline.start_session(0.0, 0.5)
        self.assertEqual(prediction, 0.5)
        old = list(INITIAL_NLMS_WEIGHTS)
        features = [0.5, 0.5, 1.0]
        target = 0.9
        denominator = NLMS_EPSILON + sum(item * item for item in features)
        error = target - prediction
        deltas = [max(-UPDATE_LIMIT, min(
            UPDATE_LIMIT, NLMS_LEARNING_RATE * error * item / denominator))
            for item in features]
        expected = [max(-PARAMETER_LIMIT, min(
            PARAMETER_LIMIT, weight + delta))
            for weight, delta in zip(old, deltas)]

        baseline.advance(0.2, target)
        self.assertEqual(baseline._weights, expected)
        self.assertEqual(baseline.metrics()["updates"], 1)
        self.assertEqual(baseline.metrics()["parameter_visits"], 3)
        self.assertEqual(baseline.metrics()["changed_parameters"], 3)
        baseline.end_session()

    def test_frozen_nlms_keeps_advancing_ar_history(self) -> None:
        baseline = SensingBaseline("nlms", learning=False)
        before = baseline.weight_hash()
        baseline.start_session(0.0, 0.4)
        baseline.advance(0.2, 0.8)
        self.assertEqual(baseline.weight_hash(), before)
        self.assertEqual(baseline._pending_features, [0.8, 0.4, 1.0])
        baseline.advance(0.4, 0.7)
        self.assertEqual(baseline.weight_hash(), before)
        self.assertEqual(baseline._pending_features, [0.7, 0.8, 1.0])
        self.assertEqual(baseline.metrics()["scored_targets"], 2)
        self.assertEqual(baseline.metrics()["updates"], 0)
        baseline.end_session()

    def test_invalid_start_is_atomic(self) -> None:
        baseline = SensingBaseline("nlms")
        before = baseline.snapshot()
        invalid = ((float("nan"), 0.5), (0.0, -0.1),
                   (0.0, 1.1), (True, 0.5), (0.0, float("inf")))
        for timestamp, value in invalid:
            with self.subTest(timestamp=timestamp, value=value):
                with self.assertRaises(ValueError):
                    baseline.start_session(timestamp, value)
                self.assertEqual(baseline.snapshot(), before)

    def test_invalid_advance_is_atomic(self) -> None:
        baseline = SensingBaseline("ema")
        baseline.start_session(1.0, 0.5)
        before = copy.deepcopy(baseline.__dict__)
        invalid = ((1.0, 0.4), (0.9, 0.4), (1.2, float("inf")),
                   (1.2, 1.1), (True, 0.4))
        for timestamp, value in invalid:
            with self.subTest(timestamp=timestamp, value=value):
                with self.assertRaises(ValueError):
                    baseline.advance(timestamp, value)
                self.assertEqual(baseline.__dict__, before)
        baseline.end_session()

    def test_overflowed_elapsed_time_is_rejected_atomically(self) -> None:
        baseline = SensingBaseline("ema")
        baseline.start_session(-1e308, 0.5)
        before = copy.deepcopy(baseline.__dict__)
        with self.assertRaisesRegex(ValueError, "elapsed time"):
            baseline.advance(1e308, 0.6)
        self.assertEqual(baseline.__dict__, before)
        baseline.end_session()

    def test_session_boundary_resets_history_but_keeps_nlms_weights(
            self) -> None:
        baseline = SensingBaseline("nlms")
        baseline.start_session(0.0, 0.4)
        baseline.advance(0.2, 0.9)
        trained_hash = baseline.weight_hash()
        baseline.end_session()
        self.assertIsNone(baseline._last_value)
        self.assertIsNone(baseline._previous_value)
        self.assertIsNone(baseline._pending_prediction)
        self.assertIsNone(baseline._pending_features)
        baseline.start_session(10.0, 0.6)
        self.assertEqual(baseline.weight_hash(), trained_hash)
        self.assertEqual(baseline._pending_features, [0.6, 0.6, 1.0])
        baseline.end_session()

    def test_snapshot_restore_and_hash_are_stable(self) -> None:
        baseline = SensingBaseline("nlms")
        baseline.start_session(0.0, 0.5)
        baseline.advance(0.2, 0.9)
        baseline.end_session()
        snapshot = baseline.snapshot()
        restored = SensingBaseline("nlms")
        restored.restore(snapshot)
        self.assertEqual(restored.snapshot(), snapshot)
        self.assertEqual(restored.weight_hash(), snapshot["weight_hash"])

    def test_restore_rejects_bad_hash_atomically(self) -> None:
        baseline = SensingBaseline("nlms")
        snapshot = baseline.snapshot()
        damaged = copy.deepcopy(snapshot)
        damaged["weights"][0] += 0.1
        before = baseline.snapshot()
        with self.assertRaisesRegex(ValueError, "weight hash"):
            baseline.restore(damaged)
        self.assertEqual(baseline.snapshot(), before)

    def test_restore_rejects_different_mode(self) -> None:
        baseline = SensingBaseline("ema")
        before = baseline.snapshot()
        with self.assertRaisesRegex(ValueError, "configuration"):
            baseline.restore(SensingBaseline("persistence").snapshot())
        self.assertEqual(baseline.snapshot(), before)

    def test_evaluation_copy_disables_learning_without_mutating_source(
            self) -> None:
        trained = SensingBaseline("nlms")
        trained.start_session(0.0, 0.5)
        trained.advance(0.2, 0.9)
        trained.end_session()
        original = trained.snapshot()
        frozen = trained.evaluation_copy()
        frozen_hash = frozen.weight_hash()
        frozen.start_session(5.0, 0.3)
        frozen.advance(5.2, 0.8)
        self.assertEqual(frozen.weight_hash(), frozen_hash)
        self.assertEqual(frozen.metrics()["updates"], 0)
        self.assertFalse(frozen.metrics()["learning"])
        self.assertEqual(trained.snapshot(), original)
        frozen.end_session()

    def test_boundary_operations_require_inactive_session(self) -> None:
        baseline = SensingBaseline("persistence")
        snapshot = baseline.snapshot()
        baseline.start_session(0.0, 0.5)
        with self.assertRaises(RuntimeError):
            baseline.snapshot()
        with self.assertRaises(RuntimeError):
            baseline.evaluation_copy()
        with self.assertRaises(RuntimeError):
            baseline.restore(snapshot)
        baseline.end_session()


if __name__ == "__main__":
    unittest.main()
