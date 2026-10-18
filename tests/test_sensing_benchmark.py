"""Causal loop, split and alarm-contract checks for the sensing pilot."""
from __future__ import annotations

import copy
import unittest
from typing import Any
from unittest.mock import patch

from rsi_programmer.sensing_baseline import SensingBaseline
from rsi_programmer.sensing_benchmark import (
    TEST_SPEEDS, TRAIN_SPEEDS, calibrate, load_protocol, run_sequence,
    state_hash, stream_spec,
)
from rsi_programmer.sensing_detector import (
    SensingAlarm, calibrated_threshold, event_metrics,
)
from rsi_programmer.sensing_summary import summarize


class SensingBenchmarkTests(unittest.TestCase):
    def test_real_protocol_counts_and_split(self) -> None:
        protocol = load_protocol()
        self.assertTrue(set(TRAIN_SPEEDS).isdisjoint(TEST_SPEEDS))
        self.assertEqual(protocol["evaluation"]["expected_stream_rows"], 924)
        identities = [(speed, kind, order) for speed in TEST_SPEEDS
                      for kind in ("positive", "sham") for order in (0, 1)]
        self.assertEqual(len(identities), 44)
        for speed, kind, order in identities:
            slices, events = stream_spec(speed, kind, order)
            self.assertEqual({part["speed"] for part in slices}, {speed})
            self.assertEqual(sum(p["stop"] - p["start"] for p in slices),
                             3000)
            if kind == "positive":
                self.assertEqual(events, [75, 150, 225])
                self.assertNotEqual(slices[0], slices[2])
            else:
                self.assertEqual(events, [])
                self.assertEqual({p["config"] for p in slices}, {1})
                self.assertEqual({p["start"] for p in slices},
                                 {0, 750, 1500, 2250})

    def test_alarm_precedes_target_update(self) -> None:
        calls: list[str] = []
        old_observe = SensingAlarm.observe
        old_advance = SensingBaseline.advance

        def observe(alarm: SensingAlarm, index: int,
                    score: float | None) -> bool:
            calls.append("alarm")
            return old_observe(alarm, index, score)

        def advance(model: SensingBaseline, timestamp: float,
                    value: float) -> float:
            calls.append("update")
            return old_advance(model, timestamp, value)

        with patch.object(SensingAlarm, "observe", observe), patch.object(
                SensingBaseline, "advance", advance):
            result = run_sequence(SensingBaseline("persistence"),
                                  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 0.01)
        self.assertEqual(calls, ["alarm"] + ["alarm", "update"] * 5)
        self.assertEqual(result["windows"][1]["prediction_before"], 0.1)
        self.assertAlmostEqual(result["windows"][1]["score"], 0.1)

    def test_future_value_does_not_change_earlier_predictions(self) -> None:
        prefix = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        first = run_sequence(SensingBaseline("nlms"), prefix + [0.7], 0.1)
        second = run_sequence(SensingBaseline("nlms"), prefix + [0.9], 0.1)
        self.assertEqual(first["windows"][:6], second["windows"][:6])
        self.assertNotEqual(first["windows"][6]["score"],
                            second["windows"][6]["score"])

    def test_calibration_does_not_mutate_trained_source(self) -> None:
        model = SensingBaseline("nlms")
        before = copy.deepcopy(model.snapshot())
        features = {(1, speed): [0.4] * 300 for speed in TRAIN_SPEEDS}
        thresholds, rows = calibrate({("nlms", 0): model}, features)
        self.assertEqual(model.snapshot(), before)
        self.assertEqual(len(rows[0]["scores"]), 6 * 70)
        self.assertEqual(thresholds[("nlms", 0)],
                         calibrated_threshold(rows[0]["scores"]))

    def test_warmup_threshold_and_refractory(self) -> None:
        alarm = SensingAlarm(0.1)
        fired = [alarm.observe(i, None if i == 0 else 0.2)
                 for i in range(12)]
        self.assertEqual([i for i, yes in enumerate(fired) if yes], [5, 10])
        self.assertEqual(calibrated_threshold([0.0] * 100), 1e-9)
        self.assertEqual(calibrated_threshold(list(range(100))), 99)

    def test_event_window_exclusive_end_and_one_match(self) -> None:
        windows: list[dict[str, Any]] = [
            {"i": i, "scored": i >= 5, "score": 0.1,
             "alarm": i in (74, 75, 80)} for i in range(300)]
        metrics = event_metrics(windows, [75])
        self.assertEqual(metrics["detected"], 1)
        self.assertEqual(metrics["false_alarms"], 2)
        self.assertAlmostEqual(metrics["delay_seconds"][0], 0.2)
        self.assertEqual(metrics["stable_seconds"], 58.0)
        self.assertEqual(event_metrics(windows, [80])["detected"], 1)

    def test_invalid_alarm_input_does_not_consume_index(self) -> None:
        alarm = SensingAlarm(0.1)
        with self.assertRaises(ValueError):
            alarm.observe(0, float("nan"))
        self.assertFalse(alarm.observe(0, None))
        with self.assertRaises(ValueError):
            alarm.observe(1, None)
        self.assertFalse(alarm.observe(1, 0.2))

    def test_empty_matrix_is_not_a_passing_benchmark(self) -> None:
        with self.assertRaisesRegex(ValueError, "matrix"):
            summarize([])
        self.assertEqual(state_hash({"x": 1, "y": 2}),
                         state_hash({"y": 2, "x": 1}))
