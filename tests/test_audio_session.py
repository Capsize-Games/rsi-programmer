"""Phase, causality, and immutability checks for AudioSession."""
from __future__ import annotations

import copy
import unittest
from typing import Any

from rsi_programmer.audio_session import AudioSession


def frame(index: int) -> dict[str, list[float]]:
    """Build deterministic six-band levels with a changing pattern."""
    return {"band_dbfs": [
        -50.0 + band * 2.0 + ((index * (band + 1)) % 7) * 0.4
        for band in range(6)
    ]}


def learner_state(
    learner: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Copy learner and per-band values without comparing object identity."""
    attributes = learner.__dict__
    wrapper = {key: copy.deepcopy(value) for key, value in attributes.items()
               if key != "_bands"}
    bands = [copy.deepcopy(band.__dict__) for band in learner._bands]
    return wrapper, bands


class AudioSessionTests(unittest.TestCase):
    """Exercise short sessions across all protocol phases."""

    def make_session(self) -> AudioSession:
        return AudioSession({
            "fit_frames": 10,
            "calibration_frames": 10,
            "probe_frames": 5,
            "warmup_frames": 1,
            "cooldown_frames": 2,
            "block_frames": 3,
            "seed": 4,
        })

    def feed(self, session: AudioSession, count: int,
             offset: int = 0) -> list[dict[str, object]]:
        return [session.observe(frame(offset + index),
                                (offset + index) * 0.2)
                for index in range(count)]

    def test_fit_calibration_probe_monitor_sequence(self) -> None:
        session = self.make_session()
        fit_rows = self.feed(session, 10)
        self.assertEqual(session.phase, "calibration")
        self.assertTrue(all(not row["arms"] for row in fit_rows))
        self.feed(session, 10, 10)
        self.assertEqual(session.phase, "probe")
        self.feed(session, 5, 20)
        self.assertTrue(session.ready)
        summary = session.summary()
        self.assertEqual(summary["sealed_probe"]["frames"], 5)
        self.assertEqual(
            summary["sealed_probe"]["arms"]["fixed_snn"]
            ["feature_target_count"], 24)
        self.assertEqual(set(summary["thresholds"]), {
            "adaptive_snn", "fixed_snn", "adaptive_ann", "ema", "persistence"
        })
        row = session.observe(frame(25), 5.0)
        self.assertEqual(set(row["arms"]), set(summary["thresholds"]))
        self.assertEqual(row["timestamp"], 5.2)

    def test_fixed_bank_is_unchanged_by_probe_and_checkpoints(self) -> None:
        session = self.make_session()
        self.feed(session, 25)
        initial_hash = session.summary()["fixed_hash"]
        self.feed(session, 15, 25)
        self.assertEqual(session.summary()["fixed_hash"], initial_hash)
        for index in range(6):
            session.observe(frame(40 + index), 8.0 + index * 0.2)
        summary = session.summary()
        self.assertTrue(summary["checkpoints"])
        initial_mse = summary["checkpoints"][0][
            "sealed_probe"]["arms"]["fixed_snn"]["mse"]
        for checkpoint in summary["checkpoints"]:
            probe = checkpoint["sealed_probe"]
            self.assertEqual(probe["fixed_hash_before"], initial_hash)
            self.assertEqual(probe["fixed_hash_after"], initial_hash)
            self.assertEqual(checkpoint["fixed_hash"], initial_hash)
            self.assertEqual(
                probe["arms"]["fixed_snn"]["mse"], initial_mse)

    def test_snapshot_does_not_end_or_change_live_models(self) -> None:
        session = self.make_session()
        self.feed(session, 25)
        self.feed(session, 5, 25)
        row = session.observe(frame(30), 6.0)
        before = row["arms"]["adaptive_snn"]["hash_after"]
        self.assertTrue(session._models["adaptive_snn"].active)
        saved = session.snapshot()
        self.assertTrue(session._models["adaptive_snn"].active)
        self.assertEqual(saved["phase"], "monitor")
        self.assertEqual(
            session._bank_hash(session._models["adaptive_snn"]), before)
        model_state = learner_state(session._models["adaptive_snn"])
        baseline_state = copy.deepcopy(session._monitor_predictions)
        saved["models"]["adaptive_snn"]["bands"][0][
            "input_weights"][0] += 1.0
        saved["baseline_state"]["ema"][0] += 1.0
        self.assertEqual(learner_state(session._models["adaptive_snn"]),
                         model_state)
        self.assertEqual(session._monitor_predictions, baseline_state)

    def test_monitor_score_precedes_update_and_blocks_are_paired(self) -> None:
        session = self.make_session()
        self.feed(session, 25)
        rows = [session.observe(frame(25 + index), 5.0 + index * 0.2)
                for index in range(4)]
        scored = rows[2]["arms"]["adaptive_snn"]
        self.assertTrue(scored["scored"])
        self.assertIsNotNone(scored["prediction_before"])
        self.assertEqual(
            session.summary()["blocks"][0]["arms"]["fixed_snn"]
            ["feature_target_count"], 12)

    def test_first_scored_row_uses_previous_forecast_and_sse(self) -> None:
        session = self.make_session()
        self.feed(session, 25)
        prior = session.observe(frame(25), 5.0)
        current = session.observe(frame(26), 5.2)
        for arm in current["arms"]:
            scored = current["arms"][arm]
            self.assertEqual(scored["prediction_before"],
                             prior["arms"][arm]["prediction_next"])
            errors = [prediction - actual for prediction, actual in zip(
                scored["prediction_before"], current["normalized"])]
            sse = sum(error * error for error in errors)
            self.assertAlmostEqual(scored["sse"], sse, places=14)
            self.assertAlmostEqual(scored["score"] ** 2 * 6,
                                   scored["sse"], places=14)
            if scored["scored"]:
                total = session.summary()["cumulative"][arm]
                self.assertEqual(total["frame_count"], 1)
                self.assertEqual(total["feature_target_count"], 6)
                self.assertAlmostEqual(total["sse"], sse, places=14)

    def test_invalid_observation_does_not_mutate_monitor(self) -> None:
        session = self.make_session()
        self.feed(session, 25)
        session.observe(frame(25), 5.0)
        index = session._index
        received = session._last_received_seconds
        model = learner_state(session._models["adaptive_snn"])
        predictions = copy.deepcopy(session._monitor_predictions)
        with self.assertRaises(ValueError):
            session.observe(
                {"band_dbfs": [0.0, 0.0, 0.0, 0.0, 0.0, float("nan")]},
                5.2)
        self.assertEqual(session._index, index)
        self.assertEqual(session._last_received_seconds, received)
        self.assertEqual(learner_state(session._models["adaptive_snn"]), model)
        self.assertEqual(session._monitor_predictions, predictions)

    def test_monitor_aggregate_uses_constant_space_totals(self) -> None:
        session = self.make_session()
        self.feed(session, 25)
        for index in range(25, 35):
            session.observe(frame(index), index * 0.2)
        self.assertFalse(hasattr(session, "_monitor_errors"))
        self.assertEqual(session._monitor_scored_frames["fixed_snn"], 9)
        summary = session.summary()["cumulative"]["fixed_snn"]
        self.assertEqual(summary["frame_count"], 9)
        self.assertEqual(summary["feature_target_count"], 54)
        self.assertEqual(summary["sse"], session._monitor_sse["fixed_snn"])

    def test_strict_threshold_and_cooldown(self) -> None:
        session = self.make_session()
        session._thresholds["ema"] = 0.5
        self.assertFalse(session._alarm("ema", 0.5, True))
        self.assertTrue(session._alarm("ema", 0.6, True))
        self.assertFalse(session._alarm("ema", 0.9, True))
        self.assertTrue(session._alarm("ema", 0.9, True))


if __name__ == "__main__":
    unittest.main()
