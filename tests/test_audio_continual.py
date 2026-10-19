"""Controlled-arm construction and memory isolation checks."""
from __future__ import annotations

import unittest
from typing import Any

from rsi_programmer.audio_continual import (
    CONTINUAL_ARMS,
    AudioContinualSession,
)
from rsi_programmer.audio_memory import AudioMemoryLearner
from rsi_programmer.audio_session import AudioSession


def _protocol(**overrides: Any) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "fit_frames": 150,
        "calibration_frames": 26,
        "probe_frames": 26,
        "warmup_frames": 25,
        "block_frames": 300,
        "arms": list(CONTINUAL_ARMS),
        "memory_capacity_frames": 150,
        "memory_interval_targets": 300,
    }
    protocol.update(overrides)
    return protocol


def _features(index: int) -> dict[str, list[float]]:
    return {"band_dbfs": [
        -62.0 + ((index * (band + 3)) % 31) * 0.4
        for band in range(6)
    ]}


def _feed(session: AudioContinualSession, count: int,
          offset: int = 0) -> None:
    for index in range(offset, offset + count):
        session.observe(_features(index), index * 0.2)


class AudioContinualSessionTests(unittest.TestCase):
    def test_fit_initializes_all_snn_arms_with_matching_parameters(
            self) -> None:
        session = AudioContinualSession(_protocol())
        _feed(session, 150)
        snn_arms = ("adaptive_snn", "readout_snn", "memory_snn",
                    "recent_memory_snn", "fixed_snn")
        hashes = {
            arm: session._models[arm].weight_hash() for arm in snn_arms
        }
        self.assertEqual(len({tuple(value) for value in hashes.values()}), 1)
        ann_hashes = {
            arm: session._models[arm].weight_hash()
            for arm in ("adaptive_ann", "memory_ann")
        }
        self.assertEqual(ann_hashes["adaptive_ann"],
                         ann_hashes["memory_ann"])
        self.assertFalse(session._models["readout_snn"]._encoder_learning)

    def test_calibration_and_probe_copies_do_not_touch_source_memories(
            self) -> None:
        session = AudioContinualSession(_protocol())
        _feed(session, 150)
        fit_memory = list(
            session._models["memory_snn"]._bands[0]._memory_rows)
        self.assertEqual(len(fit_memory), 150)
        self.assertEqual(
            session._models["recent_memory_snn"]._bands[0]._memory_rows, [])

        _feed(session, 25, offset=150)
        self.assertTrue(all(
            model._memory_mode == "disabled"
            for model in session._cal_models.values()
            if hasattr(model, "_memory_mode")))
        _feed(session, 1, offset=175)

        _feed(session, 26, offset=176)
        self.assertTrue(session.ready)
        self.assertEqual(
            session._models["memory_snn"]._bands[0]._memory_rows, fit_memory)
        self.assertEqual(
            session._models["recent_memory_snn"]._bands[0]._memory_rows, [])
        self.assertTrue(all(
            session._models[name].metrics()["replay_targets"] == 0
            for name in ("memory_snn", "recent_memory_snn", "memory_ann")))
        work = session.summary()["work_accounting"]
        self.assertEqual(
            work["calibration_copies"]["memory_snn"]["predictions"],
            26 * 6)
        self.assertEqual(
            work["sealed_probe_copies"]["memory_snn"]["predictions"],
            26 * 6)

        _feed(session, 1, offset=202)
        recent = session._models[
            "recent_memory_snn"]._bands[0]._memory_rows
        self.assertEqual(len(recent), 1)

    def test_replay_cadence_and_metrics_are_matched_across_memory_arms(
            self) -> None:
        session = AudioContinualSession(_protocol())
        _feed(session, 150 + 26 + 26)
        input_changes_before = session._models["readout_snn"].metrics()[
            "changed_input_parameters"]
        _feed(session, 301, offset=202)
        metrics = session.summary()["model_metrics"]
        for arm in ("memory_snn", "recent_memory_snn", "memory_ann"):
            self.assertEqual(metrics[arm]["replay_passes"], 6)
            self.assertEqual(metrics[arm]["replay_targets"], 6 * 149)
        self.assertFalse(metrics["readout_snn"]["encoder_learning"])
        self.assertEqual(metrics["readout_snn"]["changed_input_parameters"],
                         input_changes_before)
        self.assertGreaterEqual(metrics["processing_wall_seconds"], 0)

    def test_legacy_session_keeps_its_five_arm_surface(self) -> None:
        self.assertEqual(AudioSession.ARM_NAMES, (
            "adaptive_snn", "fixed_snn", "adaptive_ann", "ema",
            "persistence"))
        self.assertEqual(AudioContinualSession.ARM_NAMES, CONTINUAL_ARMS)

    def test_invalid_memory_bank_rows_leave_all_band_buffers_unchanged(
            self) -> None:
        learner = AudioMemoryLearner(seed=3)
        learner.configure_memory("fit", [[0.3] * 6 for _ in range(150)])
        before = [list(model._memory_rows) for model in learner._bands]
        rows = [[0.4] * 6 for _ in range(150)]
        rows[-1] = [0.4, 0.4, 0.4, 0.4, 0.4, float("nan")]
        with self.assertRaises(ValueError):
            learner.configure_memory("fit", rows)
        self.assertEqual(
            [model._memory_rows for model in learner._bands], before)
