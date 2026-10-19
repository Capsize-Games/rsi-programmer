"""Data-only adversarial tests for the continual-audio trace auditor."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts.audit_audio_continual import (
    ARMS,
    NEURAL_ARMS,
    audit_data,
    load_protocol,
)

SPEC = {
    "version": 1,
    "sample_rate": 16000,
    "frame_seconds": 0.2,
    "band_edges_hz": [80, 160, 320, 640, 1280, 2560, 6000],
    "fit_frames": 3,
    "calibration_frames": 3,
    "probe_frames": 3,
    "block_frames": 2,
    "warmup_frames": 1,
    "cooldown_frames": 1,
    "ema_alpha": 1.0 - math.exp(-0.2 / 5.0),
    "seed": 0,
    "seeds": [0, 1, 2],
    "monitor_frames": 4,
    "memory_capacity_frames": 150,
    "memory_interval_targets": 300,
    "arms": list(ARMS),
}
COUNTERS = (
    "sessions", "predictions", "scored_targets", "updates",
    "neuron_ticks", "spikes", "readout_synapse_visits",
    "input_synapse_visits", "changed_readout_parameters",
    "changed_input_parameters",
)


def _metric(scores: list[float], *, probe: bool = False) \
        -> dict[str, Any]:
    frames = len(scores)
    targets = frames * 6
    sse = sum(score * score for score in scores) * 6.0
    return {
        "count": targets if probe else frames,
        "frame_count": frames,
        "feature_target_count": targets,
        "sse": sse,
        "mse": sse / targets if targets else None,
    }


def _apply_alarms(rows: list[dict[str, Any]], thresholds: dict[str, float],
                 spec: dict[str, Any]) \
        -> tuple[dict[str, int], dict[str, list[int]]]:
    totals = {arm: 0 for arm in ARMS}
    blocks = {arm: [0] * (spec["monitor_frames"] // spec["block_frames"])
              for arm in ARMS}
    last = {arm: None for arm in ARMS}
    start = (spec["fit_frames"] + spec["calibration_frames"]
             + spec["probe_frames"])
    for row in rows[start:]:
        position = row["j"]
        for arm in ARMS:
            item = row["arms"][arm]
            score = item["score"]
            elapsed = (spec["cooldown_frames"] if last[arm] is None
                       else position - last[arm])
            alarm = (item["scored"] and score is not None
                     and score > thresholds[arm]
                     and elapsed >= spec["cooldown_frames"])
            item["alarm"] = alarm
            if alarm:
                totals[arm] += 1
                blocks[arm][position // spec["block_frames"]] += 1
                last[arm] = position
    return totals, blocks


def _make_case() -> tuple[
        list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    spec = copy.deepcopy(SPEC)
    total = (spec["fit_frames"] + spec["calibration_frames"]
             + spec["probe_frames"] + spec["monitor_frames"])
    starts = {"fit": 0, "calibration": 3, "probe": 6, "monitor": 9}
    rows: list[dict[str, Any]] = []
    for index in range(total):
        phase = ("fit" if index < 3 else
                 "calibration" if index < 6 else
                 "probe" if index < 9 else "monitor")
        position = index - starts[phase]
        local_levels = (-49.0, -47.0, -45.0)
        rows.append({
            "i": index, "phase": phase, "j": position,
            "timestamp": (index + 1) * 0.2,
            "received_seconds": index * 0.2,
            "features": {"band_dbfs": [
                local_levels[position % 3] + band * 0.5
                for band in range(6)]},
            "normalized": None, "arms": {},
        })
    centers = []
    scales = []
    for band in range(6):
        values = [row["features"]["band_dbfs"][band]
                  for row in rows[:spec["fit_frames"]]]
        center = sorted(values)[1]
        deviations = sorted(abs(value - center) for value in values)
        centers.append(center)
        scales.append(max(3.0, 1.4826 * deviations[1]))
    normalizer = {"centers_db": centers, "scales_db": scales}

    normalized_rows = []
    for row in rows[spec["fit_frames"]:]:
        normalized = [
            0.5 + 0.5 * (value - center) / (scale + abs(value - center))
            for value, center, scale in zip(
                row["features"]["band_dbfs"], centers, scales)]
        row["normalized"] = normalized
        normalized_rows.append(normalized)

    snn_bands = ["1" * 64] * 6
    ann_bands = ["2" * 64] * 6
    fixed_hash = hashlib.sha256(
        json.dumps(snn_bands, separators=(",", ":")).encode()).hexdigest()
    hashes = {
        "adaptive_snn": fixed_hash, "readout_snn": fixed_hash,
        "memory_snn": fixed_hash, "recent_memory_snn": fixed_hash,
        "fixed_snn": fixed_hash,
        "adaptive_ann": "ann-fit-hash", "memory_ann": "ann-fit-hash",
        "ema": "baseline", "persistence": "baseline",
    }
    previous: dict[str, list[float] | None] = {arm: None for arm in ARMS}
    calibration_scores = {arm: [] for arm in ARMS}
    monitor_scores = {arm: [] for arm in ARMS}
    for row in rows:
        if row["phase"] in ("fit", "probe"):
            continue
        position = row["j"]
        if position == 0:
            previous = {arm: None for arm in ARMS}
        actual = row["normalized"]
        arm_rows = {}
        for arm in ARMS:
            before = previous[arm]
            if arm == "ema":
                if before is None:
                    after = actual[:]
                else:
                    alpha = spec["ema_alpha"]
                    after = [alpha * target + (1.0 - alpha) * old
                             for target, old in zip(actual, before)]
            elif arm == "persistence":
                after = actual[:]
            else:
                after = [0.65] * 6
            scored = before is not None and position >= spec["warmup_frames"]
            score = None
            sse = None
            if before is not None:
                sse = sum((guess - target) ** 2
                          for guess, target in zip(before, actual))
                score = math.sqrt(sse / 6.0)
                if scored and row["phase"] == "calibration":
                    calibration_scores[arm].append(score)
                if scored and row["phase"] == "monitor":
                    monitor_scores[arm].append(score)
            arm_rows[arm] = {
                "prediction_before": before,
                "prediction_next": after,
                "score": score, "sse": sse,
                "scored": scored, "alarm": False,
                "hash_before": hashes[arm], "hash_after": hashes[arm],
            }
            previous[arm] = after
        row["arms"] = arm_rows

    thresholds = {arm: max(1e-9, max(calibration_scores[arm]))
                  for arm in ARMS}
    alarm_totals, block_alarm_totals = _apply_alarms(
        rows, thresholds, spec)
    blocks = []
    for block_index, frame_end in enumerate((2, 4)):
        block_arms = {}
        start = 0 if block_index == 0 else block_index * 2 - 1
        end = frame_end - 1
        for arm in ARMS:
            values = monitor_scores[arm][start:end]
            record = _metric(values)
            alarm_count = block_alarm_totals[arm][block_index]
            rate = (alarm_count / (len(values) * 0.2 / 60.0)
                    if values else None)
            record.update({"alarm_count": alarm_count,
                           "alarms_per_minute": rate})
            block_arms[arm] = record
        fixed_mse = block_arms["fixed_snn"]["mse"]
        blocks.append({"block": block_index + 1,
                       "frames": frame_end, "arms": block_arms,
                       "relative_improvement_vs_fixed": {
                           arm: ((fixed_mse - block_arms[arm]["mse"])
                                 / fixed_mse if fixed_mse else None)
                           for arm in ARMS if arm != "fixed_snn"}})

    cumulative = {}
    for arm in ARMS:
        cumulative[arm] = _metric(monitor_scores[arm])
        rate = (alarm_totals[arm] / (len(monitor_scores[arm]) * 0.2 / 60.0)
                if monitor_scores[arm] else None)
        cumulative[arm].update({"alarm_count": alarm_totals[arm],
                                "alarms_per_minute": rate})
    paired = {}
    for arm in ARMS:
        if arm == "fixed_snn":
            continue
        fixed_mse = cumulative["fixed_snn"]["mse"]
        arm_mse = cumulative[arm]["mse"]
        paired[arm] = {
            "block_mse_delta": [
                block["arms"][arm]["mse"]
                - block["arms"]["fixed_snn"]["mse"] for block in blocks],
            "cumulative_mse_delta": arm_mse - fixed_mse,
            "relative_improvement": (
                (fixed_mse - arm_mse) / fixed_mse if fixed_mse else None),
        }
    probe_records = {}
    probe_start = spec["calibration_frames"]
    probe_end = probe_start + spec["probe_frames"]
    probe_vectors = normalized_rows[probe_start:probe_end]
    for arm in ARMS:
        if arm in ("ema", "persistence"):
            previous_probe = None
            probe_scores = []
            for position, actual in enumerate(probe_vectors):
                if (previous_probe is not None
                        and position >= spec["warmup_frames"]):
                    probe_scores.append(math.sqrt(sum(
                        (guess - target) ** 2 for guess, target in
                        zip(previous_probe, actual)) / 6.0))
                if arm == "persistence" or previous_probe is None:
                    previous_probe = actual[:]
                else:
                    previous_probe = [
                        spec["ema_alpha"] * target
                        + (1.0 - spec["ema_alpha"]) * old
                        for target, old in zip(actual, previous_probe)]
            probe_records[arm] = _metric(probe_scores, probe=True)
        else:
            probe_records[arm] = {
                "count": 12, "frame_count": 2,
                "feature_target_count": 12, "sse": 0.12,
                "mse": 0.01,
            }
    probe = {"frames": 3, "warmup_frames": 1,
             "arms": probe_records,
             "fixed_hash_before": fixed_hash,
             "fixed_hash_after": fixed_hash}
    checkpoints = [
        {"monitor_frame": frame, "sealed_probe": copy.deepcopy(probe),
         "fixed_hash": fixed_hash}
        for frame in (0, 2, 4)]

    model_metrics = {}
    for arm in NEURAL_ARMS:
        is_spiking = arm not in ("adaptive_ann", "memory_ann")
        learning = arm != "fixed_snn"
        encoder_learning = arm not in ("fixed_snn", "readout_snn")
        band_weights = snn_bands if is_spiking else ann_bands
        band_encoder = ["3" * 64] * 6
        bands = []
        for index in range(6):
            band = {counter: 0 for counter in COUNTERS}
            band.update({"spiking": is_spiking, "learning": learning,
                         "encoder_learning": encoder_learning,
                         "weight_hash": band_weights[index],
                         "encoder_hash": band_encoder[index]})
            if arm in ("memory_snn", "recent_memory_snn", "memory_ann"):
                band.update({"replay_passes": 0, "replay_targets": 0,
                             "online_training_targets": 0})
            bands.append(band)
        metric = {counter: 0 for counter in COUNTERS}
        metric.update({"spiking": is_spiking, "learning": learning,
                       "encoder_learning": encoder_learning,
                       "bands": bands, "weight_hashes": band_weights,
                       "encoder_hashes": band_encoder})
        if arm in ("memory_snn", "recent_memory_snn", "memory_ann"):
            mode = ("recent" if arm == "recent_memory_snn" else "fit")
            metric.update({"replay_passes": 0, "replay_targets": 0,
                           "online_training_targets": 0,
                           "memory_capacity_frames": 150,
                           "memory_mode": mode,
                           "replay_wall_seconds": 0.0})
        model_metrics[arm] = metric
    model_metrics.update({
        "processing_wall_seconds": 0.0,
        "initial_fit_weight_hashes": {
            arm: hashes[arm] for arm in NEURAL_ARMS},
        "memory_capacity_frames": 150,
        "memory_interval_targets": 300,
        "memory_samples_per_arm": {
            "memory_snn": 150, "recent_memory_snn": 150,
            "memory_ann": 150,
        },
        "memory_values_per_bank": 900,
        "trainable_parameters_per_model": 222,
    })
    summary = {
        "experiment": "audio_continual_v1", "phase": "finished",
        "frame_count": total, "monitor_frames": 4,
        "protocol": {**spec}, "normalizer": normalizer,
        "fixed_hash": fixed_hash, "reference_hash": fixed_hash,
        "thresholds": thresholds,
        "calibration": {arm: _metric(calibration_scores[arm])
                        for arm in ARMS},
        "sealed_probe": probe, "blocks": blocks,
        "cumulative": cumulative,
        "monitor_alarms": alarm_totals,
        "observed_alarms_per_minute": {
            arm: cumulative[arm]["alarms_per_minute"] for arm in ARMS},
        "adaptive_vs_fixed": paired, "checkpoints": checkpoints,
        "model_metrics": model_metrics,
    }
    return rows, summary, spec


class ContinualAudioAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows, self.summary, self.spec = _make_case()

    def test_valid_synthetic_trace(self) -> None:
        result = audit_data(self.rows, self.summary, self.spec, 0)
        self.assertTrue(result["valid"])
        self.assertEqual(result["monitor_frames"], 4)
        self.assertEqual(result["inversion_diagnostic"]["adaptive_snn"][
            "feature_target_count"], 18)

    def test_rejects_prior_forecast_mutation(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[10]["arms"]["adaptive_snn"]["prediction_before"] = [0.4] * 6
        with self.assertRaisesRegex(ValueError, "prior forecast"):
            audit_data(rows, self.summary, self.spec)

    def test_rejects_row_sse_mutation(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[4]["arms"]["adaptive_snn"]["sse"] += 0.01
        with self.assertRaisesRegex(ValueError, "sse mismatch"):
            audit_data(rows, self.summary, self.spec)

    def test_rejects_fixed_hash_drift(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[10]["arms"]["fixed_snn"]["hash_after"] = "drifted"
        with self.assertRaisesRegex(ValueError, "parameters changed"):
            audit_data(rows, self.summary, self.spec)

    def test_rejects_normalizer_fit_on_future_data(self) -> None:
        summary = copy.deepcopy(self.summary)
        summary["normalizer"]["centers_db"][0] += 0.1
        with self.assertRaisesRegex(ValueError, "normalizer"):
            audit_data(self.rows, summary, self.spec)

    def test_rejects_calibration_threshold_mismatch(self) -> None:
        summary = copy.deepcopy(self.summary)
        summary["thresholds"]["adaptive_snn"] += 0.01
        with self.assertRaisesRegex(ValueError, "thresholds"):
            audit_data(self.rows, summary, self.spec)

    def test_rejects_declared_fit_hash_mismatch(self) -> None:
        summary = copy.deepcopy(self.summary)
        summary["model_metrics"]["initial_fit_weight_hashes"][
            "adaptive_snn"] = "tampered"
        with self.assertRaisesRegex(ValueError, "declared fit hash mismatch"):
            audit_data(self.rows, summary, self.spec)

    def test_rejects_missing_probe_checkpoint(self) -> None:
        summary = copy.deepcopy(self.summary)
        summary["checkpoints"].pop()
        with self.assertRaisesRegex(ValueError, "checkpoint count"):
            audit_data(self.rows, summary, self.spec)

    def test_joint_prediction_and_target_complement_is_invariant(self) -> None:
        result = audit_data(self.rows, self.summary, self.spec)
        record = result["inversion_diagnostic"]["adaptive_snn"]
        self.assertAlmostEqual(
            record["original_mse"],
            record["prediction_and_target_complement_mse"], places=12)

    def test_fixture_is_pinned_and_declares_all_seeds(self) -> None:
        protocol = load_protocol()
        self.assertEqual(protocol["arms"], list(ARMS))
        self.assertEqual(protocol["seeds"], [0, 1, 2])


    def _audit_production_length(self, monitor_frames: int) -> dict[str, Any]:
        from rsi_programmer.audio_continual import AudioContinualSession

        protocol = dict(load_protocol(), monitor_frames=monitor_frames)
        session = AudioContinualSession(dict(protocol, seed=0))
        rows = []
        total = (protocol["fit_frames"] + protocol["calibration_frames"]
                 + protocol["probe_frames"] + protocol["monitor_frames"])
        for index in range(total):
            features = {"band_dbfs": [
                -54.0 + band * 1.5 + math.sin(index / 13.0 + band) * 2.0
                for band in range(6)]}
            rows.append(session.observe(features, index * 0.2))
        session.finish()
        return audit_data(rows, session.summary(), expected_seed=0)

    def test_continual_session_output_is_independently_auditable(self) -> None:
        result = self._audit_production_length(9000)
        self.assertTrue(result["valid"])
        self.assertEqual(result["monitor_frames"], 9000)

    def test_zero_monitor_session_retains_fit_hash_validation(self) -> None:
        result = self._audit_production_length(0)
        self.assertTrue(result["valid"])
        self.assertEqual(result["monitor_frames"], 0)
        self.assertTrue(result["initial_fit_hashes"])

    def test_first_monitor_frame_retains_hash_and_threshold_validation(
            self) -> None:
        result = self._audit_production_length(1)
        self.assertTrue(result["valid"])
        self.assertEqual(result["monitor_frames"], 1)

    def test_file_api_reads_only_the_summary_prefix(self) -> None:
        from scripts.audit_audio_continual import audit_session

        rows, summary, _ = _make_case()
        summary["frame_count"] = len(rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.jsonl"
            summary_path = root / "summary.json"
            with trace_path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
                stream.write("not part of the summary prefix\n")
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            result = audit_session(trace_path, summary_path, 0,
                                   protocol=self.spec)
        self.assertTrue(result["valid"])
        self.assertEqual(result["frames"], len(rows))

    def test_online_comparison_ignores_timing_only_fields(self) -> None:
        from scripts.measure_audio_continual import _model_core, _same_json

        base = {"arms": {"snn": {"updates": 4}},
                "runtime": {"pid": 1}}
        timed = {"arms": {"snn": {"updates": 4}},
                 "runtime": {"pid": 2}, "processing_wall_seconds": 0.8}
        self.assertTrue(_same_json(_model_core(base), _model_core(timed)))
        timed["arms"]["snn"]["updates"] = 5
        self.assertFalse(_same_json(_model_core(base), _model_core(timed)))


if __name__ == "__main__":
    unittest.main()
