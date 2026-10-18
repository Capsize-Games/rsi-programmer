"""Adversarial checks for the private audio-session trace auditor."""
from __future__ import annotations

import copy
import math
import unittest
from typing import Any

from rsi_programmer.audio_session import AudioSession
from scripts.audit_audio_session import audit_data, load_protocol

SPEC = {
    "version": 1, "frame_seconds": 0.2, "fit_frames": 3,
    "calibration_frames": 3, "probe_frames": 3, "block_frames": 2,
    "warmup_frames": 1, "cooldown_frames": 1, "seed": 0,
    "ema_alpha": 1.0 - math.exp(-0.2 / 5.0),
    "band_edges_hz": [80, 160, 320, 640, 1280, 2560, 6000],
    "arms": ["adaptive_snn", "fixed_snn", "adaptive_ann", "ema",
             "persistence"],
}
ARMS = ("adaptive_snn", "fixed_snn", "adaptive_ann", "ema",
        "persistence")


def _normalizer(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    centers = []
    scales = []
    for band in range(6):
        values = sorted(row["features"]["band_dbfs"][band] for row in rows)
        center = values[1]
        deviations = sorted(abs(value - center) for value in values)
        centers.append(center)
        scales.append(max(3.0, 1.4826 * deviations[1]))
    return {"centers_db": centers, "scales_db": scales}


def _normalized(row: dict[str, Any], norm: dict[str, list[float]]) \
        -> list[float]:
    return [0.5 + 0.5 * (value - center) / (scale + abs(value - center))
            for value, center, scale in zip(
                row["features"]["band_dbfs"], norm["centers_db"],
                norm["scales_db"])]


def _metric(scores: list[float]) -> dict[str, Any]:
    sse = sum(score * score for score in scores) * 6.0
    count = len(scores) * 6
    return {"count": len(scores), "frame_count": len(scores),
            "feature_target_count": count, "sse": sse,
            "mse": sse / count if count else None}


def _make_case() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fit_n = SPEC["fit_frames"]
    cal_n = SPEC["calibration_frames"]
    probe_n = SPEC["probe_frames"]
    rows: list[dict[str, Any]] = []
    for index in range(fit_n + cal_n + probe_n + 4):
        phase = ("fit" if index < fit_n else
                 "calibration" if index < fit_n + cal_n else
                 "probe" if index < fit_n + cal_n + probe_n else "monitor")
        start = {"fit": 0, "calibration": fit_n,
                 "probe": fit_n + cal_n,
                 "monitor": fit_n + cal_n + probe_n}[phase]
        position = index - start
        rows.append({
            "i": index, "phase": phase, "j": position,
            "timestamp": (index + 1) * 0.2,
            "received_seconds": index * 0.2,
            "features": {"band_dbfs": [
                -55.0 + band * 2.0 + index * (band + 1) * 0.17
                for band in range(6)]},
            "normalized": None, "arms": {},
        })
    norm = _normalizer(rows[:fit_n])
    prior: dict[str, list[float] | None] = {arm: None for arm in ARMS}
    scores: dict[str, list[float]] = {arm: [] for arm in ARMS}
    cal_scores: dict[str, list[float]] = {arm: [] for arm in ARMS}
    thresholds: dict[str, float] = {}
    for row in rows:
        if row["phase"] == "fit":
            continue
        if row["phase"] == "probe":
            row["normalized"] = _normalized(row, norm)
            continue
        if row["j"] == 0:
            prior = {arm: None for arm in ARMS}
        row["normalized"] = _normalized(row, norm)
        arm_rows = {}
        for arm in ARMS:
            before = prior[arm]
            if before is None or arm == "persistence":
                after = row["normalized"]
            elif arm == "ema":
                alpha = SPEC["ema_alpha"]
                after = [alpha * actual + (1 - alpha) * prior_value
                         for actual, prior_value in
                         zip(row["normalized"], before)]
            else:
                after = [0.5] * 6
            scored = before is not None and row["j"] >= SPEC["warmup_frames"]
            score = None
            sse = None
            alarm = False
            if before is not None:
                sse = sum((guess - actual) ** 2 for guess, actual in
                          zip(before, row["normalized"]))
                score = (sse / 6.0) ** 0.5
                if scored and row["phase"] == "calibration":
                    cal_scores[arm].append(score)
                if scored and row["phase"] == "monitor":
                    scores[arm].append(score)
            digest = {"adaptive_snn": "ha", "adaptive_ann": "hn",
                      "fixed_snn": "hf", "ema": "baseline",
                      "persistence": "baseline"}[arm]
            arm_rows[arm] = {
                "prediction_before": before, "prediction_next": after,
                "score": score, "sse": sse, "scored": scored,
                "alarm": alarm, "hash_before": digest,
                "hash_after": digest,
            }
            prior[arm] = after
        row["arms"] = arm_rows
        if row["phase"] == "calibration" and row["j"] == cal_n - 1:
            thresholds = {arm: max(1e-9, max(cal_scores[arm]))
                          for arm in ARMS}

    thresholds = {arm: max(1e-9, max(cal_scores[arm])) for arm in ARMS}
    for row in rows:
        if row["phase"] == "monitor":
            for arm in ARMS:
                item = row["arms"][arm]
                # Rebuild alarms now thresholds are known for every arm.
                item["alarm"] = False
    last_alarm = {arm: None for arm in ARMS}
    monitor_alarms = {arm: 0 for arm in ARMS}
    for row in rows:
        if row["phase"] != "monitor":
            continue
        for arm in ARMS:
            item = row["arms"][arm]
            if (item["scored"] and item["score"] > thresholds[arm]
                    and (last_alarm[arm] is None or row["j"]
                         - last_alarm[arm] >= SPEC["cooldown_frames"])):
                item["alarm"] = True
                last_alarm[arm] = row["j"]
                monitor_alarms[arm] += 1

    probe_arms = {arm: {
        "sse": 1.2, "count": 12, "frame_count": 2,
        "feature_target_count": 12, "mse": 0.1,
    } for arm in ARMS}
    probe = {"frames": probe_n, "warmup_frames": 1, "arms": probe_arms,
             "fixed_hash_before": "hf", "fixed_hash_after": "hf"}
    blocks = []
    checkpoints = [{"monitor_frame": 0, "sealed_probe": copy.deepcopy(probe),
                    "fixed_hash": "hf"}]
    flat_scores = {arm: [item for item in scores[arm]] for arm in ARMS}
    for block_index, frame_end in enumerate((2, 4), 1):
        block_arms = {}
        for arm in ARMS:
            offset = (block_index - 1) * SPEC["block_frames"]
            lo = max(0, offset - SPEC["warmup_frames"])
            hi = frame_end - SPEC["warmup_frames"]
            block_values = flat_scores[arm][lo:hi]
            metric = _metric(block_values)
            count = sum(1 for row in rows[
                fit_n + cal_n + probe_n + (block_index - 1) * 2:
                fit_n + cal_n + probe_n + frame_end]
                if row["arms"][arm]["alarm"])
            metric["alarm_count"] = count
            metric["alarms_per_minute"] = (
                count / (len(block_values) * 0.2 / 60)
                if block_values else None)
            block_arms[arm] = metric
        blocks.append({"block": block_index, "frames": frame_end,
                       "arms": block_arms})
        checkpoints.append({"monitor_frame": frame_end,
                            "sealed_probe": copy.deepcopy(probe),
                            "fixed_hash": "hf"})
    cumulative = {}
    rates = {}
    for arm in ARMS:
        cumulative[arm] = _metric(scores[arm])
        cumulative[arm]["alarm_count"] = monitor_alarms[arm]
        cumulative[arm]["alarms_per_minute"] = (
            monitor_alarms[arm] / (len(scores[arm]) * 0.2 / 60)
            if scores[arm] else None)
        rates[arm] = cumulative[arm]["alarms_per_minute"]
    comparisons = {}
    for arm in ARMS:
        if arm == "fixed_snn":
            continue
        fixed_mse = cumulative["fixed_snn"]["mse"]
        arm_mse = cumulative[arm]["mse"]
        comparisons[arm] = {
            "block_mse_delta": [
                block["arms"][arm]["mse"]
                - block["arms"]["fixed_snn"]["mse"] for block in blocks],
            "cumulative_mse_delta": arm_mse - fixed_mse,
            "relative_improvement": ((fixed_mse - arm_mse) / fixed_mse
                                     if fixed_mse else None),
        }
    for block in blocks:
        fixed_mse = block["arms"]["fixed_snn"]["mse"]
        block["relative_improvement_vs_fixed"] = {
            arm: ((fixed_mse - block["arms"][arm]["mse"]) / fixed_mse
                  if fixed_mse else None)
            for arm in ARMS if arm != "fixed_snn"
        }
    summary = {
        "protocol": dict(SPEC), "phase": "monitor",
        "frame_count": len(rows), "normalizer": norm,
        "reference_hash": "hf", "fixed_hash": "hf",
        "thresholds": thresholds,
        "calibration": {arm: _metric(cal_scores[arm]) for arm in ARMS},
        "sealed_probe": probe, "monitor_frames": 4,
        "monitor_alarms": monitor_alarms,
        "observed_alarms_per_minute": rates,
        "cumulative": cumulative, "blocks": blocks,
        "checkpoints": checkpoints, "adaptive_vs_fixed": comparisons,
    }
    return rows, summary


class AudioSessionAuditTests(unittest.TestCase):
    """Reject inconsistent claims in a synthetic private trace."""

    def setUp(self) -> None:
        self.rows, self.summary = _make_case()

    def audit(self, rows: list[dict[str, Any]],
              summary: dict[str, Any]) -> None:
        audit_data(rows, summary, SPEC)

    def test_accepts_consistent_rows_and_aggregate(self) -> None:
        self.audit(self.rows, self.summary)

    def test_rejects_changed_prior_forecast(self) -> None:
        rows = copy.deepcopy(self.rows)
        prediction = list(rows[4]["arms"]["adaptive_snn"]
                          ["prediction_before"])
        prediction[0] += 0.1
        rows[4]["arms"]["adaptive_snn"]["prediction_before"] = prediction
        with self.assertRaisesRegex(ValueError, "prior forecast"):
            self.audit(rows, self.summary)

    def test_rejects_wrong_paired_sse(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[4]["arms"]["fixed_snn"]["sse"] += 1.0
        with self.assertRaisesRegex(ValueError, "fixed_snn.sse"):
            self.audit(rows, self.summary)

    def test_rejects_fixed_weight_hash_drift(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[-1]["arms"]["fixed_snn"]["hash_after"] = "changed"
        with self.assertRaisesRegex(
                ValueError, "fixed_snn parameters changed"):
            self.audit(rows, self.summary)

    def test_rejects_normalizer_that_uses_future_rows(self) -> None:
        summary = copy.deepcopy(self.summary)
        summary["normalizer"]["centers_db"][0] += 1.0
        with self.assertRaisesRegex(ValueError, "normalizer"):
            self.audit(self.rows, summary)

    def test_rejects_threshold_mismatch(self) -> None:
        summary = copy.deepcopy(self.summary)
        summary["thresholds"]["ema"] += 0.1
        with self.assertRaisesRegex(ValueError, "thresholds.ema"):
            self.audit(self.rows, summary)

    def test_rejects_incomplete_checkpoint_curve(self) -> None:
        summary = copy.deepcopy(self.summary)
        summary["checkpoints"].pop()
        with self.assertRaisesRegex(ValueError, "checkpoint count"):
            self.audit(self.rows, summary)

    def test_allows_adaptive_probe_error_to_change_at_checkpoint(self) -> None:
        summary = copy.deepcopy(self.summary)
        item = summary["checkpoints"][1]["sealed_probe"]["arms"][
            "adaptive_snn"]
        item["sse"] = 1.8
        item["mse"] = 0.15
        self.audit(self.rows, summary)

    def test_rejects_fixed_probe_error_change_at_checkpoint(self) -> None:
        summary = copy.deepcopy(self.summary)
        item = summary["checkpoints"][1]["sealed_probe"]["arms"][
            "fixed_snn"]
        item["sse"] = 1.8
        item["mse"] = 0.15
        with self.assertRaisesRegex(ValueError, "fixed_snn"):
            self.audit(self.rows, summary)

    def test_audits_a_full_fixture_session_trace(self) -> None:
        session = AudioSession(load_protocol())
        rows = []
        total = (150 + 150 + 100 + 300)
        for index in range(total):
            bands = [
                -56.0 + band * 2.5
                + ((index * (band + 2)) % 17) * 0.21
                for band in range(6)
            ]
            rows.append(session.observe(
                {"band_dbfs": bands}, index * 0.2))
        result = audit_data(rows, session.summary())
        self.assertTrue(result["valid"])
        self.assertEqual(result["frames"], total)
        self.assertEqual(result["monitor_frames"], 300)


if __name__ == "__main__":
    unittest.main()
