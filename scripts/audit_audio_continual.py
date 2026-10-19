"""Independently verify private traces from the continual audio experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "fixtures" / "audio_continual_v1.json"
PROTOCOL_SHA256 = (
    "ff4a161882904aca6ba31a8498a859a1347addaa289e989ec0ea1e231a70fb58")
ARMS = (
    "adaptive_snn", "readout_snn", "memory_snn", "recent_memory_snn",
    "fixed_snn", "adaptive_ann", "memory_ann", "ema", "persistence",
)
NEURAL_ARMS = ARMS[:7]
BASELINE_ARMS = ("ema", "persistence")
TOLERANCE = 1e-10
COUNTERS = (
    "sessions", "predictions", "scored_targets", "updates",
    "neuron_ticks", "spikes", "readout_synapse_visits",
    "input_synapse_visits", "changed_readout_parameters",
    "changed_input_parameters",
)


def _finite(value: Any, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _close(actual: Any, expected: float, label: str) -> None:
    value = _finite(actual, label)
    if not math.isclose(value, expected, rel_tol=TOLERANCE,
                        abs_tol=TOLERANCE):
        raise ValueError(f"{label} mismatch")


def _same(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label} keys mismatch")
        for key, value in expected.items():
            _same(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{label} length mismatch")
        for index, value in enumerate(expected):
            _same(actual[index], value, f"{label}[{index}]")
    elif isinstance(expected, bool):
        if type(actual) is not bool or actual is not expected:
            raise ValueError(f"{label} mismatch")
    elif type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise ValueError(f"{label} mismatch")
    elif isinstance(expected, float):
        _close(actual, expected, label)
    elif actual != expected:
        raise ValueError(f"{label} mismatch")


def _vector(value: Any, label: str, *, unit_interval: bool = False
            ) -> list[float]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError(f"{label} must contain six values")
    result = [_finite(item, f"{label}[{index}]")
              for index, item in enumerate(value)]
    if unit_interval and any(item < 0.0 or item > 1.0 for item in result):
        raise ValueError(f"{label} values must be in [0, 1]")
    return result


def load_protocol() -> dict[str, Any]:
    """Load the frozen fixture and verify its exact checked-in digest."""
    payload = PROTOCOL_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("continual protocol hash mismatch")
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("experiment") != (
            "audio_continual_v1"):
        raise ValueError("unsupported continual experiment protocol")
    if value.get("arms") != list(ARMS) or value.get("seeds") != [0, 1, 2]:
        raise ValueError("continual arm or seed contract changed")
    if value.get("monitor_frames") != 9000:
        raise ValueError("continual monitor length changed")
    return value


def _normalizer(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    columns = [[row["features"]["band_dbfs"][band] for row in rows]
               for band in range(6)]
    centers = [statistics.median(column) for column in columns]
    scales = [max(3.0, 1.4826 * statistics.median(
        [abs(value - center) for value in column]))
        for column, center in zip(columns, centers)]
    return {"centers_db": centers, "scales_db": scales}


def _normalize(values: list[float], norm: dict[str, list[float]]) \
        -> list[float]:
    return [0.5 + 0.5 * (value - center) / (scale + abs(value - center))
            for value, center, scale in zip(
                values, norm["centers_db"], norm["scales_db"])]


def _metric(scores: list[float]) -> dict[str, Any]:
    sse = sum(score * score for score in scores) * 6.0
    targets = len(scores) * 6
    return {"count": len(scores), "frame_count": len(scores),
            "feature_target_count": targets, "sse": sse,
            "mse": sse / targets if targets else None}


def _check_metrics(actual: Any, scores: list[float], label: str) -> None:
    if not isinstance(actual, dict):
        raise ValueError(f"{label} must be an object")
    for key, value in _metric(scores).items():
        _same(actual.get(key), value, f"{label}.{key}")


def _ema_next(previous: list[float] | None, current: list[float],
              alpha: float, arm: str) -> list[float]:
    if previous is None or arm == "persistence":
        return current[:]
    if arm == "ema":
        return [alpha * actual + (1.0 - alpha) * prior
                for actual, prior in zip(current, previous)]
    raise ValueError("unknown baseline arm")


def _probe_expected(rows: list[list[float]], warmup: int,
                    alpha: float, arm: str) -> dict[str, Any]:
    previous: list[float] | None = None
    scores: list[float] = []
    for position, actual in enumerate(rows):
        if previous is not None and position >= warmup:
            scores.append(math.sqrt(sum((guess - target) ** 2
                                        for guess, target in
                                        zip(previous, actual)) / 6.0))
        previous = _ema_next(previous, actual, alpha, arm)
    frames = len(scores)
    targets = frames * 6
    sse = sum(score * score for score in scores) * 6.0
    return {"count": targets, "frame_count": frames,
            "feature_target_count": targets, "sse": sse,
            "mse": sse / targets if targets else None}


def _check_summary_metrics(actual: Any, scores: list[float],
                           label: str) -> None:
    _check_metrics(actual, scores, label)


def audit_data(rows: list[dict[str, Any]], summary: dict[str, Any],
               protocol: dict[str, Any] | None = None,
               expected_seed: int | None = None) -> dict[str, Any]:
    """Validate trace structure and independently recompute its arithmetic."""
    spec = load_protocol() if protocol is None else protocol
    if not isinstance(summary, dict) or not isinstance(rows, list):
        raise ValueError("trace and summary must be JSON objects")
    if spec.get("arms") != list(ARMS):
        raise ValueError("protocol arm contract mismatch")
    session_protocol = summary.get("protocol")
    if not isinstance(session_protocol, dict):
        raise ValueError("summary protocol is missing")
    seed = session_protocol.get("seed")
    allowed_seeds = spec.get("seeds", [spec.get("seed", 0)])
    if type(seed) is not int or seed not in allowed_seeds:
        raise ValueError("summary seed is outside the frozen set")
    if expected_seed is not None and seed != expected_seed:
        raise ValueError("summary seed does not match requested seed")
    for key in ("frame_seconds", "fit_frames", "calibration_frames",
                "probe_frames", "block_frames", "warmup_frames",
                "cooldown_frames", "ema_alpha"):
        _same(session_protocol.get(key), spec[key], f"protocol.{key}")
    _same(session_protocol.get("arms"), spec["arms"], "protocol.arms")
    if type(session_protocol.get("seed")) is not int:
        raise ValueError("protocol.seed must be an integer")

    fit_n = int(spec["fit_frames"])
    cal_n = int(spec["calibration_frames"])
    probe_n = int(spec["probe_frames"])
    warmup = int(spec["warmup_frames"])
    block_n = int(spec["block_frames"])
    frame_seconds = float(spec["frame_seconds"])
    monitor_frames = summary.get("monitor_frames")
    if type(monitor_frames) is not int or monitor_frames < 0:
        raise ValueError("summary monitor_frames must be non-negative int")
    if monitor_frames > int(spec.get("monitor_frames", monitor_frames)):
        raise ValueError("monitor frame count exceeds frozen protocol")
    frame_count = summary.get("frame_count")
    expected_count = fit_n + cal_n + probe_n + monitor_frames
    if type(frame_count) is not int or frame_count != expected_count:
        raise ValueError("summary frame_count mismatch")
    if len(rows) != expected_count:
        raise ValueError("trace row count does not match summary prefix")

    fit_rows = rows[:fit_n]
    norm = _normalizer(fit_rows)
    _same(summary.get("normalizer"), norm, "normalizer")
    offsets = {"fit": 0, "calibration": fit_n,
               "probe": fit_n + cal_n,
               "monitor": fit_n + cal_n + probe_n}
    scores = {arm: [] for arm in ARMS}
    calibration_scores = {arm: [] for arm in ARMS}
    alarms = {arm: 0 for arm in ARMS}
    block_alarms: dict[str, list[int]] = {arm: [] for arm in ARMS}
    prior: dict[str, list[float] | None] = {}
    hashes: dict[str, str | None] = {arm: None for arm in ARMS}
    fixed_hash: str | None = None
    last_received: float | None = None
    thresholds: dict[str, float] = {}
    last_alarm: dict[str, int | None] = {arm: None for arm in ARMS}
    last_phase: str | None = None
    probe_vectors: list[list[float]] = []
    inversion_sse = {arm: [0.0, 0.0, 0.0] for arm in ARMS}
    scored_count = 0
    fitted_hashes: dict[str, str] = {}
    monitor_initial_hashes: dict[str, str] = {}

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("each trace row must be an object")
        phase = ("fit" if index < fit_n else
                 "calibration" if index < fit_n + cal_n else
                 "probe" if index < fit_n + cal_n + probe_n else "monitor")
        position = index - offsets[phase]
        if phase != last_phase:
            prior.clear()
            for arm in ARMS:
                hashes[arm] = None
            if phase == "monitor":
                thresholds = {arm: max(1e-9, max(calibration_scores[arm]))
                              for arm in ARMS}
            last_phase = phase
        if (type(row.get("i")) is not int or row.get("i") != index
                or row.get("phase") != phase
                or type(row.get("j")) is not int
                or row.get("j") != position):
            raise ValueError("frame index or phase sequence mismatch")
        _close(row.get("timestamp"), (index + 1) * frame_seconds,
               "sample-clock timestamp")
        received = _finite(row.get("received_seconds"), "received_seconds")
        if last_received is not None and received < last_received:
            raise ValueError("received_seconds moved backwards")
        last_received = received
        features = row.get("features")
        if not isinstance(features, dict):
            raise ValueError("features must be an object")
        bands = _vector(features.get("band_dbfs"), "features.band_dbfs")
        arm_rows = row.get("arms")
        if not isinstance(arm_rows, dict):
            raise ValueError("arms must be an object")
        if phase == "fit":
            if arm_rows or row.get("normalized") is not None:
                raise ValueError("fit rows cannot expose model outputs")
            continue
        normalized = _vector(row.get("normalized"), "normalized",
                             unit_interval=True)
        _same(normalized, _normalize(bands, norm), "normalized")
        if phase == "probe":
            if arm_rows:
                raise ValueError("probe rows cannot expose model outputs")
            probe_vectors.append(normalized)
            continue
        if set(arm_rows) != set(ARMS):
            raise ValueError("arm keys mismatch")
        if position == 0:
            prior.clear()
            for arm in ARMS:
                hashes[arm] = None
                last_alarm[arm] = None
        for arm in ARMS:
            item = arm_rows[arm]
            if not isinstance(item, dict):
                raise ValueError(f"{arm} row must be an object")
            before_value = item.get("prediction_before")
            after = _vector(item.get("prediction_next"),
                            f"{arm}.prediction_next", unit_interval=True)
            previous = prior.get(arm)
            if previous is None:
                if before_value is not None:
                    raise ValueError(
                        f"{arm} first prior forecast must be null")
                if (item.get("score") is not None
                        or item.get("sse") is not None):
                    raise ValueError(f"{arm} first score must be null")
                score = None
                sse = None
                is_scored = False
            else:
                before = _vector(before_value, f"{arm}.prediction_before",
                                 unit_interval=True)
                _same(before, previous, f"{arm}.prior forecast")
                errors = [guess - target for guess, target in
                          zip(before, normalized)]
                sse = sum(error * error for error in errors)
                score = math.sqrt(sse / 6.0)
                _close(item.get("score"), score, f"{arm}.score")
                _close(item.get("sse"), sse, f"{arm}.sse")
                is_scored = position >= warmup
            if item.get("scored") is not is_scored:
                raise ValueError(f"{arm}.scored mismatch")
            if arm in BASELINE_ARMS:
                expected_after = _ema_next(previous, normalized,
                                           float(spec["ema_alpha"]), arm)
                _same(after, expected_after, f"{arm}.prediction_next")
            if phase == "calibration" and is_scored and score is not None:
                calibration_scores[arm].append(score)
            if phase == "monitor" and is_scored and score is not None:
                scores[arm].append(score)
                scored_count += int(arm == ARMS[0])
                if sse is None:
                    raise ValueError("scored row has no SSE")
                error = errors
                predictions = before
                target = normalized
                complement_error = sum((1.0 - p - y) ** 2
                                       for p, y in zip(predictions, target))
                both_error = sum(((1.0 - p) - (1.0 - y)) ** 2
                                 for p, y in zip(predictions, target))
                inversion_sse[arm][0] += sse
                inversion_sse[arm][1] += complement_error
                inversion_sse[arm][2] += both_error
            if phase == "monitor":
                alarm = item.get("alarm")
                if type(alarm) is not bool:
                    raise ValueError(f"{arm}.alarm must be boolean")
                expected_alarm = False
                if is_scored and score is not None:
                    last = last_alarm[arm]
                    elapsed = (int(spec["cooldown_frames"]) if last is None
                               else position - last)
                    if (score > thresholds[arm]
                            and elapsed >= int(spec["cooldown_frames"])):
                        expected_alarm = True
                        last_alarm[arm] = position
                if alarm is not expected_alarm:
                    raise ValueError(f"{arm}.alarm mismatch")
                alarms[arm] += int(alarm)
                block_index = position // block_n
                while len(block_alarms[arm]) <= block_index:
                    block_alarms[arm].append(0)
                block_alarms[arm][block_index] += int(alarm)
            elif item.get("alarm") is not False:
                raise ValueError(f"{arm} non-monitor alarm must be false")
            _validate_hash(item, arm, hashes)
            if phase == "calibration" and position == 0 \
                    and arm in NEURAL_ARMS:
                fitted_hashes[arm] = item["hash_before"]
            if phase == "monitor" and position == 0 \
                    and arm in NEURAL_ARMS:
                monitor_initial_hashes[arm] = item["hash_before"]
            if arm == "fixed_snn":
                if fixed_hash is None:
                    fixed_hash = item["hash_before"]
                if (item["hash_before"] != fixed_hash
                        or item["hash_after"] != fixed_hash):
                    raise ValueError("fixed SNN hash changed")
            prior[arm] = after

    if monitor_frames == 0:
        thresholds = {arm: max(1e-9, max(calibration_scores[arm]))
                      for arm in ARMS}

    if scored_count != max(0, monitor_frames - warmup):
        raise ValueError("scored monitor frame count mismatch")
    if summary.get("experiment") != "audio_continual_v1":
        raise ValueError("summary experiment name mismatch")
    _same(summary.get("thresholds"), thresholds, "thresholds")
    if summary.get("fixed_hash") != fixed_hash:
        raise ValueError("summary fixed hash mismatch")
    if summary.get("reference_hash") != fixed_hash:
        raise ValueError("summary reference hash mismatch")
    model_metrics = summary.get("model_metrics")
    declared_fit_hashes = (model_metrics.get("initial_fit_weight_hashes")
                           if isinstance(model_metrics, dict) else None)
    _check_initial_hashes(fitted_hashes, monitor_initial_hashes,
                          declared_fit_hashes, monitor_frames)
    _same(summary.get("monitor_alarms"), alarms, "monitor_alarms")
    _check_calibration(summary, calibration_scores)
    _check_cumulative(summary, scores, alarms, frame_seconds)
    _check_blocks(summary, scores, block_alarms, block_n, warmup)
    _check_probe(summary, spec, probe_vectors, fixed_hash)
    _check_model_metrics(summary, fixed_hash)
    _check_paired(summary)

    targets = max(0, monitor_frames - warmup) * 6
    inversion = {}
    for arm in ARMS:
        original, prediction_flip, both_flip = inversion_sse[arm]
        inversion[arm] = {
            "scored_frames": max(0, monitor_frames - warmup),
            "feature_target_count": targets,
            "original_mse": original / targets if targets else None,
            "prediction_only_complement_mse": (
                prediction_flip / targets if targets else None),
            "prediction_and_target_complement_mse": (
                both_flip / targets if targets else None),
        }
        if not math.isclose(original, both_flip, rel_tol=TOLERANCE,
                            abs_tol=TOLERANCE):
            raise ValueError("joint inversion failed its MSE invariance check")
    return {
        "valid": True, "experiment": "audio_continual_v1",
        "seed": seed, "frames": len(rows),
        "monitor_frames": monitor_frames, "fixed_hash": fixed_hash,
        "thresholds": thresholds, "initial_fit_hashes": fitted_hashes,
        "inversion_diagnostic": inversion,
    }


def _validate_hash(item: dict[str, Any], arm: str,
                   previous: dict[str, str | None]) -> None:
    before = item.get("hash_before")
    after = item.get("hash_after")
    if (not isinstance(before, str) or not before
            or not isinstance(after, str) or not after):
        raise ValueError(f"{arm} hashes must be nonempty strings")
    if previous[arm] is not None and before != previous[arm]:
        raise ValueError(f"{arm} hash continuity mismatch")
    if arm == "fixed_snn" and before != after:
        raise ValueError("fixed SNN parameters changed")
    if arm in BASELINE_ARMS and before != after:
        raise ValueError(f"{arm} unexpectedly has a parameter hash change")
    previous[arm] = after


def _check_calibration(summary: dict[str, Any],
                       scores: dict[str, list[float]]) -> None:
    thresholds = summary.get("thresholds")
    calibration = summary.get("calibration")
    if not isinstance(thresholds, dict) or set(thresholds) != set(ARMS):
        raise ValueError("calibration threshold arm keys mismatch")
    if not isinstance(calibration, dict) or set(calibration) != set(ARMS):
        raise ValueError("calibration arm keys mismatch")
    for arm in ARMS:
        if not scores[arm]:
            raise ValueError(f"{arm} has no scored calibration forecasts")
        _same(thresholds[arm], max(1e-9, max(scores[arm])),
              f"thresholds.{arm}")
        _check_summary_metrics(calibration[arm], scores[arm],
                               f"calibration.{arm}")


def _check_cumulative(summary: dict[str, Any],
                      scores: dict[str, list[float]],
                      alarms: dict[str, int], frame_seconds: float) -> None:
    cumulative = summary.get("cumulative")
    if not isinstance(cumulative, dict) or set(cumulative) != set(ARMS):
        raise ValueError("cumulative arm keys mismatch")
    rates = summary.get("observed_alarms_per_minute")
    if not isinstance(rates, dict) or set(rates) != set(ARMS):
        raise ValueError("observed alarm rate keys mismatch")
    for arm in ARMS:
        _check_metrics(cumulative[arm], scores[arm], f"cumulative.{arm}")
        _same(cumulative[arm].get("alarm_count"), alarms[arm],
              f"cumulative.{arm}.alarm_count")
        frames = len(scores[arm])
        expected = (alarms[arm] / (frames * frame_seconds / 60.0)
                    if frames else None)
        _same(cumulative[arm].get("alarms_per_minute"), expected,
              f"cumulative.{arm}.alarms_per_minute")
        _same(rates[arm], expected, f"observed_alarms_per_minute.{arm}")


def _check_blocks(summary: dict[str, Any], scores: dict[str, list[float]],
                  alarm_counts: dict[str, list[int]], block_n: int,
                  warmup: int) -> None:
    monitor_frames = int(summary["monitor_frames"])
    count = monitor_frames // block_n
    blocks = summary.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != count:
        raise ValueError("complete block count mismatch")
    for index, block in enumerate(blocks):
        if (type(block.get("block")) is not int
                or block["block"] != index + 1
                or type(block.get("frames")) is not int
                or block["frames"] != (index + 1) * block_n):
            raise ValueError("block boundary mismatch")
        arm_data = block.get("arms")
        if not isinstance(arm_data, dict) or set(arm_data) != set(ARMS):
            raise ValueError("block arm keys mismatch")
        for arm in ARMS:
            start = 0 if index == 0 else index * block_n - warmup
            end = (index + 1) * block_n - warmup
            expected_scores = scores[arm][start:end]
            item = arm_data[arm]
            _check_metrics(item, expected_scores,
                           f"blocks[{index}].{arm}")
            alarm_count = (alarm_counts[arm][index]
                           if index < len(alarm_counts[arm]) else 0)
            _same(item.get("alarm_count"), alarm_count,
                  f"blocks[{index}].{arm}.alarm_count")
            minutes = len(expected_scores) * 0.2 / 60.0
            expected_rate = alarm_count / minutes if minutes else None
            _same(item.get("alarms_per_minute"), expected_rate,
                  f"blocks[{index}].{arm}.alarms_per_minute")


def _check_probe(summary: dict[str, Any], spec: dict[str, Any],
                 rows: list[list[float]], fixed_hash: str | None) -> None:
    probe = summary.get("sealed_probe")
    expected_n = int(spec["probe_frames"])
    warmup = int(spec["warmup_frames"])
    if len(rows) != expected_n:
        raise ValueError("sealed probe row count mismatch")
    if (not isinstance(probe, dict) or probe.get("frames") != expected_n
            or probe.get("warmup_frames") != warmup):
        raise ValueError("sealed probe metadata mismatch")
    checkpoints = summary.get("checkpoints")
    monitor_frames = int(summary["monitor_frames"])
    expected_count = monitor_frames // int(spec["block_frames"]) + 1
    if not isinstance(checkpoints, list) or len(checkpoints) != expected_count:
        raise ValueError("checkpoint count mismatch")
    baseline_probe = probe.get("arms")
    if (not isinstance(baseline_probe, dict)
            or set(baseline_probe) != set(ARMS)):
        raise ValueError("sealed probe arm keys mismatch")
    for arm in BASELINE_ARMS:
        expected = _probe_expected(rows, warmup, float(spec["ema_alpha"]), arm)
        _same(_metric_values(baseline_probe[arm]), expected,
              f"sealed_probe.{arm}")
    if probe.get("fixed_hash_before") != fixed_hash \
            or probe.get("fixed_hash_after") != fixed_hash:
        raise ValueError("sealed probe fixed hash mismatch")
    for arm in ARMS:
        _check_probe_record(baseline_probe[arm], expected_n - warmup,
                            f"sealed_probe.{arm}")
    for index, checkpoint in enumerate(checkpoints):
        frame = index * int(spec["block_frames"])
        if (not isinstance(checkpoint, dict)
                or type(checkpoint.get("monitor_frame")) is not int
                or checkpoint["monitor_frame"] != frame):
            raise ValueError("checkpoint boundary mismatch")
        checkpoint_probe = checkpoint.get("sealed_probe")
        if not isinstance(checkpoint_probe, dict):
            raise ValueError("checkpoint missing probe")
        if checkpoint.get("fixed_hash") != fixed_hash:
            raise ValueError("checkpoint fixed hash changed")
        if (checkpoint_probe.get("fixed_hash_before") != fixed_hash
                or checkpoint_probe.get("fixed_hash_after") != fixed_hash):
            raise ValueError("checkpoint changed fixed hash")
        if index == 0:
            _same(checkpoint_probe, probe, "checkpoints[0].sealed_probe")
        cp_arms = checkpoint_probe.get("arms")
        if not isinstance(cp_arms, dict) or set(cp_arms) != set(ARMS):
            raise ValueError("checkpoint probe arm keys mismatch")
        for arm in ARMS:
            _check_probe_record(cp_arms[arm], expected_n - warmup,
                                f"checkpoint.{arm}")
        for arm in ("fixed_snn", *BASELINE_ARMS):
            _same(cp_arms[arm], baseline_probe[arm],
                  f"checkpoint.{arm} invariant")


def _metric_values(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("metric record must be an object")
    return {key: value.get(key) for key in (
        "count", "frame_count", "feature_target_count", "sse", "mse")}


def _check_probe_record(value: Any, expected_frames: int,
                        label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    expected_targets = expected_frames * 6
    for key, expected in (("count", expected_targets),
                          ("feature_target_count", expected_targets),
                          ("frame_count", expected_frames)):
        if type(value.get(key)) is not int or value[key] != expected:
            raise ValueError(f"{label}.{key} mismatch")
    sse = _finite(value.get("sse"), f"{label}.sse")
    if sse < 0.0:
        raise ValueError(f"{label}.sse must be non-negative")
    _close(value.get("mse"), sse / expected_targets, f"{label}.mse")


def _check_initial_hashes(fitted: dict[str, str],
                          monitored: dict[str, str],
                          declared: Any,
                          monitor_frames: int) -> None:
    if not isinstance(declared, dict) or set(declared) != set(NEURAL_ARMS):
        raise ValueError("declared initial fit hashes are incomplete")
    if set(fitted) != set(NEURAL_ARMS):
        raise ValueError("initial neural parameter hashes are incomplete")
    for arm in NEURAL_ARMS:
        if not isinstance(declared[arm], str) or not declared[arm]:
            raise ValueError(f"{arm} declared fit hash is invalid")
        if fitted[arm] != declared[arm]:
            raise ValueError(f"{arm} declared fit hash mismatch")
    if monitor_frames == 0:
        monitored = declared
    if set(monitored) != set(NEURAL_ARMS):
        raise ValueError("initial monitor parameter hashes are incomplete")
    for arm in NEURAL_ARMS:
        if declared[arm] != monitored[arm]:
            raise ValueError(f"{arm} changed between fit and monitor start")
    snn_hashes = {fitted[arm] for arm in
                  ("adaptive_snn", "readout_snn", "memory_snn",
                   "recent_memory_snn", "fixed_snn")}
    ann_hashes = {fitted[arm] for arm in ("adaptive_ann", "memory_ann")}
    if len(snn_hashes) != 1 or len(ann_hashes) != 1:
        raise ValueError("matched initial model parameters differ")


def _check_model_metrics(summary: dict[str, Any],
                         fixed_hash: str | None) -> None:
    metrics = summary.get("model_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("model_metrics must be an object")
    if summary.get("experiment") != "audio_continual_v1":
        raise ValueError("continual experiment marker missing")
    for name in NEURAL_ARMS:
        item = metrics.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"model_metrics.{name} missing")
        bands = item.get("bands")
        if not isinstance(bands, list) or len(bands) != 6:
            raise ValueError(f"model_metrics.{name}.bands mismatch")
        spiking = name not in ("adaptive_ann", "memory_ann")
        learning = name != "fixed_snn"
        encoder_learning = name not in ("fixed_snn", "readout_snn")
        _same(item.get("spiking"), spiking, f"model_metrics.{name}.spiking")
        _same(item.get("learning"), learning,
              f"model_metrics.{name}.learning")
        _same(item.get("encoder_learning"), encoder_learning,
              f"model_metrics.{name}.encoder_learning")
        for counter in COUNTERS:
            total = item.get(counter)
            if type(total) is not int or total < 0:
                raise ValueError(f"model_metrics.{name}.{counter} invalid")
            band_total = 0
            for band in bands:
                value = band.get(counter)
                if type(value) is not int or value < 0:
                    raise ValueError(
                        f"model_metrics.{name}.bands.{counter} invalid")
                band_total += value
            if band_total != total:
                raise ValueError(
                    f"model_metrics.{name}.{counter} total mismatch")
        for key in ("weight_hashes", "encoder_hashes"):
            value = item.get(key)
            if (not isinstance(value, list) or len(value) != 6
                    or any(not isinstance(digest, str) or not digest
                           for digest in value)):
                raise ValueError(f"model_metrics.{name}.{key} invalid")
        for index, band in enumerate(bands):
            if band.get("weight_hash") != item["weight_hashes"][index]:
                raise ValueError(
                    f"model_metrics.{name} weight hashes mismatch")
            if band.get("encoder_hash") != item["encoder_hashes"][index]:
                raise ValueError(
                    f"model_metrics.{name} encoder hashes mismatch")
        if name == "fixed_snn":
            if item["updates"] or item["changed_readout_parameters"] \
                    or item["changed_input_parameters"]:
                raise ValueError("fixed model reports parameter updates")
        if name in ("memory_snn", "recent_memory_snn", "memory_ann"):
            if item.get("memory_capacity_frames") != 150:
                raise ValueError(f"{name} memory capacity mismatch")
            expected_mode = ("recent" if name == "recent_memory_snn"
                             else "fit")
            if item.get("memory_mode") != expected_mode:
                raise ValueError(f"{name} memory mode mismatch")
            passes = item.get("replay_passes")
            targets = item.get("replay_targets")
            if (type(passes) is not int or passes < 0
                    or type(targets) is not int or targets < 0):
                raise ValueError(f"{name} replay counters invalid")
            if item.get("online_training_targets") != max(
                    int(band["online_training_targets"]) for band in bands):
                raise ValueError(f"{name} online target count mismatch")
            if item.get("replay_passes") != sum(
                    int(band["replay_passes"]) for band in bands):
                raise ValueError(f"{name} replay pass total mismatch")
            if item.get("replay_targets") != sum(
                    int(band["replay_targets"]) for band in bands):
                raise ValueError(f"{name} replay target total mismatch")
    if metrics.get("memory_capacity_frames") != 150:
        raise ValueError("summary memory capacity mismatch")
    if metrics.get("memory_interval_targets") != 300:
        raise ValueError("summary memory interval mismatch")
    _same(metrics.get("memory_samples_per_arm"), {
        "memory_snn": 150, "recent_memory_snn": 150,
        "memory_ann": 150,
    }, "model_metrics.memory_samples_per_arm")
    if metrics.get("memory_values_per_bank") != 900:
        raise ValueError("summary memory values per bank mismatch")
    if metrics.get("trainable_parameters_per_model") != 222:
        raise ValueError("summary parameter count mismatch")
    _finite(metrics.get("processing_wall_seconds"),
            "model_metrics.processing_wall_seconds")
    fixed_hashes = metrics["fixed_snn"]["weight_hashes"]
    encoded_hashes = json.dumps(fixed_hashes, separators=(",", ":"))
    computed_fixed = hashlib.sha256(encoded_hashes.encode("utf-8")).hexdigest()
    if computed_fixed != fixed_hash:
        raise ValueError("fixed hash does not match fixed parameter metrics")


def _check_paired(summary: dict[str, Any]) -> None:
    paired = summary.get("adaptive_vs_fixed")
    expected_arms = {arm for arm in ARMS if arm != "fixed_snn"}
    if not isinstance(paired, dict) or set(paired) != expected_arms:
        raise ValueError("paired comparison arms mismatch")
    blocks = summary["blocks"]
    cumulative = summary["cumulative"]
    for arm in expected_arms:
        deltas = [block["arms"][arm]["mse"]
                  - block["arms"]["fixed_snn"]["mse"]
                  for block in blocks]
        _same(paired[arm].get("block_mse_delta"), deltas,
              f"adaptive_vs_fixed.{arm}.block_mse_delta")
        fixed = cumulative["fixed_snn"]["mse"]
        current = cumulative[arm]["mse"]
        delta = current - fixed if current is not None and fixed is not None \
            else None
        relative = ((fixed - current) / fixed
                    if fixed not in (None, 0)
                    and current is not None else None)
        _same(paired[arm].get("cumulative_mse_delta"), delta,
              f"adaptive_vs_fixed.{arm}.cumulative_mse_delta")
        _same(paired[arm].get("relative_improvement"), relative,
              f"adaptive_vs_fixed.{arm}.relative_improvement")


def audit_session(trace_path: str | Path,
                  summary_path: str | Path,
                  expected_seed: int | None = None,
                  protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    """Audit a trace prefix and its matching direct or wrapped summary."""
    payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("summary must be a JSON object")
    summary = payload.get("summary", payload)
    if not isinstance(summary, dict):
        raise ValueError("summary body must be an object")
    frame_count = summary.get("frame_count")
    if type(frame_count) is not int or frame_count < 0:
        raise ValueError("summary frame_count must be a non-negative int")
    rows: list[dict[str, Any]] = []
    with Path(trace_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if len(rows) >= frame_count:
                break
            if not line.endswith("\n"):
                raise ValueError(f"incomplete trace line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"trace line {line_number} is not an object")
            rows.append(value)
    if len(rows) != frame_count:
        raise ValueError("trace is shorter than summary prefix")
    result = audit_data(rows, summary, protocol=protocol,
                        expected_seed=expected_seed)
    result["summary_sha256"] = hashlib.sha256(
        Path(summary_path).read_bytes()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("summary")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    result = audit_session(args.trace, args.summary, args.seed)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
