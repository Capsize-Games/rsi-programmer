"""Independently verify a private live-audio session trace and summary."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "fixtures" / "audio_fan_live_v1.json"
PROTOCOL_SHA256 = (
    "ef7019083c0fcb24f3477a8384a5b3deccd260fd1ad9f7ef1d133c433b065561")
ARMS = ("adaptive_snn", "fixed_snn", "adaptive_ann", "ema", "persistence")
TOLERANCE = 1e-10


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


def _vector(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError(f"{label} must contain six values")
    return [_finite(item, f"{label}[{index}]")
            for index, item in enumerate(value)]


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
    elif isinstance(expected, (int, float)):
        _close(actual, float(expected), label)
    elif actual != expected:
        raise ValueError(f"{label} mismatch")


def load_protocol() -> dict[str, Any]:
    """Load the checked-in protocol and return its JSON object."""
    payload = PROTOCOL_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("audio protocol hash mismatch")
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("unsupported audio protocol")
    if (value.get("sample_rate") != 16000
            or value.get("band_edges_hz") != [80, 160, 320, 640,
                                              1280, 2560, 6000]
            or value.get("arms") != list(ARMS)
            or value.get("ema_alpha") != 1.0 - math.exp(-0.2 / 5.0)):
        raise ValueError("audio feature or arm contract changed")
    return value


def _normalizer(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    columns = [[row["features"]["band_dbfs"][band] for row in rows]
               for band in range(6)]
    centers = [statistics.median(column) for column in columns]
    scales = [max(3.0, 1.4826 * statistics.median(
        [abs(value - center) for value in column]))
        for column, center in zip(columns, centers)]
    return {"centers_db": centers, "scales_db": scales}


def _normalize(values: list[float], normalizer: dict[str, list[float]]) \
        -> list[float]:
    return [0.5 + 0.5 * (value - center) / (scale + abs(value - center))
            for value, center, scale in zip(
                values, normalizer["centers_db"], normalizer["scales_db"])]


def _summary(scores: list[float]) -> dict[str, Any]:
    total = sum(score * score for score in scores) * 6.0
    targets = len(scores) * 6
    return {"count": len(scores), "frame_count": len(scores),
            "feature_target_count": targets, "sse": total,
            "mse": total / targets if targets else None}


def _check_summary(actual: Any, scores: list[float], label: str) -> None:
    _same(actual, _summary(scores), label)


def _check_metrics(actual: Any, scores: list[float], label: str) -> None:
    if not isinstance(actual, dict):
        raise ValueError(f"{label} must be an object")
    expected = _summary(scores)
    for key, value in expected.items():
        _same(actual.get(key), value, f"{label}.{key}")


def audit_data(rows: list[dict[str, Any]], summary: dict[str, Any],
               protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate rows and independently recompute all trace-derived metrics."""
    spec = load_protocol() if protocol is None else protocol
    if not isinstance(summary, dict) or not isinstance(rows, list):
        raise ValueError("trace and summary must be JSON objects")
    if (spec.get("band_edges_hz")
            != [80, 160, 320, 640, 1280, 2560, 6000]
            or spec.get("arms") != list(ARMS)):
        raise ValueError("protocol band or arm contract mismatch")
    session_protocol = summary.get("protocol")
    if not isinstance(session_protocol, dict):
        raise ValueError("summary protocol is missing")
    protocol_keys = ("frame_seconds", "fit_frames", "calibration_frames",
                     "probe_frames", "warmup_frames", "cooldown_frames",
                     "block_frames", "seed", "ema_alpha")
    for key in protocol_keys:
        if key in session_protocol:
            expected_value = spec.get(key, 0.2 if key == "ema_alpha" else None)
            _same(session_protocol[key], expected_value,
                  f"protocol.{key}")
        elif key != "ema_alpha":
            raise ValueError(f"protocol.{key} is missing")
    fit_n = int(spec["fit_frames"])
    cal_n = int(spec["calibration_frames"])
    probe_n = int(spec["probe_frames"])
    warmup = int(spec["warmup_frames"])
    block_n = int(spec["block_frames"])
    monitor_frames = summary.get("monitor_frames")
    if type(monitor_frames) is not int or monitor_frames < 0:
        raise ValueError("summary monitor_frames must be non-negative int")
    frame_count = summary.get("frame_count")
    if type(frame_count) is not int or frame_count < 0:
        raise ValueError("summary frame_count must be non-negative int")
    expected_count = fit_n + cal_n + probe_n + monitor_frames
    if len(rows) != expected_count:
        raise ValueError("trace row count does not match summary prefix")
    if frame_count != len(rows):
        raise ValueError("summary frame_count mismatch")

    fit_rows = rows[:fit_n]
    expected_normalizer = _normalizer(fit_rows)
    _same(summary.get("normalizer"), expected_normalizer, "normalizer")
    phase_offsets = {"fit": 0, "calibration": fit_n,
                     "probe": fit_n + cal_n,
                     "monitor": fit_n + cal_n + probe_n}
    monitor_scores: dict[str, list[float]] = {arm: [] for arm in ARMS}
    calibration_scores: dict[str, list[float]] = {arm: [] for arm in ARMS}
    alarms = {arm: 0 for arm in ARMS}
    block_alarm_counts: dict[str, list[int]] = {arm: [] for arm in ARMS}
    prior: dict[str, list[float] | None] = {}
    hashes: dict[str, str | None] = {arm: None for arm in ARMS}
    fixed_hash: str | None = None
    last_received: float | None = None
    thresholds: dict[str, float] = {}
    last_alarm_position: dict[str, int | None] = {
        arm: None for arm in ARMS
    }
    last_phase: str | None = None

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("each trace row must be an object")
        phase = ("fit" if index < fit_n else
                 "calibration" if index < fit_n + cal_n else
                 "probe" if index < fit_n + cal_n + probe_n else "monitor")
        position = index - phase_offsets[phase]
        if phase != last_phase:
            prior.clear()
            for arm in ARMS:
                hashes[arm] = None
            if phase == "monitor":
                thresholds = {
                    arm: max(1e-9, max(calibration_scores[arm]))
                    for arm in ARMS
                }
            last_phase = phase
        if (type(row.get("i")) is not int or row.get("i") != index
                or row.get("phase") != phase
                or type(row.get("j")) is not int
                or row.get("j") != position):
            raise ValueError("frame index or phase sequence mismatch")
        _close(row.get("timestamp"), (index + 1)
               * float(spec["frame_seconds"]),
               "sample-clock timestamp")
        received = _finite(row.get("received_seconds"), "received_seconds")
        if last_received is not None and received < last_received:
            raise ValueError("received_seconds moved backwards")
        last_received = received
        features = row.get("features")
        if not isinstance(features, dict):
            raise ValueError("features must be an object")
        bands = _vector(features.get("band_dbfs"), "features.band_dbfs")
        arms = row.get("arms")
        if not isinstance(arms, dict):
            raise ValueError("arms must be an object")
        if phase == "fit":
            if arms:
                raise ValueError("fit rows must not expose model arms")
            if row.get("normalized") is not None:
                raise ValueError("fit normalized field must be null")
        elif phase == "probe":
            if arms:
                raise ValueError("probe rows must not expose model arms")
            normalized = _vector(row.get("normalized"), "normalized")
            _same(normalized, _normalize(bands, expected_normalizer),
                  "normalized")
        else:
            normalized = _vector(row.get("normalized"), "normalized")
            _same(normalized, _normalize(bands, expected_normalizer),
                  "normalized")
            if set(arms) != set(ARMS):
                raise ValueError("arm keys mismatch")
            for arm in ARMS:
                item = arms[arm]
                if not isinstance(item, dict):
                    raise ValueError(f"{arm} row must be an object")
                before = item.get("prediction_before")
                after = _vector(item.get("prediction_next"),
                                f"{arm}.prediction_next")
                previous = prior.get(arm)
                if previous is None:
                    if before is not None:
                        raise ValueError(f"{arm} first prediction_before")
                    if (item.get("score") is not None
                            or item.get("sse") is not None):
                        raise ValueError(f"{arm} first score must be null")
                    scored = False
                    score = None
                    sse = None
                else:
                    before_vector = _vector(before,
                                            f"{arm}.prediction_before")
                    _same(before_vector, previous,
                          f"{arm}.prior forecast")
                    errors = [guess - target for guess, target in
                              zip(before_vector, normalized)]
                    sse = sum(error * error for error in errors)
                    score = math.sqrt(sse / 6.0)
                    _close(item.get("score"), score, f"{arm}.score")
                    _close(item.get("sse"), sse, f"{arm}.sse")
                    scored = position >= warmup
                if item.get("scored") is not scored:
                    raise ValueError(f"{arm}.scored mismatch")
                if arm in ("ema", "persistence"):
                    if previous is None or arm == "persistence":
                        expected_after = normalized
                    else:
                        alpha = float(session_protocol["ema_alpha"])
                        expected_after = [
                            alpha * actual + (1.0 - alpha) * prior_value
                            for actual, prior_value in zip(
                                normalized, previous)
                        ]
                    _same(after, expected_after, f"{arm}.prediction_next")
                if phase == "calibration" and scored and score is not None:
                    calibration_scores[arm].append(score)
                if phase == "monitor" and scored and score is not None:
                    monitor_scores[arm].append(score)
                if phase == "monitor":
                    alarm = item.get("alarm")
                    if type(alarm) is not bool:
                        raise ValueError(f"{arm}.alarm must be boolean")
                    expected_alarm = False
                    if scored and score is not None:
                        previous_alarm = last_alarm_position[arm]
                        elapsed = (int(spec["cooldown_frames"])
                                   if previous_alarm is None
                                   else position - previous_alarm)
                        if (score > thresholds[arm]
                                and elapsed >= int(spec["cooldown_frames"])):
                            expected_alarm = True
                            last_alarm_position[arm] = position
                    if alarm is not expected_alarm:
                        raise ValueError(f"{arm}.alarm mismatch")
                    if alarm:
                        alarms[arm] += 1
                    block_index = position // block_n
                    while len(block_alarm_counts[arm]) <= block_index:
                        block_alarm_counts[arm].append(0)
                    block_alarm_counts[arm][block_index] += int(alarm)
                elif item.get("alarm") is not False:
                    raise ValueError(f"{arm} non-monitor alarm must be false")
                _validate_hashes(item, arm, hashes)
                if arm == "fixed_snn":
                    if fixed_hash is None:
                        fixed_hash = item["hash_before"]
                    if item["hash_before"] != fixed_hash \
                            or item["hash_after"] != fixed_hash:
                        raise ValueError("fixed SNN hash changed")
                prior[arm] = after

    for arm in ARMS:
        if not calibration_scores[arm]:
            raise ValueError("calibration has no scored forecasts")
        thresholds[arm] = max(1e-9, max(calibration_scores[arm]))
        _close(summary.get("thresholds", {}).get(arm), thresholds[arm],
               f"thresholds.{arm}")
        _check_summary(summary.get("calibration", {}).get(arm),
                       calibration_scores[arm], f"calibration.{arm}")
        _check_metrics(summary.get("cumulative", {}).get(arm),
                       monitor_scores[arm], f"cumulative.{arm}")
        cumulative_arm = summary["cumulative"][arm]
        _same(cumulative_arm.get("alarm_count"), alarms[arm],
              f"cumulative.{arm}.alarm_count")
        scored_count = len(monitor_scores[arm])
        expected_rate = (alarms[arm] / (scored_count
                         * float(spec["frame_seconds"]) / 60.0)
                         if scored_count else None)
        if expected_rate is None:
            if cumulative_arm.get("alarms_per_minute") is not None:
                raise ValueError(f"cumulative.{arm}.alarms_per_minute")
        else:
            _close(cumulative_arm.get("alarms_per_minute"), expected_rate,
                   f"cumulative.{arm}.alarms_per_minute")
    _same(summary.get("monitor_alarms"), alarms, "monitor_alarms")
    if fixed_hash is not None and summary.get("fixed_hash") != fixed_hash:
        raise ValueError("summary fixed_hash mismatch")
    if summary.get("reference_hash") != summary.get("fixed_hash"):
        raise ValueError("reference_hash differs from fixed_hash")
    rates = summary.get("observed_alarms_per_minute")
    if not isinstance(rates, dict) or set(rates) != set(ARMS):
        raise ValueError("observed alarm rate keys mismatch")
    for arm in ARMS:
        scored_count = len(monitor_scores[arm])
        expected_rate = (alarms[arm] / (scored_count
                         * float(spec["frame_seconds"]) / 60.0)
                         if scored_count else None)
        if expected_rate is None:
            if rates[arm] is not None:
                raise ValueError(f"observed alarm rate mismatch: {arm}")
        else:
            _close(rates[arm], expected_rate,
                   f"observed_alarms_per_minute.{arm}")
    _check_probe(summary, spec, fixed_hash)
    _check_blocks(summary, monitor_scores, block_alarm_counts, block_n,
                  warmup, spec)
    _check_paired_improvements(summary)
    return {"valid": True, "frames": len(rows),
            "monitor_frames": len(rows) - fit_n - cal_n - probe_n,
            "fixed_hash": fixed_hash, "thresholds": thresholds}


def _validate_hashes(item: dict[str, Any], arm: str,
                     previous: dict[str, str | None]) -> None:
    before = item.get("hash_before")
    after = item.get("hash_after")
    if (not isinstance(before, str) or not before
            or not isinstance(after, str) or not after):
        raise ValueError(f"{arm} hashes must be nonempty strings")
    if previous[arm] is not None and before != previous[arm]:
        raise ValueError(f"{arm} hash continuity mismatch")
    if arm in ("fixed_snn", "ema", "persistence") and before != after:
        raise ValueError(f"{arm} parameters changed")
    previous[arm] = after


def _check_probe(summary: dict[str, Any], spec: dict[str, Any],
                 fixed_hash: str | None) -> None:
    probe = summary.get("sealed_probe")
    if (not isinstance(probe, dict)
            or type(probe.get("frames")) is not int
            or probe.get("frames") != int(spec["probe_frames"])):
        raise ValueError("sealed probe frame count mismatch")
    warmup = int(spec["warmup_frames"])
    if (type(probe.get("warmup_frames")) is not int
            or probe.get("warmup_frames") != warmup):
        raise ValueError("sealed probe warmup mismatch")
    counts = probe.get("arms", {})
    if set(counts) != set(ARMS):
        raise ValueError("sealed probe arm keys mismatch")
    for arm in ARMS:
        item = counts[arm]
        expected_count = (int(spec["probe_frames"]) - warmup) * 6
        if (type(item.get("count")) is not int
                or item.get("count") != expected_count
                or type(item.get("feature_target_count")) is not int
                or item.get("feature_target_count") != expected_count
                or type(item.get("frame_count")) is not int
                or item.get("frame_count") != expected_count // 6):
            raise ValueError(f"sealed_probe.{arm}.count mismatch")
        sse = _finite(item.get("sse"), f"sealed_probe.{arm}.sse")
        if sse < 0.0:
            raise ValueError(f"sealed_probe.{arm}.sse must be non-negative")
        _close(item.get("mse"), item["sse"] / expected_count,
               f"sealed_probe.{arm}.mse")
    if probe.get("fixed_hash_before") != fixed_hash \
            or probe.get("fixed_hash_after") != fixed_hash:
        raise ValueError("sealed probe fixed hash mismatch")


def _check_blocks(summary: dict[str, Any], scores: dict[str, list[float]],
                  alarm_counts: dict[str, list[int]], block_n: int,
                  warmup: int, spec: dict[str, Any]) -> None:
    monitor_frames = summary.get("monitor_frames")
    if type(monitor_frames) is not int or monitor_frames < 0:
        raise ValueError("monitor_frames must be a non-negative int")
    expected_count = monitor_frames // block_n
    blocks = summary.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != expected_count:
        raise ValueError("complete block count mismatch")
    for index, block in enumerate(blocks):
        if (type(block.get("block")) is not int
                or block.get("block") != index + 1
                or type(block.get("frames")) is not int
                or block.get("frames") != (index + 1) * block_n):
            raise ValueError("block boundary mismatch")
        for arm in ARMS:
            expected = scores[arm][index * block_n - warmup:
                                   (index + 1) * block_n - warmup]
            if index == 0:
                expected = scores[arm][:block_n - warmup]
            item = block.get("arms", {}).get(arm)
            _check_metrics(item, expected, f"blocks[{index}].{arm}")
            count = alarm_counts[arm][index] if index < len(
                alarm_counts[arm]) else 0
            _same(item.get("alarm_count"), count,
                  f"blocks[{index}].{arm}.alarm_count")
            rate = (count / (len(expected) * float(spec["frame_seconds"])
                             / 60.0) if expected else None)
            if rate is None:
                if item.get("alarms_per_minute") is not None:
                    raise ValueError("block alarm rate mismatch")
                continue
            _close(item.get("alarms_per_minute"), rate,
                   f"blocks[{index}].{arm}.alarms_per_minute")
    checkpoints = summary.get("checkpoints")
    if not isinstance(checkpoints, list) \
            or len(checkpoints) != expected_count + 1:
        raise ValueError("checkpoint count mismatch")
    for index, checkpoint in enumerate(checkpoints):
        expected_frame = index * block_n
        if (type(checkpoint.get("monitor_frame")) is not int
                or checkpoint.get("monitor_frame") != expected_frame):
            raise ValueError("checkpoint boundary mismatch")
        probe = checkpoint.get("sealed_probe")
        if not isinstance(probe, dict):
            raise ValueError("checkpoint missing sealed probe")
        _check_probe({"sealed_probe": probe}, spec,
                     summary.get("fixed_hash"))
        if index == 0:
            _same(probe, summary.get("sealed_probe"),
                  "checkpoints[0].sealed_probe")
        for arm in ("fixed_snn", "ema", "persistence"):
            _same(probe["arms"][arm],
                  summary["sealed_probe"]["arms"][arm],
                  f"checkpoints[{index}].sealed_probe.{arm}")
        if checkpoint.get("fixed_hash") != summary.get("fixed_hash"):
            raise ValueError("checkpoint fixed hash mismatch")


def _check_paired_improvements(summary: dict[str, Any]) -> None:
    """Recompute per-block and cumulative changes against fixed SNN."""
    comparisons = summary.get("adaptive_vs_fixed")
    if not isinstance(comparisons, dict) or set(comparisons) != {
            arm for arm in ARMS if arm != "fixed_snn"}:
        raise ValueError("paired comparison arms mismatch")
    blocks = summary["blocks"]
    cumulative = summary["cumulative"]
    for arm in comparisons:
        deltas = []
        for block in blocks:
            fixed_mse = block["arms"]["fixed_snn"]["mse"]
            arm_mse = block["arms"][arm]["mse"]
            if fixed_mse is not None and arm_mse is not None:
                deltas.append(arm_mse - fixed_mse)
        _same(comparisons[arm].get("block_mse_delta"), deltas,
              f"adaptive_vs_fixed.{arm}.block_mse_delta")
        fixed_total = cumulative["fixed_snn"]["mse"]
        arm_total = cumulative[arm]["mse"]
        delta = (arm_total - fixed_total
                 if fixed_total is not None and arm_total is not None
                 else None)
        relative = ((fixed_total - arm_total) / fixed_total
                    if fixed_total not in (None, 0) and arm_total is not None
                    else None)
        _same(comparisons[arm].get("cumulative_mse_delta"), delta,
              f"adaptive_vs_fixed.{arm}.cumulative_mse_delta")
        _same(comparisons[arm].get("relative_improvement"), relative,
              f"adaptive_vs_fixed.{arm}.relative_improvement")
    for index, block in enumerate(blocks):
        relative_values = block.get("relative_improvement_vs_fixed")
        expected = {}
        fixed_mse = block["arms"]["fixed_snn"]["mse"]
        for arm in ARMS:
            if arm == "fixed_snn":
                continue
            arm_mse = block["arms"][arm]["mse"]
            expected[arm] = ((fixed_mse - arm_mse) / fixed_mse
                             if fixed_mse not in (None, 0)
                             and arm_mse is not None else None)
        _same(relative_values, expected,
              f"blocks[{index}].relative_improvement_vs_fixed")


def audit_session(trace_path: str | Path,
                  summary_path: str | Path) -> dict[str, Any]:
    """Audit newline-delimited private rows and a corresponding summary."""
    rows = []
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("summary must be an object")
    core = summary.get("summary", summary)
    frame_count = core.get("frame_count")
    if type(frame_count) is not int or frame_count < 0:
        raise ValueError("summary frame_count must be a non-negative integer")
    with Path(trace_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if len(rows) >= frame_count:
                break
            if not line.endswith("\n"):
                if len(rows) >= frame_count:
                    break
                raise ValueError(f"incomplete trace line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"trace line {line_number} is not an object")
            rows.append(value)
    if len(rows) < frame_count:
        raise ValueError("trace is shorter than the summary prefix")
    return audit_data(rows[:frame_count], core)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("summary")
    args = parser.parse_args()
    result = audit_session(args.trace, args.summary)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
