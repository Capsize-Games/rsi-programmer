"""Export an exact, sanitized complete-block audio checkpoint report."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

ARMS = (
    "adaptive_snn",
    "fixed_snn",
    "adaptive_ann",
    "ema",
    "persistence",
)
CONTROLS = ("fixed_snn", "adaptive_ann", "ema", "persistence")
PROTOCOL_KEYS = (
    "version",
    "sample_rate",
    "frame_seconds",
    "band_edges_hz",
    "fit_frames",
    "calibration_frames",
    "probe_frames",
    "block_frames",
    "warmup_frames",
    "cooldown_frames",
    "seed",
    "ema_alpha",
    "arms",
)
RUNTIME_KEYS = ("git_head", "protocol_sha256", "status", "started_at")
EXPECTED_PROTOCOL_SHA256 = (
    "ef7019083c0fcb24f3477a8384a5b3deccd260fd1ad9f7ef1d133c433b065561"
)


def _summary_from_status(status: dict[str, Any]) -> dict[str, Any]:
    """Find the audio session summary in direct or wrapped status data."""
    candidates = [status]
    for key in ("summary", "session", "result", "aggregate"):
        value = status.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    for candidate in candidates:
        if isinstance(candidate.get("blocks"), list):
            return candidate
    raise ValueError("status JSON has no block summary")


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    """Require a finite non-boolean number at or above a minimum."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value < minimum):
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return float(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    """Require an exact integer value without accepting booleans."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _protocol(summary: dict[str, Any]) -> dict[str, Any]:
    raw = _mapping(summary.get("protocol"), "protocol")
    result = {key: raw[key] for key in PROTOCOL_KEYS if key in raw}
    for key in ("version", "sample_rate", "fit_frames",
                "calibration_frames", "probe_frames", "block_frames",
                "warmup_frames", "cooldown_frames", "seed"):
        minimum = 0 if key in ("warmup_frames", "cooldown_frames",
                               "seed") else 1
        _integer(result.get(key), f"protocol.{key}", minimum=minimum)
    if _integer(result.get("block_frames"), "block_frames", minimum=1) != 300:
        raise ValueError("block_frames must be exactly 300")
    if _number(result.get("frame_seconds"), "frame_seconds") != 0.2:
        raise ValueError("frame_seconds must be exactly 0.2")
    if _integer(result.get("probe_frames"), "probe_frames", minimum=1) != 100:
        raise ValueError("probe_frames must be exactly 100")
    if _integer(result.get("warmup_frames"), "warmup_frames") != 25:
        raise ValueError("warmup_frames must be exactly 25")
    if result["version"] != 1:
        raise ValueError("unsupported audio protocol version")
    if result["sample_rate"] != 16000:
        raise ValueError("sample_rate must be exactly 16000")
    if result["fit_frames"] != 150 or result["calibration_frames"] != 150:
        raise ValueError("fit and calibration windows must each be 150 frames")
    if result["seed"] != 0:
        raise ValueError("seed must be exactly 0")
    alpha = _number(result.get("ema_alpha"), "protocol.ema_alpha")
    expected_alpha = 1.0 - math.exp(-0.2 / 5.0)
    if alpha != expected_alpha:
        raise ValueError(
            "protocol.ema_alpha does not match the five-second EMA")
    edges = result.get("band_edges_hz")
    expected_edges = [80, 160, 320, 640, 1280, 2560, 6000]
    if edges != expected_edges:
        raise ValueError("band_edges_hz does not match the fixed protocol")
    if result.get("arms") != list(ARMS):
        raise ValueError("protocol arms do not match the five fixed arms")
    if result["cooldown_frames"] != 25:
        raise ValueError("cooldown_frames must be exactly 25")
    return result


def _safe_runtime(status: dict[str, Any],
                  summary: dict[str, Any]) -> dict[str, Any]:
    """Keep only explicitly permitted runtime provenance fields."""
    runtime = status.get("runtime")
    if not isinstance(runtime, dict):
        runtime = summary.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
    protocol_hash = runtime.get("protocol_sha256")
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(
            "runtime protocol_sha256 does not match frozen recipe")
    result: dict[str, Any] = {
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    }
    git_head = runtime.get("git_head")
    if git_head is not None:
        if (not isinstance(git_head, str)
                or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}",
                                    git_head)):
            raise ValueError(
                "runtime git_head must be a safe hexadecimal hash")
        result["git_head"] = git_head
    status_value = runtime.get("status")
    if status_value is not None:
        if (not isinstance(status_value, str) or not status_value
                or len(status_value) > 32
                or not re.fullmatch(r"[A-Za-z0-9_-]+", status_value)):
            raise ValueError("runtime status must be a short safe label")
        result["status"] = status_value
    started_at = runtime.get("started_at")
    if started_at is not None:
        if (not isinstance(started_at, str) or len(started_at) > 64
                or not re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}T[0-9:.+Z-]+", started_at)):
            raise ValueError(
                "runtime started_at must be an ISO-like timestamp")
        result["started_at"] = started_at
    return result


def _mse_record(arm_data: Any, label: str,
                *, expected_frames: int | None = None,
                require_alarm: bool = True,
                ) -> dict[str, Any]:
    """Validate frame, feature-target, SSE, and MSE evidence."""
    value = _mapping(arm_data, label)
    frames = _integer(value.get("frame_count"), f"{label}.frame_count",
                      minimum=1)
    targets = _integer(value.get("feature_target_count"),
                       f"{label}.feature_target_count", minimum=1)
    sse = _number(value.get("sse"), f"{label}.sse")
    mse = _number(value.get("mse"), f"{label}.mse")
    if targets != frames * 6:
        raise ValueError(f"{label} target count must equal six per frame")
    if expected_frames is not None and frames != expected_frames:
        raise ValueError(f"{label} has an unexpected scored-frame count")
    expected_mse = sse / targets
    if not math.isclose(mse, expected_mse, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"{label} MSE does not match SSE / target count")
    record = {
        "frame_count": frames,
        "feature_target_count": targets,
        "sse": sse,
        "mse": mse,
    }
    if require_alarm:
        alarms = _integer(value.get("alarm_count"), f"{label}.alarm_count")
        rate = _number(value.get("alarms_per_minute"),
                       f"{label}.alarms_per_minute")
        record["alarm_count"] = alarms
        record["alarms_per_minute"] = rate
    return record


def _validate_alarm_rate(record: dict[str, Any], frame_seconds: float,
                         label: str) -> None:
    scored_minutes = record["frame_count"] * frame_seconds / 60.0
    expected = record["alarm_count"] / scored_minutes
    if not math.isclose(record["alarms_per_minute"], expected,
                        rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"{label} alarm rate does not match scored frames")


def _reduction(current: float, control: float) -> float | None:
    if control == 0:
        return None
    return 100.0 * (control - current) / control


def _probe_checkpoint(
    checkpoint: Any, index: int, probe_frames: int, warmup_frames: int,
    fixed_hash: str, frame_seconds: float,
) -> dict[str, Any]:
    """Validate one exact sealed-probe checkpoint and its fixed hash."""
    value = _mapping(checkpoint, f"checkpoint[{index}]")
    expected_frame = index * 300
    if _integer(value.get("monitor_frame"),
                f"checkpoint[{index}].monitor_frame") != expected_frame:
        raise ValueError("probe checkpoints must be exactly 300 frames apart")
    if value.get("fixed_hash") != fixed_hash:
        raise ValueError("fixed hash changed at a probe checkpoint")
    probe = _mapping(value.get("sealed_probe"),
                     f"checkpoint[{index}].sealed_probe")
    if probe.get("fixed_hash_before") != fixed_hash:
        raise ValueError("probe evaluation began with a different fixed hash")
    if probe.get("fixed_hash_after") != fixed_hash:
        raise ValueError("probe evaluation changed the fixed hash")
    if _integer(probe.get("frames"), "probe.frames") != probe_frames:
        raise ValueError("probe frame count does not match protocol")
    if _integer(probe.get("warmup_frames"),
                "probe.warmup_frames") != warmup_frames:
        raise ValueError("probe warmup does not match protocol")
    arms = _mapping(probe.get("arms"), "probe.arms")
    scored_frames = probe_frames - warmup_frames
    records = {
        arm: _mse_record(arms.get(arm), f"probe.{arm}",
                         expected_frames=scored_frames, require_alarm=False)
        for arm in ARMS
    }
    for record in records.values():
        record["count"] = record["feature_target_count"]
    minute = expected_frame * frame_seconds / 60.0
    return {
        "checkpoint": index,
        "monitor_frame": expected_frame,
        "minute": minute,
        "fixed_hash": fixed_hash,
        "sealed_probe": {
            "frames": probe_frames,
            "warmup_frames": warmup_frames,
            "fixed_hash_before": fixed_hash,
            "fixed_hash_after": fixed_hash,
            "arms": records,
        },
        "arms": records,
    }


def _comparisons(arms: dict[str, Any]) -> dict[str, Any]:
    snn_mse = arms["adaptive_snn"]["mse"]
    return {
        control: {
            "adaptive_snn_mse_reduction_percent": _reduction(
                snn_mse, arms[control]["mse"]),
            "adaptive_snn_sse_ratio": (
                arms["adaptive_snn"]["sse"] / arms[control]["sse"]
                if arms[control]["sse"] else None
            ),
        }
        for control in CONTROLS
    }


def build_report(status: dict[str, Any], source_bytes: bytes,
                 minutes: int) -> dict[str, Any]:
    """Select and recompute an exact first-N-minute report."""
    if type(minutes) is not int or minutes < 1:
        raise ValueError("minutes must be a positive whole number")
    summary = _summary_from_status(status)
    protocol = _protocol(summary)
    block_frames = int(protocol["block_frames"])
    frame_seconds = float(protocol["frame_seconds"])
    warmup = int(protocol["warmup_frames"])
    blocks = summary["blocks"]
    if len(blocks) < minutes:
        raise ValueError("status does not contain enough complete blocks")

    selected_blocks: list[dict[str, Any]] = []
    totals = {
        arm: {
            "frame_count": 0, "feature_target_count": 0,
            "sse": 0.0, "alarm_count": 0,
        } for arm in ARMS
    }
    for block_index in range(minutes):
        block = _mapping(blocks[block_index], f"blocks[{block_index}]")
        if _integer(block.get("block"), f"blocks[{block_index}].block",
                    minimum=1) != block_index + 1:
            raise ValueError("selected block numbers must be ordered from 1")
        expected_end = (block_index + 1) * block_frames
        original_frames = _integer(
            block.get("frames"), f"blocks[{block_index}].frames")
        if original_frames != expected_end:
            raise ValueError("selected blocks must be complete and ordered")
        scored_frames = (block_frames - warmup
                         if block_index == 0 else block_frames)
        arms_data = _mapping(block.get("arms"),
                             f"blocks[{block_index}].arms")
        records: dict[str, Any] = {}
        for arm in ARMS:
            record = _mse_record(
                arms_data.get(arm), f"blocks[{block_index}].{arm}",
                expected_frames=scored_frames)
            _validate_alarm_rate(
                record, frame_seconds, f"blocks[{block_index}].{arm}")
            records[arm] = record
            total = totals[arm]
            total["frame_count"] += record["frame_count"]
            total["feature_target_count"] += record["feature_target_count"]
            total["sse"] += record["sse"]
            total["alarm_count"] += record["alarm_count"]
        selected_blocks.append({
            "block": block_index + 1,
            "frames": original_frames,
            "minute": block_index + 0.5,
            "minute_interval": [
                block_index * block_frames * frame_seconds / 60.0,
                expected_end * frame_seconds / 60.0,
            ],
            "arms": records,
            "adaptive_snn_vs_controls": _comparisons(records),
        })

    checkpoints_raw = _mapping(
        {"checkpoints": summary.get("checkpoints")},
        "checkpoint collection").get("checkpoints")
    if not isinstance(checkpoints_raw, list):
        raise ValueError("checkpoints must be an array")
    fixed_hash = summary.get("fixed_hash")
    reference_hash = summary.get("reference_hash")
    if not isinstance(fixed_hash, str) or not re.fullmatch(
            r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", fixed_hash):
        raise ValueError("fixed_hash must be a SHA-1 or SHA-256 hex digest")
    if reference_hash != fixed_hash:
        raise ValueError("reference_hash and fixed_hash must match")
    by_frame: dict[int, Any] = {}
    for checkpoint in checkpoints_raw:
        value = _mapping(checkpoint, "checkpoint")
        frame = _integer(value.get("monitor_frame"),
                         "checkpoint.monitor_frame")
        if frame in by_frame:
            raise ValueError("duplicate checkpoint frame")
        if value.get("fixed_hash") != fixed_hash:
            raise ValueError("fixed hash changed at a checkpoint")
        probe_value = _mapping(value.get("sealed_probe"),
                               f"checkpoint[{frame}].sealed_probe")
        if (probe_value.get("fixed_hash_before") != fixed_hash
                or probe_value.get("fixed_hash_after") != fixed_hash):
            raise ValueError("fixed hash changed during probe evaluation")
        by_frame[frame] = value
    if any(frame % block_frames != 0 for frame in by_frame):
        raise ValueError("checkpoints must align with block boundaries")
    validated_checkpoints = {
        frame: _probe_checkpoint(
            value, frame // block_frames, int(protocol["probe_frames"]),
            warmup, fixed_hash, frame_seconds)
        for frame, value in by_frame.items()
    }
    expected_checkpoint_frames = [
        index * block_frames for index in range(minutes + 1)
    ]
    if any(frame not in validated_checkpoints
           for frame in expected_checkpoint_frames):
        raise ValueError("required checkpoint 0..N is incomplete")
    selected_checkpoints = [
        validated_checkpoints[frame] for frame in expected_checkpoint_frames
    ]

    baseline = selected_checkpoints[0]["arms"]
    for checkpoint in validated_checkpoints.values():
        for arm in ("fixed_snn", "ema", "persistence"):
            base_record = baseline[arm]
            current_record = checkpoint["arms"][arm]
            for field in ("frame_count", "feature_target_count", "sse",
                          "mse"):
                if current_record[field] != base_record[field]:
                    raise ValueError(
                        f"{arm} probe metrics changed from checkpoint 0")
    for checkpoint in selected_checkpoints:
        checkpoint["adaptive_snn_vs_controls"] = _comparisons(
            checkpoint["arms"])
        checkpoint["change_from_checkpoint_0_percent"] = {
            arm: _reduction(
                checkpoint["arms"][arm]["mse"], baseline[arm]["mse"])
            for arm in ARMS
        }

    cumulative: dict[str, Any] = {}
    selected_scored_frames = 0
    for arm in ARMS:
        total = totals[arm]
        frames = total["frame_count"]
        target_count = total["feature_target_count"]
        sse = total["sse"]
        alarms = total["alarm_count"]
        scored_minutes = frames * frame_seconds / 60.0
        selected_scored_frames = frames
        cumulative[arm] = {
            **total,
            "mse": sse / target_count,
            "scored_minutes": scored_minutes,
            "alarms_per_minute": alarms / scored_minutes,
        }
    cumulative_comparisons = _comparisons(cumulative)
    per_minute_comparisons = [
        {
            "block": block["block"],
            "minute": block["minute"],
            "adaptive_snn_vs_controls": block[
                "adaptive_snn_vs_controls"],
        }
        for block in selected_blocks
    ]
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    status_monitor_frames = _integer(
        summary.get("monitor_frames"), "capture status monitor_frames")
    report_monitor_frames = minutes * block_frames
    if status_monitor_frames < report_monitor_frames:
        raise ValueError("status monitor frame count is below selected scope")
    runtime = _safe_runtime(status, summary)
    total_frame_count = (
        int(protocol["fit_frames"])
        + int(protocol["calibration_frames"])
        + int(protocol["probe_frames"])
        + report_monitor_frames
    )
    summary_payload = {
        "schema_version": 1,
        "protocol": protocol,
        "runtime": runtime,
        "source_status_sha256": source_hash,
        "reference_hash": reference_hash,
        "fixed_hash": fixed_hash,
        "frame_count": total_frame_count,
        "fit_frames": int(protocol["fit_frames"]),
        "calibration_frames": int(protocol["calibration_frames"]),
        "probe_frames": int(protocol["probe_frames"]),
        "monitor_frames": report_monitor_frames,
        "capture_status_monitor_frames": status_monitor_frames,
        "scored_frame_count": selected_scored_frames,
        "feature_target_count": selected_scored_frames * 6,
        "blocks": selected_blocks,
        "checkpoints": [
            {
                "monitor_frame": checkpoint["monitor_frame"],
                "fixed_hash": checkpoint["fixed_hash"],
                "sealed_probe": checkpoint["sealed_probe"],
            }
            for checkpoint in selected_checkpoints
        ],
        "cumulative": cumulative,
    }
    return {
        "schema_version": 1,
        "scope": {
            "requested_minutes": minutes,
            "block_frames": block_frames,
            "frame_seconds": frame_seconds,
            "report_monitor_frames": report_monitor_frames,
            "capture_status_monitor_frames": status_monitor_frames,
            "complete_blocks_only": True,
        },
        "metadata": {
            "source_status_sha256": source_hash,
            **runtime,
        },
        "protocol": protocol,
        "reference_hash": reference_hash,
        "fixed_hash": fixed_hash,
        "selected_blocks": selected_blocks,
        "per_minute_adaptive_snn_vs_controls": per_minute_comparisons,
        "selected_checkpoints": selected_checkpoints,
        "selected_cumulative": {
            "scored_frame_count": selected_scored_frames,
            "arms": cumulative,
            "adaptive_snn_vs_controls": cumulative_comparisons,
        },
        "summary": summary_payload,
    }


def main() -> None:
    """Validate source status and write the sanitized report JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status_json", type=Path)
    parser.add_argument("--minutes", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.minutes < 1:
        parser.error("--minutes must be a positive whole number")

    source_bytes = args.status_json.read_bytes()
    status = json.loads(source_bytes)
    if not isinstance(status, dict):
        raise ValueError("status JSON must contain an object")
    report = build_report(status, source_bytes, args.minutes)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for filename, payload in (
            ("report.json", report), ("summary.json", report["summary"])):
        with (args.output_dir / filename).open(
                "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
