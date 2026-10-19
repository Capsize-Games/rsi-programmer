"""Replay a frozen microphone feature prefix across continual learners."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rsi_programmer.audio_continual import AudioContinualSession
from rsi_programmer.audio_runtime import PROJECT_ROOT
from scripts.audit_audio_continual import (
    ARMS,
    NEURAL_ARMS,
    PROTOCOL_SHA256,
    audit_session,
    load_protocol,
)

FRAME_COUNT = 9400
LEGACY_ARMS = ("adaptive_snn", "fixed_snn", "adaptive_ann", "ema",
               "persistence")
LEGACY_REPORT = (PROJECT_ROOT / "artifacts" / "audio-live-session-20261018"
                 / "minute-030" / "report.json")
CODE_FILES = (
    "rsi_programmer/audio_features.py",
    "rsi_programmer/audio_runtime.py",
    "rsi_programmer/audio_continual_runtime.py",
    "rsi_programmer/audio_learning.py",
    "rsi_programmer/audio_memory.py",
    "rsi_programmer/audio_session.py",
    "rsi_programmer/audio_continual.py",
    "rsi_programmer/audio_capture.py",
    "rsi_programmer/sensing_predictor.py",
    "rsi_programmer/sensing_memory.py",
    "scripts/fan_microphone.py",
    "scripts/audit_audio_continual.py",
    "scripts/measure_audio_continual.py",
)


def _read_prefix(path: Path, count: int) \
        -> tuple[list[dict[str, Any]], bytes]:
    rows: list[dict[str, Any]] = []
    payload = bytearray()
    with path.open("rb") as stream:
        for line_number in range(1, count + 1):
            line = stream.readline()
            if not line:
                raise ValueError("input trace is shorter than required prefix")
            if not line.endswith(b"\n"):
                raise ValueError(
                    f"input trace line {line_number} is incomplete")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"input trace line {line_number} is not an object")
            rows.append(value)
            payload.extend(line)
    _validate_source_rows(rows)
    return rows, bytes(payload)


def _validate_source_rows(rows: list[dict[str, Any]]) -> None:
    spec = load_protocol()
    fit_n = int(spec["fit_frames"])
    cal_n = int(spec["calibration_frames"])
    probe_n = int(spec["probe_frames"])
    received_previous: float | None = None
    for index, row in enumerate(rows):
        phase = ("fit" if index < fit_n else
                 "calibration" if index < fit_n + cal_n else
                 "probe" if index < fit_n + cal_n + probe_n else "monitor")
        start = {"fit": 0, "calibration": fit_n,
                 "probe": fit_n + cal_n,
                 "monitor": fit_n + cal_n + probe_n}[phase]
        if (type(row.get("i")) is not int or row["i"] != index
                or row.get("phase") != phase
                or type(row.get("j")) is not int
                or row["j"] != index - start):
            raise ValueError("input trace frame/phase sequence mismatch")
        received = _finite(row.get("received_seconds"), "received_seconds")
        if received_previous is not None and received < received_previous:
            raise ValueError("input delivery timestamp moved backwards")
        received_previous = received
        features = row.get("features")
        if not isinstance(features, dict):
            raise ValueError("input row features must be an object")
        bands = features.get("band_dbfs")
        if not isinstance(bands, list) or len(bands) != 6:
            raise ValueError("input row must have six band_dbfs features")
        for index_band, value in enumerate(bands):
            _finite(value, f"band_dbfs[{index_band}]")


def _finite(value: Any, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _private_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents:
        raise ValueError("private output must be outside the repository")
    resolved.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(resolved, 0o700)
    return resolved


def _public_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=False, mode=0o755)
    os.chmod(resolved, 0o755)
    return resolved


def _write_private_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_private_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        for row in rows:
            stream.write(json.dumps(
                row, sort_keys=True, separators=(",", ":"),
                allow_nan=False) + "\n")


def _write_public_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o644)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _source_hashes() -> dict[str, str]:
    result = {"fixture": hashlib.sha256(
        (PROJECT_ROOT / "fixtures/audio_continual_v1.json").read_bytes()
    ).hexdigest()}
    for name in CODE_FILES:
        result[name] = hashlib.sha256(
            (PROJECT_ROOT / name).read_bytes()).hexdigest()
    return result


def _metric_public(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("metric output must be an object")
    keys = ("count", "frame_count", "feature_target_count", "sse", "mse",
            "alarm_count", "alarms_per_minute")
    return {key: value[key] for key in keys if key in value}


def _public_summary(seed: int, summary: dict[str, Any],
                    audit: dict[str, Any]) -> dict[str, Any]:
    cumulative = summary["cumulative"]
    model_metrics = summary["model_metrics"]
    model_work = {}
    for arm in NEURAL_ARMS:
        source = model_metrics[arm]
        record = {key: source[key] for key in (
            "predictions", "scored_targets", "updates", "neuron_ticks",
            "spikes", "readout_synapse_visits", "input_synapse_visits",
            "changed_readout_parameters", "changed_input_parameters",
        )}
        for key in ("replay_passes", "replay_targets",
                    "online_training_targets",
                    "memory_capacity_frames", "memory_mode",
                    "replay_wall_seconds"):
            if key in source:
                record[key] = source[key]
        model_work[arm] = record
    block_data = []
    for block in summary["blocks"]:
        block_data.append({
            "block": block["block"],
            "monitor_frame_end": block["frames"],
            "arms": {arm: _metric_public(block["arms"][arm])
                     for arm in ARMS},
        })
    probe_checkpoints = []
    initial_probe = summary["checkpoints"][0]["sealed_probe"]["arms"]
    for checkpoint in summary["checkpoints"]:
        probe = checkpoint["sealed_probe"]
        probe_arms = {arm: _metric_public(probe["arms"][arm])
                      for arm in ARMS}
        probe_checkpoints.append({
            "monitor_frame": checkpoint["monitor_frame"],
            "arms": probe_arms,
            "adaptive_snn_vs_controls": _compare_snn(probe_arms),
            "change_from_checkpoint_0_percent": {
                arm: _reduction(probe_arms[arm]["mse"],
                                initial_probe[arm]["mse"])
                for arm in ARMS
            },
        })
    cumulative_public = {
        arm: _metric_public(cumulative[arm]) for arm in ARMS}
    work_accounting = {
        arm: model_work[arm] for arm in NEURAL_ARMS
    }
    for arm in ("memory_snn", "recent_memory_snn", "memory_ann"):
        work_accounting[arm]["replay_wall_seconds"] = model_metrics[arm][
            "replay_wall_seconds"]
    work_accounting["calibration_copies"] = summary["work_accounting"][
        "calibration_copies"]
    work_accounting["sealed_probe_copies"] = summary["work_accounting"][
        "sealed_probe_copies"]
    work_accounting["wall_seconds_scope"] = summary["work_accounting"][
        "wall_seconds_scope"]
    return {
        "seed": seed,
        "independently_audited": True,
        "frame_count": summary["frame_count"],
        "monitor_frames": summary["monitor_frames"],
        "scored_frames": max(0, summary["monitor_frames"] - 25),
        "feature_target_count": max(0, summary["monitor_frames"] - 25) * 6,
        "scored_seconds": max(0, summary["monitor_frames"] - 25) * 0.2,
        "thresholds": summary["thresholds"],
        "cumulative": cumulative_public,
        "adaptive_snn_vs_controls": _compare_snn(cumulative_public),
        "adaptive_vs_fixed": summary["adaptive_vs_fixed"],
        "blocks": block_data,
        "sealed_probe_checkpoints": probe_checkpoints,
        "model_work": model_work,
        "work_accounting": work_accounting,
        "memory_capacity_frames": model_metrics["memory_capacity_frames"],
        "memory_interval_targets": model_metrics["memory_interval_targets"],
        "memory_values_per_bank": model_metrics["memory_values_per_bank"],
        "trainable_parameters_per_model": (
            model_metrics["trainable_parameters_per_model"]),
        "processing_wall_seconds": model_metrics["processing_wall_seconds"],
        "inversion_diagnostic": audit["inversion_diagnostic"],
        "initial_fit_hashes": audit["initial_fit_hashes"],
        "fixed_reference_unchanged": True,
    }


def _reduction(current: float, control: float) -> float | None:
    if control == 0:
        return None
    return 100.0 * (control - current) / control


def _compare_snn(arms: dict[str, Any]) -> dict[str, Any]:
    snn = arms["adaptive_snn"]["mse"]
    return {
        control: {
            "adaptive_snn_mse_reduction_percent": _reduction(
                snn, arms[control]["mse"]),
            "adaptive_snn_sse_ratio": (
                arms["adaptive_snn"]["sse"] / arms[control]["sse"]
                if arms[control]["sse"] else None),
        }
        for control in ARMS if control != "adaptive_snn"
    }


def _legacy_check(report_path: Path,
                  summary: dict[str, Any]) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    arms = report["selected_cumulative"]["arms"]
    actual = summary["cumulative"]
    result = {}
    for arm in LEGACY_ARMS:
        expected = arms[arm]
        observed = actual[arm]
        if (expected["feature_target_count"] !=
                observed["feature_target_count"]
                or expected["frame_count"] != observed["frame_count"]
                or not math.isclose(expected["sse"], observed["sse"],
                                    rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError(f"seed-0 legacy compatibility failed: {arm}")
        result[arm] = {"frame_count": observed["frame_count"],
                       "feature_target_count": observed[
                           "feature_target_count"],
                       "sse": observed["sse"],
                       "matches_prior_report": True}
    return result


def _model_core(summary: dict[str, Any]) -> dict[str, Any]:
    """Remove timing-only metrics before online/replay comparison."""
    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items()
                    if key not in ("runtime", "phase")
                    and not key.endswith("wall_seconds")}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return strip(summary)


def _same_json(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return (set(left) == set(right)
                and all(_same_json(left[key], right[key]) for key in left))
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_json(first, second)
            for first, second in zip(left, right))
    if (isinstance(left, (int, float)) and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)):
        return math.isclose(float(left), float(right), rel_tol=1e-10,
                            abs_tol=1e-10)
    return left == right


def _compare_online(summary_path: Path,
                    replay: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("online summary must be a JSON object")
    online = payload.get("summary", payload)
    if not isinstance(online, dict):
        raise ValueError("online summary body must be an object")
    if not _same_json(_model_core(online), _model_core(replay)):
        raise ValueError("prospective seed-0 online/replay mismatch")
    return {"matches_online_seed0": True}


def run_experiment(trace_path: Path, output_path: Path,
                   private_path: Path, stage: str,
                   legacy_report: Path | None = None,
                   online_summary: Path | None = None) -> dict[str, Any]:
    """Replay the fixed 9,400-row prefix across frozen experiment seeds."""
    spec = load_protocol()
    if stage not in ("retrospective", "prospective"):
        raise ValueError("stage must be retrospective or prospective")
    if stage == "prospective" and online_summary is None:
        raise ValueError("prospective stage requires --online-summary")
    if stage == "retrospective" and legacy_report is None:
        legacy_report = LEGACY_REPORT
    source = trace_path.expanduser().resolve()
    rows, prefix_bytes = _read_prefix(source, FRAME_COUNT)
    private_resolved = private_path.expanduser().resolve()
    public_resolved = output_path.expanduser().resolve()
    if (private_resolved == public_resolved
            or private_resolved in public_resolved.parents
            or public_resolved in private_resolved.parents):
        raise ValueError(
            "public and private output directories must be separate")
    private_root = _private_root(private_path)
    public_root = _public_root(output_path)
    prefix_hash = hashlib.sha256(prefix_bytes).hexdigest()
    receipts = [float(row["received_seconds"]) for row in rows]
    private_meta = {
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_trace_path": str(source),
        "input_prefix_sha256": prefix_hash,
        "input_prefix_frames": FRAME_COUNT,
        "input_delivery_start_seconds": receipts[0],
        "input_delivery_end_seconds": receipts[-1],
        "input_delivery_elapsed_seconds": receipts[-1] - receipts[0],
        "protocol_sha256": PROTOCOL_SHA256,
        "source_sha256": _source_hashes(),
    }
    _write_private_json(private_root / "provenance.json", private_meta)
    public_seeds = []
    private_audits = []
    legacy_result = None
    online_result = None

    for seed in spec["seeds"]:
        protocol = dict(spec)
        protocol["seed"] = seed
        session = AudioContinualSession(protocol)
        private_seed = private_root / f"seed-{seed}"
        private_seed.mkdir(mode=0o700)
        os.chmod(private_seed, 0o700)
        trace_output = private_seed / "trace.jsonl"
        with trace_output.open("x", encoding="utf-8") as stream:
            os.chmod(trace_output, 0o600)
            for source_row in rows:
                output_row = session.observe(
                    source_row["features"],
                    float(source_row["received_seconds"]))
                stream.write(json.dumps(
                    output_row, sort_keys=True, separators=(",", ":"),
                    allow_nan=False) + "\n")
        session.finish()
        summary = session.summary()
        _write_private_json(private_seed / "summary.json", summary)
        _write_private_json(private_seed / "snapshot.json", session.snapshot())
        audit = audit_session(trace_output,
                              private_seed / "summary.json", seed)
        _write_private_json(private_seed / "audit.json", audit)
        public_seeds.append(_public_summary(seed, summary, audit))
        private_audits.append({
            "seed": seed, "valid": audit["valid"],
            "frames": audit["frames"],
            "fixed_hash": audit["fixed_hash"],
            "summary_sha256": audit["summary_sha256"],
        })
        if seed == 0 and stage == "retrospective":
            assert legacy_report is not None
            legacy_result = _legacy_check(legacy_report, summary)
        if seed == 0 and stage == "prospective":
            assert online_summary is not None
            online_result = _compare_online(online_summary, summary)

    source_hashes_after = _source_hashes()
    if source_hashes_after != private_meta["source_sha256"]:
        raise ValueError("experiment source changed during replay")
    private_meta["source_sha256_after"] = source_hashes_after

    public_result = {
        "schema_version": 1,
        "experiment": "audio_continual_v1",
        "stage": stage,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol": {
            key: spec[key] for key in (
                "version", "sample_rate", "frame_seconds", "band_edges_hz",
                "fit_frames", "calibration_frames", "probe_frames",
                "warmup_frames", "cooldown_frames", "block_frames",
                "monitor_frames", "memory_capacity_frames",
                "memory_interval_targets", "seeds", "arms")
        },
        "input_prefix_frames": FRAME_COUNT,
        "input_prefix_sha256": prefix_hash,
        "source_sha256": private_meta["source_sha256"],
        "seed_runs": public_seeds,
        "legacy_seed0_compatibility": legacy_result,
        "online_replay_seed0_agreement": online_result,
        "across_seed_mean": _across_seed_mean(public_seeds),
        "across_seed_interpretation": (
            "Descriptive means across three seeds on one physical tape; "
            "seeds are not independent recordings or trials."),
        "interpretation": (
            "Unlabeled next-feature prediction error and observed alarms; "
            "not detection accuracy, fault success, independent trials, "
            "chip memory, or energy."),
    }
    _write_public_json(public_root / "report.json", public_result)
    _write_private_json(private_root / "audit-index.json", {
        "stage": stage, "input_prefix_sha256": prefix_hash,
        "protocol_sha256": PROTOCOL_SHA256,
        "audits": private_audits,
        "source_sha256": private_meta["source_sha256"],
        "source_sha256_after": source_hashes_after,
    })
    return public_result


def _across_seed_mean(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"seed_count": len(seed_runs), "arms": {}}
    for arm in ARMS:
        records = [seed["cumulative"][arm] for seed in seed_runs]
        target_counts = [record["feature_target_count"] for record in records]
        sse_values = [float(record["sse"]) for record in records]
        if not target_counts or len(set(target_counts)) != 1:
            raise ValueError("seed target counts must match for pooling")
        total_targets = sum(target_counts)
        total_sse = sum(sse_values)
        result["arms"][arm] = {
            "mean_seed_mse": statistics.fmean(
                float(record["mse"]) for record in records),
            "pooled_sse": total_sse,
            "pooled_feature_target_count": total_targets,
            "pooled_mse": total_sse / total_targets,
            "mean_alarms_per_minute": statistics.fmean(
                float(record["alarms_per_minute"]) for record in records),
        }
    result["adaptive_snn_vs_controls"] = _compare_snn({
        arm: {"mse": record["pooled_mse"], "sse": record["pooled_sse"]}
        for arm, record in result["arms"].items()
    })
    checkpoints = [seed["sealed_probe_checkpoints"] for seed in seed_runs]
    if checkpoints and len({len(items) for items in checkpoints}) != 1:
        raise ValueError("sealed-probe checkpoint counts differ across seeds")
    result["sealed_probe_checkpoints"] = []
    for index in range(len(checkpoints[0]) if checkpoints else 0):
        grouped = [items[index] for items in checkpoints]
        if len({item["monitor_frame"] for item in grouped}) != 1:
            raise ValueError("probe checkpoint frames differ across seeds")
        arm_means = {}
        for arm in ARMS:
            records = [item["arms"][arm] for item in grouped]
            targets = [record["feature_target_count"] for record in records]
            if len(set(targets)) != 1:
                raise ValueError("probe target counts differ across seeds")
            total_sse = sum(float(record["sse"]) for record in records)
            total_targets = sum(targets)
            arm_means[arm] = {
                "mean_seed_mse": statistics.fmean(
                    float(record["mse"]) for record in records),
                "pooled_sse": total_sse,
                "pooled_feature_target_count": total_targets,
                "pooled_mse": total_sse / total_targets,
            }
        result["sealed_probe_checkpoints"].append({
            "monitor_frame": grouped[0]["monitor_frame"],
            "arms": arm_means,
        })
    work_means: dict[str, Any] = {}
    for arm in NEURAL_ARMS:
        records = [seed["work_accounting"][arm] for seed in seed_runs]
        work_means[arm] = {}
        keys = set.intersection(*(set(record) for record in records))
        for key in sorted(keys):
            values = [record[key] for record in records]
            if all(isinstance(value, (int, float))
                   and not isinstance(value, bool) for value in values):
                work_means[arm][f"mean_{key}"] = statistics.fmean(
                    float(value) for value in values)
    result["mean_work_accounting"] = work_means
    result["mean_processing_wall_seconds"] = statistics.fmean(
        float(seed["processing_wall_seconds"]) for seed in seed_runs)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=("retrospective", "prospective"),
                        required=True)
    parser.add_argument("--legacy-report", type=Path,
                        default=LEGACY_REPORT)
    parser.add_argument("--online-summary", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_experiment(
            args.trace, args.output_dir, args.private_dir, args.stage,
            args.legacy_report if args.stage == "retrospective" else None,
            args.online_summary)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"continual audio measurement: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "report": str(args.output_dir / "report.json"),
        "stage": result["stage"],
        "seeds": [item["seed"] for item in result["seed_runs"]],
        "audit_count": len(result["seed_runs"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
