"""Plot pooled continual-learning audio prediction errors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ARMS = (
    "adaptive_snn", "readout_snn", "memory_snn", "recent_memory_snn",
    "fixed_snn", "adaptive_ann", "memory_ann", "ema", "persistence",
)
PROTOCOL_SHA256 = (
    "ff4a161882904aca6ba31a8498a859a1347addaa289e989ec0ea1e231a70fb58"
)
DISPLAY = {
    "adaptive_snn": "Adaptive SNN",
    "readout_snn": "Readout SNN",
    "memory_snn": "Fit-memory SNN",
    "recent_memory_snn": "Recent-memory SNN",
    "fixed_snn": "Fixed SNN",
    "adaptive_ann": "Adaptive ANN",
    "memory_ann": "Fit-memory ANN",
    "ema": "EMA",
    "persistence": "Persistence",
}
COLORS = {
    "adaptive_snn": "#0072B2",
    "readout_snn": "#56B4E9",
    "memory_snn": "#009E73",
    "recent_memory_snn": "#006B50",
    "fixed_snn": "#D55E00",
    "adaptive_ann": "#CC79A7",
    "memory_ann": "#E69F00",
    "ema": "#666666",
    "persistence": "#111111",
}
LINESTYLES = {
    "adaptive_snn": "-",
    "readout_snn": "--",
    "memory_snn": "-.",
    "recent_memory_snn": ":",
    "fixed_snn": "--",
    "adaptive_ann": "-",
    "memory_ann": "-.",
    "ema": ":",
    "persistence": "--",
}
MARKERS = {
    "adaptive_snn": "o",
    "readout_snn": "s",
    "memory_snn": "^",
    "recent_memory_snn": "D",
    "fixed_snn": "x",
    "adaptive_ann": "v",
    "memory_ann": "P",
    "ema": "*",
    "persistence": "X",
}
STAGE_LABELS = {
    "retrospective": "Retrospective recording replay",
    "prospective": "Prospective recording",
}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, minimum: float = 0.0) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value < minimum):
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return float(value)


def _arm_metrics(value: Any, label: str, expected_frames: int
                 ) -> dict[str, float | int]:
    record = _object(value, label)
    frames = _integer(record.get("frame_count"), f"{label}.frame_count", 1)
    targets = _integer(record.get("feature_target_count"),
                       f"{label}.feature_target_count", 1)
    sse = _number(record.get("sse"), f"{label}.sse")
    mse = _number(record.get("mse"), f"{label}.mse")
    if frames != expected_frames or targets != frames * 6:
        raise ValueError(f"{label} frame or feature count does not match")
    if not math.isclose(mse, sse / targets,
                        rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"{label} MSE does not match SSE / targets")
    return {
        "frame_count": frames,
        "feature_target_count": targets,
        "sse": sse,
        "mse": mse,
    }


def _seed_runs(report: dict[str, Any], block_frames: int,
               warmup_frames: int, monitor_frames: int,
               probe_frames: int) -> list[dict[str, Any]]:
    runs = _array(report.get("seed_runs"), "seed_runs")
    protocol = _object(report.get("protocol"), "protocol")
    expected_seeds = _array(protocol.get("seeds"), "protocol.seeds")
    if len(runs) != 3 or expected_seeds != [0, 1, 2]:
        raise ValueError("report must contain the three frozen seed runs")
    expected_blocks = monitor_frames // block_frames
    if monitor_frames % block_frames:
        raise ValueError("monitor frame count must contain complete blocks")
    expected_probe_frames = probe_frames - warmup_frames
    validated: list[dict[str, Any]] = []
    for index, raw_run in enumerate(runs):
        run = _object(raw_run, f"seed_runs[{index}]")
        if _integer(run.get("seed"), f"seed_runs[{index}].seed") != index:
            raise ValueError("seed runs must be ordered 0, 1, 2")
        if _integer(run.get("monitor_frames"), "monitor_frames") != (
                monitor_frames):
            raise ValueError("seed monitor lengths differ from protocol")
        blocks = _array(run.get("blocks"), f"seed_runs[{index}].blocks")
        if len(blocks) != expected_blocks:
            raise ValueError("seed run does not contain all complete blocks")
        checked_blocks = []
        for block_index, raw_block in enumerate(blocks):
            block = _object(raw_block, f"seed[{index}].blocks[{block_index}]")
            expected_number = block_index + 1
            if _integer(block.get("block"), "block", 1) != expected_number:
                raise ValueError("block numbers must be consecutive")
            end_frame = _integer(block.get("monitor_frame_end"),
                                 "monitor_frame_end", 1)
            if end_frame != expected_number * block_frames:
                raise ValueError("monitor blocks must be complete and ordered")
            frames = (block_frames - warmup_frames
                      if block_index == 0 else block_frames)
            arm_data = _object(block.get("arms"), "block.arms")
            checked_blocks.append({
                "block": expected_number,
                "monitor_frame_end": end_frame,
                "arms": {
                    arm: _arm_metrics(arm_data.get(arm),
                                      f"block[{expected_number}].{arm}",
                                      frames)
                    for arm in ARMS
                },
            })
        checkpoints = _array(
            run.get("sealed_probe_checkpoints"),
            f"seed_runs[{index}].sealed_probe_checkpoints")
        if len(checkpoints) != expected_blocks + 1:
            raise ValueError("seed run lacks required probe checkpoints")
        checked_checkpoints = []
        for checkpoint_index, raw_checkpoint in enumerate(checkpoints):
            checkpoint = _object(
                raw_checkpoint,
                f"seed[{index}].checkpoint[{checkpoint_index}]")
            expected_frame = checkpoint_index * block_frames
            frame = _integer(checkpoint.get("monitor_frame"),
                             "checkpoint.monitor_frame")
            if frame != expected_frame:
                raise ValueError("probe checkpoints must align with blocks")
            arm_data = _object(checkpoint.get("arms"), "checkpoint.arms")
            checked_checkpoints.append({
                "monitor_frame": frame,
                "arms": {
                    arm: _arm_metrics(arm_data.get(arm),
                                      f"checkpoint[{expected_frame}].{arm}",
                                      expected_probe_frames)
                    for arm in ARMS
                },
            })
        validated.append({
            "seed": index,
            "blocks": checked_blocks,
            "sealed_probe_checkpoints": checked_checkpoints,
        })
    return validated


def build_chart_data(report: dict[str, Any],
                     report_sha256: str | None = None) -> dict[str, Any]:
    """Validate report data and pool SSE and feature targets across seeds."""
    _object(report, "report")
    if report.get("experiment") != "audio_continual_v1":
        raise ValueError("unsupported experiment report")
    if report.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("report protocol hash does not match frozen fixture")
    if report.get("stage") not in ("retrospective", "prospective"):
        raise ValueError("report stage must be retrospective or prospective")
    protocol = _object(report.get("protocol"), "protocol")
    arms = _array(protocol.get("arms"), "protocol.arms")
    if arms != list(ARMS):
        raise ValueError("report arms do not match the frozen arm order")
    frame_seconds = _number(protocol.get("frame_seconds"),
                            "protocol.frame_seconds", 1e-12)
    block_frames = _integer(protocol.get("block_frames"),
                            "protocol.block_frames", 1)
    warmup_frames = _integer(protocol.get("warmup_frames"),
                             "protocol.warmup_frames")
    probe_frames = _integer(protocol.get("probe_frames"),
                            "protocol.probe_frames", 1)
    monitor_frames = _integer(protocol.get("monitor_frames"),
                              "protocol.monitor_frames", 1)
    if (frame_seconds != 0.2 or block_frames != 300
            or warmup_frames != 25 or probe_frames != 100
            or monitor_frames != 9000):
        raise ValueError("report timing does not match the frozen protocol")
    if probe_frames <= warmup_frames:
        raise ValueError("probe must contain scored targets after warmup")
    seeds = _seed_runs(report, block_frames, warmup_frames,
                       monitor_frames, probe_frames)

    live_series: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in ARMS
    }
    running = {
        seed["seed"]: {
            arm: {"sse": 0.0, "feature_target_count": 0}
            for arm in ARMS
        } for seed in seeds
    }
    for block_index in range(monitor_frames // block_frames):
        frame_end = (block_index + 1) * block_frames
        for arm in ARMS:
            per_seed: list[dict[str, Any]] = []
            pooled_sse = 0.0
            pooled_targets = 0
            for seed in seeds:
                record = seed["blocks"][block_index]["arms"][arm]
                total = running[seed["seed"]][arm]
                total["sse"] += record["sse"]
                total["feature_target_count"] += record[
                    "feature_target_count"]
                seed_mse = total["sse"] / total["feature_target_count"]
                per_seed.append({
                    "seed": seed["seed"],
                    "sse": total["sse"],
                    "feature_target_count": total[
                        "feature_target_count"],
                    "mse": seed_mse,
                })
                pooled_sse += record["sse"]
                pooled_targets += record["feature_target_count"]
            pooled_sse_cumulative = sum(item["sse"] for item in per_seed)
            pooled_targets_cumulative = sum(
                item["feature_target_count"] for item in per_seed)
            live_series[arm].append({
                "block": block_index + 1,
                "monitor_frame_end": frame_end,
                "minute": frame_end * frame_seconds / 60.0,
                "sse_this_block_pooled": pooled_sse,
                "feature_targets_this_block_pooled": pooled_targets,
                "pooled_sse_cumulative": pooled_sse_cumulative,
                "pooled_feature_target_count_cumulative": (
                    pooled_targets_cumulative),
                "pooled_mse_cumulative": (
                    pooled_sse_cumulative / pooled_targets_cumulative),
                "per_seed_cumulative": per_seed,
            })

    pooled_probe: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in ARMS
    }
    for checkpoint_index in range(monitor_frames // block_frames + 1):
        frame = checkpoint_index * block_frames
        for arm in ARMS:
            per_seed = []
            pooled_sse = 0.0
            pooled_targets = 0
            for seed in seeds:
                record = seed["sealed_probe_checkpoints"][
                    checkpoint_index]["arms"][arm]
                per_seed.append({
                    "seed": seed["seed"],
                    "sse": record["sse"],
                    "feature_target_count": record[
                        "feature_target_count"],
                    "mse": record["mse"],
                })
                pooled_sse += record["sse"]
                pooled_targets += record["feature_target_count"]
            pooled_probe[arm].append({
                "monitor_frame": frame,
                "minute": frame * frame_seconds / 60.0,
                "pooled_sse": pooled_sse,
                "pooled_feature_target_count": pooled_targets,
                "pooled_mse": pooled_sse / pooled_targets,
                "per_seed": per_seed,
            })

    probe_change: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in ARMS
    }
    probe_baselines: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        baseline = pooled_probe[arm][0]
        probe_baselines[arm] = {
            "checkpoint": 0,
            "pooled_sse": baseline["pooled_sse"],
            "pooled_feature_target_count": (
                baseline["pooled_feature_target_count"]),
            "pooled_mse": baseline["pooled_mse"],
            "per_seed": baseline["per_seed"],
        }
        for current in pooled_probe[arm]:
            reduction = None
            if baseline["pooled_mse"] != 0:
                reduction = 100.0 * (
                    baseline["pooled_mse"] - current["pooled_mse"]
                ) / baseline["pooled_mse"]
            probe_change[arm].append({
                **current,
                "checkpoint_0_pooled_mse": baseline["pooled_mse"],
                "mse_reduction_percent_vs_checkpoint_0": reduction,
            })

    return {
        "schema_version": 1,
        "experiment": "audio_continual_v1",
        "stage": report.get("stage"),
        "units": {
            "x": "minutes on the monitoring sample clock",
            "live_y": "pooled cumulative next-feature MSE",
            "probe_y": "MSE reduction from checkpoint 0 (%)",
        },
        "pooling": {
            "seed_count": len(seeds),
            "seeds": [seed["seed"] for seed in seeds],
            "interpretation": (
                "The three seeds use one physical feature tape; they are "
                "not independent recordings or sessions."),
            "live_formula": (
                "sum cumulative block SSE across seeds divided by summed "
                "feature target counts across seeds"),
            "probe_formula": (
                "100 * (pooled checkpoint-0 MSE - pooled current MSE) / "
                "pooled checkpoint-0 MSE for each arm"),
        },
        "protocol": {
            "frame_seconds": frame_seconds,
            "block_frames": block_frames,
            "warmup_frames": warmup_frames,
            "probe_frames": probe_frames,
            "monitor_frames": monitor_frames,
            "arms": list(ARMS),
        },
        "live_cumulative": {
            "series": live_series,
            "lower_is_better": True,
        },
        "sealed_probe_change": {
            "baseline_by_arm": probe_baselines,
            "series": probe_change,
            "positive_is_lower_error": True,
            "negative_values_retained": True,
        },
        "metadata": {
            "protocol_sha256": report.get("protocol_sha256"),
            "input_prefix_sha256": report.get("input_prefix_sha256"),
            "report_sha256": report_sha256,
        },
    }


def render_charts(data: dict[str, Any], output_dir: Path) -> list[Path]:
    """Render two pooled error charts and write their exact source data."""
    if not output_dir.is_dir():
        raise ValueError("--output-dir must be an existing directory")
    names = (
        "audio-continual-live-cumulative-mse.png",
        "audio-continual-sealed-probe-change.png",
        "audio-continual-chart-data.json",
    )
    paths = [output_dir / name for name in names]
    if any(path.exists() for path in paths):
        raise ValueError("one or more continual plot outputs already exist")

    import matplotlib.pyplot as plt

    live = data["live_cumulative"]["series"]
    probe = data["sealed_probe_change"]["series"]
    stage_label = STAGE_LABELS[data["stage"]]
    live_path, probe_path, data_path = paths
    _render_panel(
        live_path,
        title=f"{stage_label}: live cumulative next-feature error",
        ylabel="Pooled cumulative MSE",
        series={arm: [
            (point["minute"], point["pooled_mse_cumulative"])
            for point in live[arm]
        ] for arm in ARMS},
        reference_zero=False,
        note=("Lower error is better. Three seeds share one physical tape; "
              "they are not independent sessions."),
        plt=plt,
    )
    _render_panel(
        probe_path,
        title=f"{stage_label}: sealed probe change from checkpoint 0",
        ylabel="Pooled MSE reduction (%)",
        series={arm: [
            (point["minute"], point[
                "mse_reduction_percent_vs_checkpoint_0"])
            for point in probe[arm]
        ] for arm in ARMS},
        reference_zero=True,
        note=("Positive values mean lower probe error; negative outcomes "
              "are retained. Baseline is each arm's pooled checkpoint-0 MSE."),
        plt=plt,
    )
    with data_path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return [live_path, probe_path, data_path]


def _render_panel(path: Path, title: str, ylabel: str,
                  series: dict[str, list[tuple[float, float | None]]],
                  reference_zero: bool, note: str, plt: Any) -> None:
    figure, axis = plt.subplots(figsize=(12.8, 7.4), constrained_layout=True)
    for arm in ARMS:
        points = series[arm]
        x_values = [point[0] for point in points]
        y_values = [float("nan") if point[1] is None else point[1]
                    for point in points]
        axis.plot(
            x_values, y_values, label=DISPLAY[arm], color=COLORS[arm],
            linestyle=LINESTYLES[arm], marker=MARKERS[arm],
            linewidth=2.0 if arm in ("adaptive_snn", "adaptive_ann") else 1.5,
            markersize=5,
        )
    if reference_zero:
        axis.axhline(0.0, color="#222222", linewidth=1.0, alpha=0.7)
    axis.set_title(title, fontsize=14)
    axis.set_xlabel("Monitoring time (minutes, sample clock)", fontsize=12)
    axis.set_ylabel(ylabel, fontsize=12)
    axis.tick_params(axis="both", labelsize=10)
    axis.grid(True, which="major", alpha=0.25)
    axis.legend(loc="best", ncol=3, fontsize=10, frameon=True)
    figure.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=10)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        source_bytes = args.report.read_bytes()
        report = json.loads(source_bytes)
        report_hash = hashlib.sha256(source_bytes).hexdigest()
        data = build_chart_data(
            _object(report, "report"), report_sha256=report_hash)
        paths = render_charts(data, args.output_dir)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({"outputs": [str(path) for path in paths]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
