"""Plot sanitized live and sealed-probe audio prediction errors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ARMS = (
    "adaptive_snn",
    "fixed_snn",
    "adaptive_ann",
    "ema",
    "persistence",
)
ADAPTIVE_ARMS = ("adaptive_snn", "adaptive_ann")
DISPLAY = {
    "adaptive_snn": "Adaptive SNN",
    "fixed_snn": "Fixed SNN",
    "adaptive_ann": "Adaptive ANN",
    "ema": "EMA",
    "persistence": "Persistence",
}
COLORS = {
    "adaptive_snn": "#0072B2",
    "fixed_snn": "#D55E00",
    "adaptive_ann": "#009E73",
    "ema": "#CC79A7",
    "persistence": "#666666",
}
LINESTYLES = {
    "adaptive_snn": "-",
    "fixed_snn": "--",
    "adaptive_ann": "-.",
    "ema": ":",
    "persistence": "-",
}
MARKERS = {
    "adaptive_snn": "o",
    "fixed_snn": "s",
    "adaptive_ann": "^",
    "ema": "D",
    "persistence": "x",
}
PROTOCOL_FIELDS = (
    "version",
    "frame_seconds",
    "sample_rate",
    "fit_frames",
    "calibration_frames",
    "probe_frames",
    "warmup_frames",
    "cooldown_frames",
    "block_frames",
    "ema_alpha",
    "seed",
)


def _summary_from_status(status: dict[str, Any]) -> dict[str, Any]:
    """Find a session summary without retaining unrelated status fields."""
    candidates = [status]
    for key in ("summary", "session", "result", "aggregate"):
        nested = status.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)
    for candidate in candidates:
        if isinstance(candidate.get("blocks"), list) and isinstance(
                candidate.get("checkpoints"), list):
            return candidate
    raise ValueError("status has no audio session blocks and checkpoints")


def _finite_nonnegative(value: Any) -> float | None:
    """Return a finite nonnegative number, or None for a missing value."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value < 0):
        return None
    return float(value)


def _arm_mse(block: dict[str, Any], arm: str) -> float | None:
    """Read an arm's MSE from a summary mapping."""
    arms = block.get("arms")
    if not isinstance(arms, dict):
        return None
    value = arms.get(arm)
    if not isinstance(value, dict):
        return None
    return _finite_nonnegative(value.get("mse"))


def _protocol_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep only numeric protocol settings that define plotted timing."""
    protocol = summary.get("protocol")
    if not isinstance(protocol, dict):
        return {}
    return {
        key: protocol[key] for key in PROTOCOL_FIELDS
        if key in protocol and isinstance(protocol[key], (int, float))
        and not isinstance(protocol[key], bool)
        and math.isfinite(float(protocol[key]))
    }


def build_plot_data(
    status: dict[str, Any], source_bytes: bytes
) -> dict[str, Any]:
    """Build plotted series and safe provenance from a session status."""
    summary = _summary_from_status(status)
    protocol = _protocol_metadata(summary)
    frame_seconds = float(protocol.get("frame_seconds", 0.2))
    block_frames = float(protocol.get("block_frames", 300))
    live_series: dict[str, list[dict[str, float]]] = {
        arm: [] for arm in ARMS
    }
    for block in summary["blocks"]:
        if not isinstance(block, dict):
            continue
        end_frame = _finite_nonnegative(block.get("frames"))
        if end_frame is None:
            continue
        midpoint = max(0.0, end_frame - block_frames / 2.0)
        minute = midpoint * frame_seconds / 60.0
        for arm in ARMS:
            mse = _arm_mse(block, arm)
            if mse is not None:
                live_series[arm].append({"minute": minute, "mse": mse})

    probe_series: dict[str, list[dict[str, float]]] = {
        arm: [] for arm in ARMS
    }
    checkpoints: list[dict[str, Any]] = []
    for checkpoint in summary["checkpoints"]:
        if not isinstance(checkpoint, dict):
            continue
        monitor_frame = _finite_nonnegative(checkpoint.get("monitor_frame"))
        probe = checkpoint.get("sealed_probe")
        if monitor_frame is None or not isinstance(probe, dict):
            continue
        checkpoints.append(checkpoint)
        minute = monitor_frame * frame_seconds / 60.0
        for arm in ARMS:
            mse = _arm_mse(probe, arm)
            if mse is not None:
                probe_series[arm].append({"minute": minute, "mse": mse})

    change_comparison = _change_comparison(
        summary, checkpoints, frame_seconds, block_frames)
    safe_timestamps = _safe_timestamps(status, summary)
    runtime = status.get("runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
    metadata = {
        "source_status_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "protocol_sha256": _safe_hash(runtime.get("protocol_sha256")),
        "git_head": _safe_hash(runtime.get("git_head")),
        "protocol": protocol,
        "arms": {
            arm: {"label": DISPLAY[arm], "color": COLORS[arm]}
            for arm in ARMS
        },
        "reference_hash": _safe_hash(summary.get("reference_hash")),
        "fixed_hash": _safe_hash(summary.get("fixed_hash")),
        "monitor_frames": _finite_nonnegative(
            summary.get("monitor_frames")),
        "timestamps": safe_timestamps,
    }
    return {
        "schema_version": 1,
        "units": {"x": "minutes of sample clock", "y": "MSE"},
        "panels": [
            {
                "key": "live",
                "label": "Live next-feature MSE",
                "x": "minute",
                "series": live_series,
            },
            {
                "key": "sealed_probe",
                "label": "Sealed 20-second probe MSE",
                "x": "minute",
                "series": probe_series,
            },
        ],
        "change_comparison": change_comparison,
        "metadata": metadata,
    }


def _change_comparison(
    summary: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    frame_seconds: float,
    block_frames: float,
) -> dict[str, Any]:
    """Calculate adaptive MSE reductions and retain each denominator."""
    live: dict[str, list[dict[str, float | None]]] = {
        arm: [] for arm in ADAPTIVE_ARMS
    }
    for block in summary["blocks"]:
        if not isinstance(block, dict):
            continue
        end_frame = _finite_nonnegative(block.get("frames"))
        fixed_mse = _arm_mse(block, "fixed_snn")
        if end_frame is None:
            continue
        minute = max(0.0, end_frame - block_frames / 2.0)
        minute *= frame_seconds / 60.0
        for arm in ADAPTIVE_ARMS:
            current_mse = _arm_mse(block, arm)
            live[arm].append({
                "minute": minute,
                "mse_reduction_percent": _percent_reduction(
                    current_mse, fixed_mse),
                "current_mse": current_mse,
                "denominator_mse": fixed_mse,
            })

    initial_probe = next(
        (checkpoint.get("sealed_probe") for checkpoint in checkpoints
         if checkpoint.get("monitor_frame") == 0
         and isinstance(checkpoint.get("sealed_probe"), dict)),
        {},
    )
    initial_denominators = {
        arm: _arm_mse(initial_probe, arm) for arm in ADAPTIVE_ARMS
    }
    sealed_probe: dict[str, list[dict[str, float | None]]] = {
        arm: [] for arm in ADAPTIVE_ARMS
    }
    for checkpoint in checkpoints:
        monitor_frame = _finite_nonnegative(checkpoint.get("monitor_frame"))
        probe = checkpoint.get("sealed_probe")
        if monitor_frame is None or not isinstance(probe, dict):
            continue
        minute = monitor_frame * frame_seconds / 60.0
        for arm in ADAPTIVE_ARMS:
            current_mse = _arm_mse(probe, arm)
            denominator = initial_denominators[arm]
            sealed_probe[arm].append({
                "minute": minute,
                "mse_reduction_percent": _percent_reduction(
                    current_mse, denominator),
                "current_mse": current_mse,
                "denominator_mse": denominator,
            })
    return {
        "positive_means_lower_mse": True,
        "live_denominator_rule": "fixed_snn_mse_per_block",
        "sealed_probe_denominator_rule": "same_arm_checkpoint_0_mse",
        "series": {"live": live, "sealed_probe": sealed_probe},
    }


def _percent_reduction(
    current_mse: float | None, denominator_mse: float | None
) -> float | None:
    """Return percent reduction, retaining negative relative changes."""
    if (current_mse is None or denominator_mse is None
            or denominator_mse == 0):
        return None
    return 100.0 * (denominator_mse - current_mse) / denominator_mse


def _safe_hash(value: Any) -> str | None:
    """Retain only a hexadecimal model fingerprint."""
    if not isinstance(value, str) or not value:
        return None
    if len(value) > 128 or any(char not in "0123456789abcdefABCDEF"
                               for char in value):
        return None
    return value


def _safe_timestamps(
    status: dict[str, Any], summary: dict[str, Any]
) -> dict[str, str]:
    """Keep only ISO-like timestamp fields, excluding source paths."""
    result: dict[str, str] = {}
    for source in (status, summary):
        for key in ("created_at", "started_at", "updated_at", "completed_at"):
            value = source.get(key)
            if (isinstance(value, str) and len(value) <= 64
                    and "T" in value):
                result[key] = value
    return result


def render_plot(data: dict[str, Any], output: Path,
                title: str | None = None) -> None:
    """Render two aligned MSE panels for all five comparison arms."""
    import matplotlib.pyplot as plt

    panels = data["panels"]
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    figure.suptitle(title or "Fan microphone prediction error", fontsize=15)
    for axis, panel in zip(axes, panels):
        series = panel["series"]
        for arm in ARMS:
            points = series[arm]
            axis.plot(
                [point["minute"] for point in points],
                [point["mse"] for point in points],
                color=COLORS[arm],
                linestyle=LINESTYLES[arm],
                marker=MARKERS[arm],
                markersize=4,
                linewidth=1.7,
                label=DISPLAY[arm],
            )
        axis.set_title(panel["label"], loc="left", fontsize=11)
        axis.set_ylabel("Mean squared error", fontsize=11)
        axis.grid(True, color="#D9DDE2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=10)
        axis.legend(frameon=False, ncol=3, fontsize=10, loc="best")
    axes[-1].set_xlabel("Minutes of sample clock", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_change_plot(data: dict[str, Any], output: Path) -> None:
    """Render live and sealed-probe relative MSE change for adaptive arms."""
    import matplotlib.pyplot as plt

    comparison = data["change_comparison"]["series"]
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    figure.suptitle("Adaptive model MSE reduction", fontsize=15)
    panels = (
        (axes[0], comparison["live"],
         "Live complete-block reduction vs fixed SNN"),
        (axes[1], comparison["sealed_probe"],
         "Sealed-probe reduction vs same arm at checkpoint 0"),
    )
    for axis, series, label in panels:
        for arm in ADAPTIVE_ARMS:
            points = series[arm]
            axis.plot(
                [point["minute"] for point in points],
                [point["mse_reduction_percent"] for point in points],
                color=COLORS[arm],
                linestyle=LINESTYLES[arm],
                marker=MARKERS[arm],
                markersize=4,
                linewidth=1.7,
                label=DISPLAY[arm],
            )
        axis.axhline(0.0, color="#444444", linewidth=0.9)
        axis.set_title(label, loc="left", fontsize=11)
        axis.set_ylabel("MSE reduction (%)", fontsize=11)
        axis.tick_params(axis="both", labelsize=10)
        axis.grid(True, color="#D9DDE2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=10, loc="best")
    axes[-1].set_xlabel("Minutes of sample clock", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Read aggregate status and write the figure plus plotted data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status_json", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--title")
    args = parser.parse_args()

    source_bytes = args.status_json.read_bytes()
    status = json.loads(source_bytes)
    if not isinstance(status, dict):
        raise ValueError("status JSON must contain an object")
    data = build_plot_data(status, source_bytes)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    render_plot(data, args.output_dir / "audio-progress.png", args.title)
    render_change_plot(data, args.output_dir / "learning-change.png")
    with (args.output_dir / "plotted-data.json").open(
            "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
