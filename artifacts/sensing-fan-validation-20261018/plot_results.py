"""Plot completed frozen sensing results without loading runtime code."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ARMS: tuple[str, ...] = (
    "plastic_snn", "readout_snn", "frozen_snn", "plastic_ann",
    "persistence", "ema", "nlms",
)
SEEDS: tuple[int, ...] = (0, 1, 2)
COLORS: dict[str, str] = {
    "plastic_snn": "#e87522",
    "plastic_ann": "#2878b5",
    "readout_snn": "#858585",
    "frozen_snn": "#858585",
    "persistence": "#858585",
    "ema": "#858585",
    "nlms": "#858585",
}
DISPLAY_NAMES: dict[str, str] = {
    "plastic_snn": "Plastic SNN",
    "readout_snn": "Readout SNN",
    "frozen_snn": "Frozen SNN",
    "plastic_ann": "Plastic ANN",
    "persistence": "Persistence",
    "ema": "EMA",
    "nlms": "NLMS",
}
METRICS: tuple[tuple[str, str, str, float | None], ...] = (
    ("Event recall", "recall", "percent", 80.0),
    ("Sham false alarms / minute",
     "sham_false_alarms_per_minute", "rate", 1.0),
    ("Test model + detector CPU seconds",
     "processing_cpu_seconds", "seconds", None),
)


def _sha256(path: Path) -> str:
    """Hash exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from one completed bundle file."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _finite_number(value: Any, label: str) -> float:
    """Return a finite numeric field, excluding booleans."""
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))):
        raise ValueError(f"invalid numeric field: {label}")
    return float(value)


def load_bundle(bundle: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate summary and manifest completeness and their binding hash."""
    summary_path = bundle / "summary.json"
    manifest_path = bundle / "manifest.json"
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("manifest has no output file hashes")
    expected_files = {
        "training.json", "calibration.json", "streams.jsonl.gz",
        "summary.json",
    }
    if set(files) != expected_files or any(
            not isinstance(value, str) or len(value) != 64
            for value in files.values()):
        raise ValueError("manifest output hashes are incomplete")
    if files.get("summary.json") != _sha256(summary_path):
        raise ValueError("summary hash does not match manifest")
    if manifest.get("source_unchanged") is not True:
        raise ValueError("manifest does not attest unchanged sources")
    if manifest.get("trained_sources_unchanged") is not True:
        raise ValueError("manifest does not attest unchanged trained sources")
    if manifest.get("stream_rows") != 924 or summary.get("stream_rows") != 924:
        raise ValueError("bundle does not contain the complete stream matrix")
    if len(summary.get("per_seed", [])) != len(ARMS) * len(SEEDS):
        raise ValueError("summary does not contain all arm/seed aggregates")
    if len(summary.get("pooled", [])) != len(ARMS):
        raise ValueError("summary does not contain all pooled aggregates")
    return summary, manifest


def _records(summary: dict[str, Any]
             ) -> tuple[dict[tuple[str, int], dict[str, Any]],
                        dict[str, dict[str, Any]]]:
    """Index and validate the published per-seed and pooled aggregates."""
    per_seed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in summary["per_seed"]:
        key = (row.get("arm"), row.get("seed"))
        if key in per_seed:
            raise ValueError(f"duplicate per-seed aggregate: {key}")
        per_seed[key] = row
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(per_seed) != expected:
        raise ValueError("per-seed aggregate keys are incomplete")
    pooled: dict[str, dict[str, Any]] = {}
    for row in summary["pooled"]:
        arm = row.get("arm")
        if arm in pooled:
            raise ValueError(f"duplicate pooled aggregate: {arm}")
        pooled[arm] = row
    if set(pooled) != set(ARMS):
        raise ValueError("pooled aggregate keys are incomplete")
    for key, row in per_seed.items():
        for _, metric_key, _, _ in METRICS:
            value = _finite_number(row.get(metric_key),
                                   f"{key}.{metric_key}")
            if value < 0 or (metric_key == "recall" and value > 1):
                raise ValueError(f"out-of-range aggregate: {key}.{metric_key}")
    for arm, row in pooled.items():
        for _, metric_key, _, _ in METRICS:
            value = _finite_number(row.get(metric_key),
                                   f"{arm}.{metric_key}")
            if value < 0 or (metric_key == "recall" and value > 1):
                raise ValueError(f"out-of-range aggregate: {arm}.{metric_key}")
    return per_seed, pooled


def _plot_value(metric_key: str, value: Any) -> float:
    """Convert recall to percent while keeping other metrics unchanged."""
    numeric = _finite_number(value, metric_key)
    return numeric * 100.0 if metric_key == "recall" else numeric


def _plotted_data(summary: dict[str, Any], manifest: dict[str, Any],
                  bundle: Path) -> dict[str, Any]:
    """Record exact values, observed ranges, and provenance hashes plotted."""
    per_seed, pooled = _records(summary)
    panels = []
    for title, metric_key, units, reference in METRICS:
        arms = []
        for arm in ARMS:
            raw_values = [
                _finite_number(per_seed[(arm, seed)][metric_key], metric_key)
                for seed in SEEDS
            ]
            values = [
                _plot_value(metric_key, per_seed[(arm, seed)][metric_key])
                for seed in SEEDS
            ]
            raw_pooled = _finite_number(pooled[arm][metric_key], metric_key)
            display_pooled = _plot_value(metric_key, raw_pooled)
            transform = "recall_percent" if metric_key == "recall" else "none"
            if metric_key == "processing_cpu_seconds":
                display_pooled /= len(SEEDS)
                transform = "divide_by_3_mean_per_seed"
            arms.append({
                "arm": arm,
                "display_name": DISPLAY_NAMES[arm],
                "seed_values_raw": dict(zip(SEEDS, raw_values)),
                "seed_values": dict(zip(SEEDS, values)),
                "median": sorted(values)[1],
                "observed_range": [min(values), max(values)],
                "pooled_raw": raw_pooled,
                "pooled_plot_value": display_pooled,
                "pooled_display_transform": transform,
            })
        panels.append({"title": title, "metric": metric_key,
                       "units": units, "reference_line": reference,
                       "arms": arms})
    return {
        "source_bundle": str(bundle.resolve()),
        "source_hashes": {
            "plot_script_sha256": _sha256(Path(__file__)),
            "summary_json": _sha256(bundle / "summary.json"),
            "manifest_json": _sha256(bundle / "manifest.json"),
            "files": manifest["files"],
            "source_files": manifest.get("source_files", {}),
            "protocol_sha256": manifest.get("protocol_sha256"),
            "dataset": manifest.get("dataset"),
        },
        "scope": (
            "Bars show pooled rates or mean CPU per seed; whiskers show "
            "the observed min-max range, not confidence intervals. CPU "
            "is pooled CPU divided by three for a mean per seed and "
            "covers test model plus detector time; "
            "provenance hashing is excluded."),
        "panels": panels,
    }


def render(data: dict[str, Any], output: Path) -> None:
    """Render readable aggregate panels to PNG and SVG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(3, 1, figsize=(10.0, 11.6))
    for axis, panel in zip(axes, data["panels"]):
        entries = panel["arms"]
        positions = list(range(len(entries)))
        pooled = [entry["pooled_plot_value"] for entry in entries]
        lows = [entry["pooled_plot_value"] - entry["observed_range"][0]
                for entry in entries]
        highs = [entry["observed_range"][1] - entry["pooled_plot_value"]
                 for entry in entries]
        colors = [COLORS[entry["arm"]] for entry in entries]
        axis.barh(positions, pooled, color=colors, height=0.62, zorder=2)
        axis.errorbar(pooled, positions, xerr=[lows, highs], fmt="none",
                      ecolor="#222222", elinewidth=1.2, capsize=3, zorder=3)
        axis.set_yticks(positions)
        axis.set_yticklabels([entry["display_name"] for entry in entries],
                            fontsize=11)
        axis.invert_yaxis()
        axis.set_title(panel["title"], fontsize=12, pad=10, loc="left")
        axis.grid(axis="x", color="#dddddd", linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)
        unit_labels = {
            "percent": "Recall (%)",
            "rate": "False alarms per minute",
            "seconds": "CPU seconds per seed",
        }
        axis.set_xlabel(unit_labels[panel["units"]], fontsize=11)
        axis.tick_params(axis="x", labelsize=10)
        reference = panel["reference_line"]
        if reference is not None:
            axis.axvline(reference, color="#555555", linestyle="--",
                         linewidth=1.0, zorder=1)
        upper = max(max(entry["observed_range"][1],
                        entry["pooled_plot_value"])
                    for entry in entries)
        bound = 100.0 if panel["units"] == "percent" else upper * 1.22
        axis.set_xlim(left=0.0, right=max(bound, 0.15))
        if reference is not None:
            axis.text(reference, -0.65, f"target {reference:g}",
                      ha="center", va="bottom", fontsize=10)
        for y, entry in zip(positions, entries):
            digits = 3 if panel["units"] == "seconds" else 2
            value = entry["pooled_plot_value"]
            label_x = max(entry["observed_range"][1], value)
            axis.annotate(f"{value:.{digits}f}", (label_x, y),
                          xytext=(6, 0), textcoords="offset points",
                          va="center", ha="left", fontsize=11)
    handles = [
        Line2D([0], [0], color="#222222", marker="|", markersize=10,
               label="3-seed range; not a confidence interval"),
    ]
    figure.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=11)
    footer = (
        "924 stitched streams from one fan; correlated repetitions.\n"
        "CPU: model + detector, mean per seed; excludes provenance hashing."
    )
    figure.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=11)
    figure.subplots_adjust(left=0.20, right=0.97, top=0.94, bottom=0.09,
                           hspace=0.57)
    figure.savefig(output / "sensing-results.png", dpi=180,
                   bbox_inches="tight")
    figure.savefig(output / "sensing-results.svg", bbox_inches="tight")
    plt.close(figure)
    svg_path = output / "sensing-results.svg"
    svg_path.write_text(svg_path.read_text(encoding="utf-8").rstrip() + "\n",
                        encoding="utf-8")


def main() -> None:
    """Load a completed bundle and write a new plot directory once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.bundle.is_dir():
        raise ValueError("--bundle must name a completed bundle directory")
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing output directory")
    summary, manifest = load_bundle(args.bundle)
    data = _plotted_data(summary, manifest, args.bundle)
    args.output.mkdir(parents=True)
    try:
        render(data, args.output)
        (args.output / "plotted-aggregates.json").write_text(
            json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
    except BaseException:
        for path in args.output.iterdir():
            path.unlink()
        args.output.rmdir()
        raise


if __name__ == "__main__":
    main()
