#!/usr/bin/env python3
"""Plot cost curves from the audited complete-shortlist bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = (
    ROOT / "artifacts/benchmark-complete-shortlist-v1-verified-20261018"
)
PROTOCOL_PATH = ROOT / "fixtures/benchmark_complete_shortlist_v1.json"
PROTOCOL_SHA256 = (
    "189fa147a11040c6a05ef173522a6146285de616996724adff49d17b5f66be0a"
)
REQUIRED_CHECKS = (
    "grid_valid", "memory_valid", "repeats_behavior_identical",
    "model_states_frozen", "no_row_errors", "source_frozen",
    "environment_frozen", "shortlist_peak_valid", "root_cap_valid",
)
CONTROL_LABELS = {
    "streaming": "streaming",
    "cheap_only": "cheap-only",
    "heuristic": "heuristic",
    "trained_snn": "trained SNN",
    "mlp": "MLP",
    "recurrent": "recurrent",
}
CONTROL_COLORS = {
    "streaming": "#111111",
    "cheap_only": "#9a6a24",
    "heuristic": "#3b6fb6",
    "trained_snn": "#bb4b3f",
    "mlp": "#3b8c68",
    "recurrent": "#8b65a5",
}
CAP_MARKERS = ("o", "s", "^", "D", "P")


def sha256_file(path: Path) -> str:
    """Hash file bytes for bundle integrity checks."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def expected_configurations(
    protocol: dict[str, Any],
) -> dict[tuple[str, str], list[int]]:
    """Build the ten declared engine/control budget series."""
    evaluation = protocol["evaluation"]
    result: dict[tuple[str, str], list[int]] = {
        ("streaming", "streaming"): list(
            evaluation["streaming_budgets"]
        ),
        ("cheap_only", "cheap_only"): list(
            evaluation["ranked_budgets"]
        ),
    }
    engines = evaluation["ranked_engines"]
    if tuple(engines) != ("full_cached", "complete_shortlist"):
        raise ValueError("protocol engines differ from the plot schema")
    for control in evaluation["ranked_controls"]:
        for engine in engines:
            result[(str(engine), str(control))] = list(
                evaluation["ranked_budgets"]
            )
    if len(result) != 10:
        raise ValueError("protocol must declare ten plot configurations")
    if tuple(evaluation["streaming_budgets"]) != (
        2048, 8192, 32768, 131072, 524288
    ):
        raise ValueError(
            "streaming cap points differ from the frozen protocol"
        )
    if tuple(evaluation["ranked_budgets"]) != (
        2048, 8192, 32768, 131072
    ):
        raise ValueError("ranked cap points differ from the frozen protocol")
    return result


def validate_bundle(
    bundle: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reject incomplete or invalid results before producing a figure."""
    required_files = (
        "manifest.json", "training.json", "aggregate.json", "raw.jsonl",
        "memory.jsonl",
    )
    missing = [name for name in required_files
               if not (bundle / name).is_file()]
    if missing:
        raise ValueError("bundle is incomplete; missing " + ", ".join(missing))
    manifest = read_json(bundle / "manifest.json")
    training = read_json(bundle / "training.json")
    aggregate = read_json(bundle / "aggregate.json")
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("protocol_id") != "benchmark-complete-shortlist-v1":
        raise ValueError("unexpected complete-shortlist protocol")
    if sha256_file(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError("protocol fixture differs from the frozen hash")
    selection = protocol["selection_evidence"]
    evidence_paths = {
        selection["path"]: selection["sha256"],
        selection["execution_path"]: selection["execution_sha256"],
    }
    for relative, expected_hash in evidence_paths.items():
        if sha256_file(ROOT / relative) != expected_hash:
            raise ValueError("pinned selection evidence hash mismatch")
    selection_dir = (ROOT / selection["path"]).parent
    for name, expected_hash in selection["files_sha256"].items():
        if sha256_file(selection_dir / name) != expected_hash:
            raise ValueError("pinned selection output hash mismatch: " + name)
    if manifest.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("bundle protocol hash does not match the fixture")
    if manifest.get("infrastructure_valid") is not True:
        raise ValueError("bundle manifest is not infrastructure-valid")
    failed = [name for name in REQUIRED_CHECKS
              if manifest.get(name) is not True]
    if failed:
        raise ValueError("bundle failed checks: " + ", ".join(failed))
    if training.get("status") != "complete":
        raise ValueError("training record is incomplete")
    if aggregate.get("infrastructure_valid") is not True:
        raise ValueError("aggregate is not infrastructure-valid")
    counts = (
        ("row_count", "expected_row_count", 11808),
        ("memory_row_count", "expected_memory_row_count", 960),
    )
    for actual_key, expected_key, expected in counts:
        if manifest.get(actual_key) != expected:
            raise ValueError(f"bundle {actual_key} is incomplete")
        if manifest.get(expected_key) != expected:
            raise ValueError(f"protocol {expected_key} differs")
    file_hashes = manifest.get("files_sha256", {})
    for name in required_files[1:]:
        expected_hash = file_hashes.get(name)
        if not isinstance(expected_hash, str):
            raise ValueError(f"manifest has no hash for {name}")
        if sha256_file(bundle / name) != expected_hash:
            raise ValueError(f"bundle file hash mismatch: {name}")
    return manifest, training, {"aggregate": aggregate, "protocol": protocol}


def cost_points(
    aggregate: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Validate declared points and join audited N=48 amortized costs."""
    expected = expected_configurations(protocol)
    attempt_count = (
        protocol["evaluation"]["task_count"]
        * len(protocol["evaluation"]["seeds"])
        * protocol["evaluation"]["repeats"]
    )
    actual: dict[tuple[str, str, int], dict[str, Any]] = {}
    points = [point for point in aggregate.get("points", [])
              if point.get("repeat") is None]
    for point in points:
        key = (
            str(point.get("engine")), str(point.get("control")),
            int(point.get("budget", -1)),
        )
        if key in actual:
            raise ValueError(f"duplicate aggregate point: {key}")
        if key[:2] not in expected or key[2] not in expected[key[:2]]:
            raise ValueError(f"unexpected aggregate point: {key}")
        if point.get("attempts") != attempt_count:
            raise ValueError(f"incomplete attempt count for {key}")
        success = point.get("success_fraction")
        cpu = point.get("attempt_cpu_ns_mean")
        if (not isinstance(success, (int, float))
                or not 0.0 <= float(success) <= 1.0):
            raise ValueError(f"invalid accepted fraction for {key}")
        if (not isinstance(cpu, (int, float)) or not math.isfinite(cpu)
                or cpu <= 0.0):
            raise ValueError(f"invalid mean attempt CPU for {key}")
        actual[key] = point

    required_keys = {
        (engine, control, budget)
        for (engine, control), budgets in expected.items()
        for budget in budgets
    }
    if set(actual) != required_keys:
        absent = sorted(required_keys - set(actual))
        extra = sorted(set(actual) - required_keys)
        raise ValueError(
            "aggregate cap grid is incomplete; "
            f"missing={absent}, extra={extra}"
        )

    amortized: dict[tuple[str, str, int], float] = {}
    for point in aggregate.get("amortization", []):
        if point.get("independent_task_count") != 48:
            continue
        key = (
            str(point.get("engine")), str(point.get("control")),
            int(point.get("budget", -1)),
        )
        value = point.get("amortized_cpu_ns_per_task")
        if (key in amortized or key not in required_keys
                or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0.0):
            raise ValueError(f"invalid N=48 amortization point: {key}")
        expected_selection_cpu = (
            protocol["selection_evidence"]["execution_charge_cpu_ns"]
            if key[0] in {"complete_shortlist", "cheap_only"} else 0
        )
        expected_selection_wall = (
            protocol["selection_evidence"]["execution_charge_wall_ns"]
            if key[0] in {"complete_shortlist", "cheap_only"} else 0
        )
        if (point.get("selection_cpu_ns") != expected_selection_cpu
                or point.get("selection_wall_ns") != expected_selection_wall):
            raise ValueError(f"selection charge mismatch for {key}")
        amortized[key] = float(value)
    if set(amortized) != required_keys:
        raise ValueError("aggregate lacks complete N=48 amortized costs")
    return {
        config: [
            {
                "budget": budget,
                "success_fraction": float(
                    actual[(config[0], config[1], budget)][
                        "success_fraction"
                    ]
                ),
                "attempt_cpu_ms": float(
                    actual[(config[0], config[1], budget)][
                        "attempt_cpu_ns_mean"
                    ]
                ) / 1_000_000.0,
                "amortized_cpu_ms": (
                    amortized[(config[0], config[1], budget)] / 1_000_000.0
                ),
            }
            for budget in budgets
        ]
        for config, budgets in expected.items()
    }


def plot_costs(
    bundle: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Render the validated measured points to PNG and SVG files."""
    manifest, _training, inputs = validate_bundle(bundle)
    aggregate = inputs["aggregate"]
    protocol = inputs["protocol"]
    curves = cost_points(aggregate, protocol)
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to render the figure"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    caps = sorted({point["budget"] for curve in curves.values()
                   for point in curve})
    if len(caps) != len(CAP_MARKERS):
        raise ValueError("cap markers do not match the frozen cap grid")
    marker_for = {cap: CAP_MARKERS[index]
                  for index, cap in enumerate(caps)}
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.78, top=0.87, bottom=0.20,
                        wspace=0.18)
    axis_values = (
        (axes[0], "attempt_cpu_ms", "Mean attempt CPU (ms)"),
        (axes[1], "amortized_cpu_ms", "N=48 training-amortized CPU (ms)"),
    )
    styles = {
        "streaming": "-", "cheap_only": "-",
        "full_cached": "--", "complete_shortlist": "-",
    }
    for (engine, control), curve in curves.items():
        color = CONTROL_COLORS.get(control)
        if color is None:
            raise ValueError(f"unknown control in protocol: {control}")
        label = f"{engine.replace('_', ' ')} | {CONTROL_LABELS[control]}"
        x_values = [point["success_fraction"] * 100.0 for point in curve]
        for axis, value_key, ylabel in axis_values:
            y_values = [point[value_key] for point in curve]
            axis.plot(x_values, y_values, color=color,
                      linestyle=styles[engine], linewidth=1.5, zorder=2)
            for point, x_value, y_value in zip(curve, x_values, y_values):
                marker = marker_for[point["budget"]]
                face = "none" if engine == "full_cached" else color
                axis.scatter(
                    [x_value], [y_value], marker=marker, s=40,
                    facecolors=face, edgecolors=color, linewidths=1.2,
                    zorder=3,
                )
            axis.set_xlabel("Accepted attempts (%)")
            axis.set_ylabel(ylabel)
            axis.set_xlim(0, 100)
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.25)
            axis.set_title(ylabel)

    config_handles = []
    for engine, control in curves:
        config_handles.append(Line2D(
            [0], [0], color=CONTROL_COLORS[control],
            linestyle=styles[engine], linewidth=1.7,
            label=(f"{engine.replace('_', ' ')} | "
                   f"{CONTROL_LABELS[control]}"),
        ))
    config_legend = fig.legend(
        handles=config_handles, loc="center left",
        bbox_to_anchor=(0.79, 0.58), frameon=False,
        title="Engine | control", fontsize=8,
    )
    cap_handles = [Line2D(
        [0], [0], color="#555555", marker=marker_for[cap], linestyle="None",
        markerfacecolor="white", markeredgecolor="#555555",
        label=f"{cap:,}",
    ) for cap in caps]
    fig.legend(
        handles=cap_handles, loc="center left", bbox_to_anchor=(0.79, 0.20),
        frameon=False, title="Candidate signature cap", fontsize=8,
    )
    fig.add_artist(config_legend)
    fig.suptitle("Complete-fallback shortlist cost and coverage", fontsize=14)
    fig.text(
        0.09, 0.06,
        "Markers show declared cap points. Lines connect measured points "
        "only; no intermediate values are estimated. N=48 costs use the "
        "aggregate's production-fit and pinned selection charges.",
        ha="left", va="bottom", fontsize=8,
    )
    fig.text(
        0.09, 0.025,
        "Audited bundle: " + bundle.name
        + " | protocol SHA256: " + str(manifest["protocol_sha256"]),
        ha="left", va="bottom", fontsize=7,
    )
    png_path = output_dir / "complete-shortlist-cost-curves.png"
    svg_path = output_dir / "complete-shortlist-cost-curves.svg"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def main(argv: list[str] | None = None) -> int:
    """Validate the audited bundle and render both standalone formats."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args(argv)
    try:
        png_path, svg_path = plot_costs(
            args.bundle.resolve(), args.output_dir.resolve()
        )
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"plot generation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"png": str(png_path), "svg": str(svg_path)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
