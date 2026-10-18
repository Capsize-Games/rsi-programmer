#!/usr/bin/env python3
"""Plot completed online plastic-encoder benchmark summary data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ARMS = (
    "persistent_snn",
    "frozen_snn",
    "episode_reset_snn",
    "online_nonspiking",
    "grammar_streaming",
    "plastic_snn",
    "reset_plastic_snn",
    "plastic_nonspiking",
)
LABELS = {
    "persistent_snn": "Readout SNN",
    "frozen_snn": "Frozen SNN",
    "episode_reset_snn": "Reset readout SNN",
    "online_nonspiking": "Readout ANN",
    "grammar_streaming": "Streaming",
    "plastic_snn": "Plastic SNN",
    "reset_plastic_snn": "Reset plastic SNN",
    "plastic_nonspiking": "Plastic ANN",
}
NEUTRAL_COLOR = "#84909b"
PLASTIC_COLOR = "#d18425"
ANN_COLOR = "#3977a8"
CONTROL_COLOR = "#9ca3aa"
EXPECTED_ATTEMPTS = 864
EXPECTED_ORDERS = 6
EXPECTED_SEEDS = 3
EXPECTED_TASKS_PER_ORDER = 144


def _number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"summary metric {key!r} is not numeric")
    return float(value)


def _pooled_rows(
    summary: dict[str, Any],
) -> dict[int, dict[str, dict[str, float]]]:
    orders = summary.get("per_order")
    if not isinstance(orders, list) or len(orders) != EXPECTED_ORDERS:
        raise ValueError("summary must contain exactly six per_order entries")
    totals = {
        budget: {
            arm: {
                key: 0.0
                for key in (
                    "attempts",
                    "accepted",
                    "first_lane_correct",
                    "candidate_evaluations",
                    "search_only_cpu_seconds",
                    "retention_probe_cpu_seconds",
                    "total_cpu_seconds",
                )
            }
            for arm in ARMS
        }
        for budget in (144, 432)
    }
    seen: set[tuple[int, str, int]] = set()
    for order_index, order in enumerate(orders):
        if order.get("matrix_complete") is not True:
            raise ValueError(f"order {order_index} matrix is incomplete")
        entries = order.get("pooled")
        if isinstance(entries, dict):
            entries = list(entries.values())
        if not isinstance(entries, list):
            raise ValueError(f"order {order_index} pooled metrics are missing")
        for row in entries:
            if not isinstance(row, dict):
                raise ValueError("pooled metric row must be an object")
            budget = row.get("budget")
            arm = row.get("arm")
            if budget not in (144, 432) or arm not in ARMS:
                raise ValueError(f"unexpected pooled row key: {budget}/{arm}")
            key = (int(budget), str(arm), order_index)
            if key in seen:
                raise ValueError(f"duplicate pooled row: {key}")
            seen.add(key)
            attempts = row.get("tasks", row.get("attempts"))
            if attempts != EXPECTED_ATTEMPTS // EXPECTED_ORDERS:
                raise ValueError(
                    f"{budget}/{arm} must contain 144 attempts per order"
                )
            seeds = row.get("seed_metrics", {})
            if (set(seeds) != {"0", "1", "2"}
                    or any(seed.get("tasks") != 48
                           for seed in seeds.values())):
                raise ValueError("expected three seeds of 48 tasks each")
            totals[int(budget)][str(arm)]["attempts"] += float(attempts)
            for metric in (
                "accepted",
                "first_lane_correct",
                "candidate_evaluations",
                "search_only_cpu_seconds",
                "retention_probe_cpu_seconds",
                "total_cpu_seconds",
            ):
                totals[int(budget)][str(arm)][metric] += _number(row, metric)
    expected = {
        (budget, arm, order_index)
        for budget in (144, 432)
        for arm in ARMS
        for order_index in range(EXPECTED_ORDERS)
    }
    if seen != expected:
        raise ValueError("summary is missing per-order budget/arm rows")
    for budget, arms in totals.items():
        for arm, row in arms.items():
            if row["attempts"] != EXPECTED_ATTEMPTS:
                raise ValueError(f"{budget}/{arm} does not total 864 attempts")
    return totals


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise ValueError("ratio denominator must be positive")
    return numerator / denominator


def _metrics(
    totals: dict[int, dict[str, dict[str, float]]],
    source_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_summary": str(source_path),
        "source_summary_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "protocol_sha256": manifest.get("protocol_sha256"),
        "git_head": manifest.get("git_head"),
        "scope": {
            "known_functions": 48,
            "orders": EXPECTED_ORDERS,
            "model_seeds": EXPECTED_SEEDS,
            "attempts_per_arm_budget": EXPECTED_ATTEMPTS,
            "independent_task_claim": False,
            "retention_claim": False,
            "capacity_claim": False,
            "cpu_panel_excludes": (
                "retention probes; includes search diagnostics"
            ),
        },
        "budgets": {},
    }
    if not result["protocol_sha256"] or not result["git_head"]:
        raise ValueError("manifest must pin protocol_sha256 and git_head")
    for budget, arms in sorted(totals.items()):
        streaming = arms["grammar_streaming"]
        plastic = arms["plastic_snn"]
        comparisons: dict[str, Any] = {}
        for control in ("persistent_snn", "plastic_nonspiking"):
            baseline = arms[control]
            comparisons[control] = {
                "accepted_delta_count": (
                    plastic["accepted"] - baseline["accepted"]
                ),
                "accepted_delta_percentage_points": 100.0 * (
                    plastic["accepted"] - baseline["accepted"]
                ) / EXPECTED_ATTEMPTS,
                "first_lane_delta_count": (
                    plastic["first_lane_correct"]
                    - baseline["first_lane_correct"]
                ),
                "first_lane_delta_percentage_points": 100.0 * (
                    plastic["first_lane_correct"]
                    - baseline["first_lane_correct"]
                ) / EXPECTED_ATTEMPTS,
                "candidate_evaluation_delta": (
                    plastic["candidate_evaluations"]
                    - baseline["candidate_evaluations"]
                ),
                "search_only_cpu_delta_seconds": (
                    plastic["search_only_cpu_seconds"]
                    - baseline["search_only_cpu_seconds"]
                ),
                "search_only_cpu_ratio": _ratio(
                    plastic["search_only_cpu_seconds"],
                    baseline["search_only_cpu_seconds"],
                ),
                "total_cpu_delta_seconds": (
                    plastic["total_cpu_seconds"]
                    - baseline["total_cpu_seconds"]
                ),
                "total_cpu_ratio": _ratio(
                    plastic["total_cpu_seconds"], baseline["total_cpu_seconds"]
                ),
            }
        result["budgets"][str(budget)] = {
            "attempts_per_arm": EXPECTED_ATTEMPTS,
            "arms": {
                arm: {
                    **row,
                    "accepted_rate": row["accepted"] / EXPECTED_ATTEMPTS,
                    "first_lane_rate": (
                        row["first_lane_correct"] / EXPECTED_ATTEMPTS
                    ),
                    "search_only_cpu_ratio_vs_streaming": _ratio(
                        row["search_only_cpu_seconds"],
                        streaming["search_only_cpu_seconds"],
                    ),
                    "candidate_evaluation_ratio_vs_streaming": _ratio(
                        row["candidate_evaluations"],
                        streaming["candidate_evaluations"],
                    ),
                    "total_cpu_ratio_vs_streaming": _ratio(
                        row["total_cpu_seconds"],
                        streaming["total_cpu_seconds"],
                    ),
                }
                for arm, row in arms.items()
            },
            "plastic_snn_comparisons": comparisons,
        }
    return result


def _bar_color(arm: str) -> str:
    if arm == "plastic_snn":
        return PLASTIC_COLOR
    if arm == "plastic_nonspiking":
        return ANN_COLOR
    return CONTROL_COLOR


def _plot(metrics: dict[str, Any], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.8))
    figure.suptitle(
        "Online plastic-encoder validation\n"
        "48 known functions × 6 orders × 3 seeds; 864 repeated attempts "
        "per arm\nPlastic methods update encoder and readout; "
        "ANN is continuous",
        fontsize=12,
    )
    panels = (
        (144, "accepted", "Budget 144: accepted searches", "percent"),
        (144, "first_lane", "Budget 144: first-lane accuracy", "percent"),
        (144, "search_cpu", "Primary search-only CPU vs streaming", "ratio"),
        (
            432,
            "candidate_ratio",
            "Full-grammar evaluations vs streaming",
            "ratio",
        ),
    )
    labels = [LABELS[arm] for arm in ARMS]
    y_positions = list(range(len(ARMS)))
    for axis, (budget, measure, title, fmt) in zip(axes.flat, panels):
        rows = metrics["budgets"][str(budget)]["arms"]
        if measure == "accepted":
            values = [100.0 * rows[arm]["accepted_rate"] for arm in ARMS]
            reference = 100.0 * rows["persistent_snn"]["accepted_rate"]
        elif measure == "first_lane":
            values = [100.0 * rows[arm]["first_lane_rate"] for arm in ARMS]
            reference = 100.0 * rows["persistent_snn"]["first_lane_rate"]
        elif measure == "search_cpu":
            values = [
                rows[arm]["search_only_cpu_ratio_vs_streaming"]
                for arm in ARMS
            ]
            reference = 1.0
        else:
            values = [
                rows[arm]["candidate_evaluation_ratio_vs_streaming"]
                for arm in ARMS
            ]
            reference = 1.0
        bars = axis.barh(
            y_positions,
            values,
            color=[_bar_color(arm) for arm in ARMS],
            edgecolor="#3d454c",
            linewidth=0.45,
        )
        axis.set_yticks(y_positions, labels=labels, fontsize=10)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=12)
        axis.axvline(reference, color="#333333", linewidth=1.0, linestyle="--")
        axis.set_xlim(left=0)
        axis.grid(axis="x", color="#d8dde1", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.set_xlabel("Percent of attempts" if fmt == "percent" else "Ratio")
        max_value = max(values) if values else 0.0
        padding = 1.45 if fmt == "percent" else 1.24
        axis.set_xlim(0, max(max_value * padding, reference * 1.18, 0.05))
        for arm, bar, value in zip(ARMS, bars, values):
            if fmt == "percent":
                key = ("accepted" if measure == "accepted"
                       else "first_lane_correct")
                text = f"{int(rows[arm][key])}/864 ({value:.1f}%)"
            else:
                text = f"{value:.2f}×"
            label_value = (max(value, reference)
                           if measure == "candidate_ratio" else value)
            axis.text(
                label_value + max(max_value, 1.0) * 0.018,
                bar.get_y() + bar.get_height() / 2,
                text,
                va="center",
                fontsize=8.2,
            )
    legend = [
        Patch(facecolor=CONTROL_COLOR, label="Other controls"),
        Patch(facecolor=PLASTIC_COLOR, label="Plastic-encoder SNN"),
        Patch(facecolor=ANN_COLOR, label="Plastic nonspiking control"),
        Line2D(
            [0], [0], color="#333333", linestyle="--",
            label="Dashed: readout SNN (top), streaming (bottom)"
        ),
    ]
    figure.legend(
        handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.035),
        ncol=2, frameon=False, fontsize=9,
    )
    figure.text(
        0.5,
        0.008,
        "The same 48 functions repeat across orders and seeds; attempts "
        "are not independent tasks.\nCPU excludes retention probes and "
        "includes search diagnostics; it does not measure device energy.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.11, 1, 1))
    figure.savefig(output_path.with_suffix(".png"), dpi=110)
    svg_path = output_path.with_suffix(".svg")
    figure.savefig(svg_path)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in
                  svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary_path = args.bundle / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(
            f"completed bundle summary is missing: {summary_path}"
        )
    output_base = args.output
    if output_base.resolve() == args.bundle.resolve():
        raise SystemExit(
            "output path must not be the completed bundle directory"
        )
    output_base.mkdir(parents=True, exist_ok=True)
    output_path = output_base / "online-plastic-encoder"
    targets = [
        output_path.with_suffix(".png"),
        output_path.with_suffix(".svg"),
        output_path.with_suffix(".json"),
    ]
    if any(target.exists() for target in targets):
        raise SystemExit("refusing to overwrite an existing plot output")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_path = args.bundle / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"completed bundle manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregate = _pooled_rows(summary)
    metrics = _metrics(aggregate, summary_path, manifest)
    _plot(metrics, output_path)
    metrics["outputs"] = {
        target.suffix[1:]: {
            "path": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
        for target in targets[:2]
    }
    targets[2].write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
