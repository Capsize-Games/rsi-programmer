#!/usr/bin/env python3
"""Summarize paired author/beam ablation evidence without rerunning it."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

BOOL_OUTCOMES = ("train_correct", "held_correct", "accepted")
WORK_FIELDS = (
    "signature_evaluations", "operand_signature_evaluations",
    "examined", "committed_nodes", "output_nodes", "wall_s",
)
HIT_FIELDS = ("train_hits", "held_hits")
ABLATIONS = ("no_exact_flag", "no_open_gap", "no_both", "no_neural")


def _family(row: dict[str, Any]) -> str:
    """Extract a stable family name from the repository-relative fixture."""
    return Path(str(row["fixture"])).stem.removeprefix("rsi_") \
        .removesuffix("_tasks")


def _float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("measurement contains a nonfinite metric")
    return result


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _better_counts(deltas: list[float]) -> dict[str, int]:
    """Count pairs where the comparison used less, equal, or more work."""
    return {
        "comparison_lower": sum(value < 0 for value in deltas),
        "equal": sum(value == 0 for value in deltas),
        "baseline_lower": sum(value > 0 for value in deltas),
    }


def _outcome_counts(
        pairs: list[tuple[dict[str, Any], dict[str, Any]]]
        ) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for metric in BOOL_OUTCOMES:
        comparison_wins = sum(
            bool(other[metric]) and not bool(base[metric])
            for base, other in pairs)
        baseline_wins = sum(
            bool(base[metric]) and not bool(other[metric])
            for base, other in pairs)
        out[metric] = {
            "comparison_wins": comparison_wins,
            "baseline_wins": baseline_wins,
            "ties": len(pairs) - comparison_wins - baseline_wins,
        }
    return out


def _numeric_summary(
        pairs: list[tuple[dict[str, Any], dict[str, Any]]]
        ) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for metric in (*HIT_FIELDS, *WORK_FIELDS):
        deltas = [_float(other[metric]) - _float(base[metric])
                  for base, other in pairs]
        out[metric] = {
            "baseline_mean": _mean([_float(base[metric])
                                    for base, _ in pairs]),
            "comparison_mean": _mean([_float(other[metric])
                                       for _, other in pairs]),
            "comparison_minus_baseline_mean": _mean(deltas),
            "lower_is_better_counts": _better_counts(deltas),
        }
    return out


def _pair_summary(
        pairs: list[tuple[dict[str, Any], dict[str, Any]]]
        ) -> dict[str, Any]:
    return {
        "pairs": len(pairs),
        "outcomes": _outcome_counts(pairs),
        "numeric": _numeric_summary(pairs),
    }


def _cell_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report raw successes and work by seed, family, path, and mode."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (int(row["seed"]), str(row["supervision"]),
               _family(row), str(row["path"]), str(row["variant"]))
        groups[key].append(row)
    result: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        train_count = sum(int(row["train_count"]) for row in group)
        held_count = sum(int(row["held_count"]) for row in group)
        item: dict[str, Any] = {
            "seed": key[0], "supervision": key[1], "family": key[2],
            "path": key[3], "variant": key[4], "tasks": len(group),
            "accepted": sum(bool(row["accepted"]) for row in group),
            "train_correct": sum(bool(row["train_correct"])
                                 for row in group),
            "held_correct": sum(bool(row["held_correct"])
                                for row in group),
            "train_hits": sum(int(row["train_hits"]) for row in group),
            "train_count": train_count,
            "held_hits": sum(int(row["held_hits"]) for row in group),
            "held_count": held_count,
            "train_hit_rate": (
                sum(int(row["train_hits"]) for row in group)
                / train_count if train_count else None),
            "held_hit_rate": (
                sum(int(row["held_hits"]) for row in group)
                / held_count if held_count else None),
            "budget_units": sorted({str(row["budget_unit"])
                                    for row in group}),
        }
        for metric in WORK_FIELDS:
            item[f"mean_{metric}"] = _mean(
                [_float(row[metric]) for row in group])
        result.append(item)
    return result


def _index_rows(
        rows: list[dict[str, Any]],
        key_fields: tuple[str, ...],
        dimension: str
        ) -> dict[tuple[Any, ...], dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        values: list[Any] = []
        for field in key_fields:
            if field == "family":
                values.append(_family(row))
            elif field == "seed":
                values.append(int(row[field]))
            else:
                values.append(str(row[field]))
        key = tuple(values)
        if key in indexed:
            raise ValueError(f"duplicate {dimension} result key: {key}")
        indexed[key] = row
    return indexed


def _paired_records(
        rows: list[dict[str, Any]],
        baseline_dimension: str,
        comparison_dimensions: list[str],
        key_fields: tuple[str, ...],
        comparisons: str
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Build task pairs and their grouped outcome/work summaries."""
    indexed = _index_rows(rows, (*key_fields, baseline_dimension), comparisons)
    pairs_by_group: dict[tuple[Any, ...],
                         list[tuple[dict[str, Any], dict[str, Any]]]] = \
        defaultdict(list)
    pair_records: list[dict[str, Any]] = []
    expected_pairs = 0
    missing_pairs = 0
    for key, baseline in indexed.items():
        base_value = str(baseline[baseline_dimension])
        if base_value not in comparison_dimensions[:1]:
            continue
        for other_value in comparison_dimensions[1:]:
            expected_pairs += 1
            other_key = (*key[:-1], other_value)
            other = indexed.get(other_key)
            if other is None:
                missing_pairs += 1
                continue
            pair_key = dict(zip(key_fields, key[:-1]))
            grouped = tuple(pair_key[field] for field in key_fields
                            if field != "task")
            group_key = (*grouped, other_value)
            pairs_by_group[group_key].append((baseline, other))
            item: dict[str, Any] = {
                **pair_key,
                "baseline": base_value,
                "comparison": other_value,
                "outcomes": {},
                "numeric_comparison_minus_baseline": {},
            }
            for metric in BOOL_OUTCOMES:
                base_success = bool(baseline[metric])
                other_success = bool(other[metric])
                item["outcomes"][metric] = {
                    "baseline": base_success,
                    "comparison": other_success,
                    "result": ("tie" if base_success == other_success
                               else "comparison-win" if other_success
                               else "baseline-win"),
                }
            for metric in (*HIT_FIELDS, *WORK_FIELDS):
                item["numeric_comparison_minus_baseline"][metric] = (
                    _float(other[metric]) - _float(baseline[metric]))
            pair_records.append(item)
    summaries: list[dict[str, Any]] = []
    for key, pairs in sorted(pairs_by_group.items()):
        grouped_fields = tuple(field for field in key_fields
                               if field != "task")
        summary: dict[str, Any] = dict(zip((*grouped_fields,
                                            "comparison"), key))
        summary.update(_pair_summary(pairs))
        summaries.append(summary)
    coverage = {
        "expected_pairs": expected_pairs,
        "matched_pairs": len(pair_records),
        "missing_pairs": missing_pairs,
    }
    return pair_records, summaries, coverage


def _weight_activity(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Identify exact/gap ablations that actually remove nonzero weights."""
    fields = list(data.get("config", {}).get("bonus_fields", []))
    records: list[dict[str, Any]] = []
    for training in data.get("training", []):
        weights = list(training.get("weights", []))
        values = dict(zip(fields, weights))
        checks = {}
        for channel in ("exact_flag", "open_gap"):
            value = values.get(channel)
            checks[channel] = {
                "weight": value,
                "nonzero": (value is not None and
                            not math.isclose(float(value), 0.0,
                                             abs_tol=1e-12)),
            }
        records.append({
            "seed": int(training["seed"]),
            "supervision": str(training["supervision"]),
            "channels": checks,
            "no_exact_flag_is_informative":
                checks["exact_flag"]["nonzero"],
            "no_open_gap_is_informative": checks["open_gap"]["nonzero"],
            "no_both_is_informative": (
                checks["exact_flag"]["nonzero"] or
                checks["open_gap"]["nonzero"]),
        })
    return records


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    """Check that the serialized acceptance bit matches its two inputs."""
    for row in rows:
        accepted = bool(row["train_correct"]) and bool(row["held_correct"])
        if bool(row["accepted"]) != accepted:
            raise ValueError("accepted bit disagrees with train/held checks")
        if int(row["train_hits"]) > int(row["train_count"]):
            raise ValueError("train hits exceed train row count")
        if int(row["held_hits"]) > int(row["held_count"]):
            raise ValueError("held hits exceed held row count")


def analyze(data: dict[str, Any], input_sha256: str) -> dict[str, Any]:
    rows = list(data.get("results", []))
    if not rows:
        raise ValueError("measurement has no result rows")
    _validate_rows(rows)
    expected_variants = list(data.get("config", {}).get("variants", []))
    if any(variant not in expected_variants for variant in ABLATIONS):
        raise ValueError("measurement lacks required ablation variants")

    ablation_records, ablation_rollups, ablation_coverage = _paired_records(
        rows, "variant", ["full", *ABLATIONS],
        ("seed", "supervision", "family", "path", "task"),
        "full_vs_ablations")
    (supervision_records, supervision_rollups,
     supervision_coverage) = _paired_records(
        rows, "supervision", ["canonical", "reachability"],
        ("seed", "family", "path", "task", "variant"),
        "reachability_vs_canonical")

    return {
        "schema": "author-ablation-analysis-v1",
        "measurement_schema": data.get("schema"),
        "measurement_sha256": input_sha256,
        "provenance": data.get("provenance", {}),
        "coverage": data.get("coverage", {}),
        "cell_summaries": _cell_summaries(rows),
        "full_vs_ablation": {
            "pair_direction": "comparison variant relative to full",
            "task_pairs": ablation_records,
            "rollups_by_seed_family_path_mode": ablation_rollups,
            "pair_coverage": ablation_coverage,
        },
        "reachability_vs_canonical": {
            "pair_direction": "reachability relative to canonical",
            "interpretation": (
                "end-to-end retraining and weight-refinement comparison; "
                "weights are fit separately by supervision mode"),
            "task_pairs": supervision_records,
            "rollups_by_seed_family_path_variant": supervision_rollups,
            "pair_coverage": supervision_coverage,
        },
        "fitted_channel_weights": _weight_activity(data),
        "limitations": [
            "Shadow held-out rows are seen-task evaluation; shadow train "
            "rows are also used for fitting and weight selection.",
            "Novel held-out rows are task-unseen evaluation; no novel rows "
            "are used for model fitting or weight selection.",
            "A zero fitted exact_flag or open_gap weight makes that channel's "
            "corresponding ablation a null contrast for that seed/mode.",
            "Author and beam use different budget units; cross-path work "
            "comparisons are descriptive and are not equal-budget claims.",
            "All candidates are signature-evaluated before ranking; canonical "
            "signature work can remain invariant across ranking ablations.",
            "The reachability/canonical comparison includes separate weight "
            "refinement per mode, so it is an end-to-end pipeline contrast.",
            "Three seeds and these fixture families do not establish broad "
            "generalization or a causal claim about candidate generation.",
            "Wall time is specific to the recorded runtime and host.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "measurement", type=Path,
        nargs="?",
        default=Path(__file__).parent / "measurement-portable" /
        "tree" / "artifacts" / "author-ablation-v1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.measurement.is_file():
        raise FileNotFoundError(
            f"measurement not available yet: {args.measurement}")
    raw = args.measurement.read_bytes()
    report = analyze(json.loads(raw), hashlib.sha256(raw).hexdigest())
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
