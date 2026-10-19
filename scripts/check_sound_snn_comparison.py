"""Validate frozen aggregate evidence for the sound SNN comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT / "artifacts" / "sound-snn-validation-20261019"
    / "frozen-comparison-v1" / "sound_comparison_report.json"
)
SEEDS = (0, 1, 2)
PROTOCOL_SHA256 = (
    "1f0be06f30f3010b1cd75d10fab4c8c676ab137c56f1e2dd046cfe4111839b83"
)
LABELS = ("bell", "knock", "whistle", "chime")
STAGES = (
    ("initial_unknown", 0, (), LABELS),
    ("three_labels", 36, LABELS[:3], ("chime",)),
    ("four_labels", 48, LABELS, ()),
    ("after_reload", 48, LABELS, ()),
)
SUMMARY_EXAMPLES = {
    "initial_unknown": (0, 0),
    "three_labels": (9, 36),
    "four_labels": (12, 84),
    "after_reload": (12, 132),
}
OUTCOMES = ("correct", "wrong", "rejected")
BACKENDS = ("snn", "raw_neural", "matcher")
GUARD_OUTCOMES = (
    "guard_rejected_raw_correct",
    "guard_rejected_raw_wrong",
    "guard_rejected_raw_unclassified",
    "guard_accepted_raw_correct",
    "guard_accepted_raw_wrong",
    "guard_accepted_raw_unclassified",
)
TIMING_FIELDS = (
    "snn_inference_wall_seconds",
    "matcher_inference_wall_seconds",
    "paired_record_wall_seconds",
)
CONTROL_NAMES = ("silence", "noise", "far_tone_low", "far_tone_high")


class Checks:
    """Count numeric comparisons while raising on the first mismatch."""

    def __init__(self) -> None:
        self.numeric_assertions = 0

    def equal_number(
        self, actual: Any, expected: int | float, path: str
    ) -> None:
        self.numeric_assertions += 1
        if (isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isfinite(float(actual))):
            raise ValueError(f"{path}: expected a number")
        if isinstance(expected, float):
            equal = math.isclose(
                float(actual), expected, rel_tol=1e-9, abs_tol=1e-9
            )
        else:
            equal = actual == expected
        if not equal:
            raise ValueError(f"{path}: expected {expected}, got {actual}")

    def at_least(self, actual: Any, expected: int, path: str) -> None:
        self.numeric_assertions += 1
        if (isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isfinite(float(actual))
                or actual < expected):
            raise ValueError(f"{path}: expected at least {expected}")

    @staticmethod
    def truth(actual: Any, path: str) -> None:
        if actual is not True:
            raise ValueError(f"{path}: expected true")

    @staticmethod
    def same(actual: Any, expected: Any, path: str) -> None:
        if actual != expected:
            raise ValueError(f"{path}: values differ")


def _outcome_total(counts: dict[str, Any]) -> int:
    return sum(counts[key] for key in OUTCOMES)


def _sum_numbers(rows: list[dict[str, Any]], key: str) -> int | float:
    return sum(row[key] for row in rows)


def _pooled_from_seeds(
    per_seed: dict[str, dict[str, Any]], stage_id: str
) -> dict[str, Any]:
    stages = [
        next(stage for stage in run["stages"] if stage["id"] == stage_id)
        for run in per_seed.values()
    ]
    pooled: dict[str, Any] = {
        "id": stage_id,
        "model_runs": len(stages),
        "queries": _sum_numbers(stages, "queries"),
        "snn": {},
        "raw_neural": {},
        "matcher": {},
        "support_guard_effect": {},
        "per_class": {},
        "unknown_family_controls": {},
        "controls": {},
    }
    for backend in BACKENDS:
        pooled[backend] = {
            result: sum(stage[backend][result] for stage in stages)
            for result in OUTCOMES
        }
    pooled["support_guard_effect"] = {
        result: sum(stage["support_guard_effect"][result]
                    for stage in stages)
        for result in GUARD_OUTCOMES
    }
    for field in TIMING_FIELDS:
        pooled[field] = _sum_numbers(stages, field)
    labels = set().union(*(set(stage["per_class"]) for stage in stages))
    for label in labels:
        rows = [stage["per_class"][label] for stage in stages
                if label in stage["per_class"]]
        merged: dict[str, Any] = {
            "queries": _sum_numbers(rows, "queries"),
            "support_guard_effect": {},
        }
        for backend in BACKENDS:
            merged[backend] = {
                outcome: sum(row[backend][outcome] for row in rows)
                for outcome in OUTCOMES
            }
        merged["support_guard_effect"] = {
            result: sum(row["support_guard_effect"][result]
                        for row in rows)
            for result in GUARD_OUTCOMES
        }
        pooled["per_class"][label] = merged
    families = set().union(*(
        set(stage["unknown_family_controls"]) for stage in stages
    ))
    for family in families:
        rows = [stage["unknown_family_controls"][family]
                for stage in stages
                if family in stage["unknown_family_controls"]]
        pooled["unknown_family_controls"][family] = {
            key: _sum_numbers(rows, key)
            for key in (
                "queries", "snn_false_accepts",
                "raw_neural_false_accepts", "matcher_false_accepts",
            )
        }
    control_names = set().union(
        *(set(stage["controls"]) for stage in stages)
    )
    for name in control_names:
        rows = [stage["controls"][name] for stage in stages
                if name in stage["controls"]]
        pooled["controls"][name] = {
            "snn_false_accepts": sum(
                int(row["snn_false_accept"]) for row in rows
            ),
            "matcher_false_accepts": sum(
                int(row["matcher_false_accept"]) for row in rows
            ),
            "raw_neural_false_accepts": sum(
                int(row["raw_neural_false_accept"]) for row in rows
            ),
            "queries": len(rows),
        }
    return pooled


def _compare_tree(
    checks: Checks, actual: Any, expected: Any, path: str
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{path}: mapping keys differ")
        for key, value in expected.items():
            _compare_tree(checks, actual[key], value, f"{path}.{key}")
    elif isinstance(expected, bool) or expected is None or isinstance(
            expected, str):
        checks.same(actual, expected, path)
    elif isinstance(expected, (int, float)):
        checks.equal_number(actual, expected, path)
    else:
        raise ValueError(f"{path}: unsupported report value")


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Recompute schedules, invariants, and seed-pooled numeric results."""
    checks = Checks()
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    checks.same(report.get("model_seeds"), list(SEEDS), "model_seeds")
    checks.same(
        report.get("protocol_name"),
        "desktop-creature-sound-backend-comparison-v1",
        "protocol_name",
    )
    checks.same(report.get("protocol_sha256"), PROTOCOL_SHA256,
                "protocol_sha256")
    checks.same(report.get("notice"),
                "Synthetic feature probes only; no microphone accuracy claim.",
                "notice")
    checks.equal_number(report.get("queries_per_seed"), 48,
                        "queries_per_seed")
    per_seed = report.get("per_seed")
    if not isinstance(per_seed, dict) or set(per_seed) != {
            str(seed) for seed in SEEDS}:
        raise ValueError("per_seed keys do not match the seed schedule")

    for seed in SEEDS:
        run = per_seed[str(seed)]
        checks.equal_number(run.get("model_seed"), seed,
                            f"seed_{seed}.model_seed")
        checks.equal_number(run.get("total_training_examples"), 12,
                            f"seed_{seed}.total_training_examples")
        checks.equal_number(
            run.get("teaching", {}).get("examples_by_label", {}).get(
                "bell", 0
            ), 3, f"seed_{seed}.bell_training_examples",
        )
        for label in LABELS[1:]:
            checks.equal_number(
                run["teaching"]["examples_by_label"].get(label, 0), 3,
                f"seed_{seed}.{label}_training_examples",
            )

        stages = {stage["id"]: stage for stage in run["stages"]}
        checks.same(set(stages), {item[0] for item in STAGES},
                    f"seed_{seed}.stage_ids")
        for stage_id, known_queries, known_labels, unknown_families in STAGES:
            stage = stages[stage_id]
            prefix = f"seed_{seed}.{stage_id}"
            checks.equal_number(stage.get("queries"), known_queries,
                                f"{prefix}.queries")
            checks.same(set(stage.get("per_class", {})), set(known_labels),
                        f"{prefix}.per_class_labels")
            for label in known_labels:
                row = stage["per_class"][label]
                checks.equal_number(row.get("queries"), 12,
                                    f"{prefix}.{label}.queries")
                for backend in BACKENDS:
                    _check_counts(
                        checks, row[backend], 12,
                        f"{prefix}.{label}.{backend}", OUTCOMES,
                    )
                _check_counts(
                    checks, row["support_guard_effect"], 12,
                    f"{prefix}.{label}.guard", GUARD_OUTCOMES,
                )
                _check_guard_partition(checks, row, f"{prefix}.{label}")
            for backend in BACKENDS:
                _check_counts(
                    checks, stage[backend], known_queries,
                    f"{prefix}.{backend}", OUTCOMES,
                )
            _check_counts(
                checks, stage["support_guard_effect"], known_queries,
                f"{prefix}.guard", GUARD_OUTCOMES,
            )
            _check_guard_partition(checks, stage, prefix)
            checks.same(
                set(stage.get("unknown_family_controls", {})),
                set(unknown_families), f"{prefix}.unknown_families",
            )
            for family in unknown_families:
                row = stage["unknown_family_controls"][family]
                checks.equal_number(row.get("queries"), 12,
                                    f"{prefix}.{family}.unknown_queries")
                for metric in (
                    "snn_false_accepts", "raw_neural_false_accepts",
                    "matcher_false_accepts",
                ):
                    _check_count_range(
                        checks, row.get(metric), 12,
                        f"{prefix}.{family}.{metric}",
                    )
                if (row["snn_false_accepts"]
                        > row["raw_neural_false_accepts"]):
                    raise ValueError(
                        f"{prefix}.{family}: guard accepted without "
                        "a raw neural label"
                    )
            expected_controls = set(CONTROL_NAMES) if stage_id in {
                "four_labels", "after_reload"
            } else set()
            controls = stage.get("controls", {})
            checks.same(set(controls), expected_controls,
                        f"{prefix}.controls")
            for name, values in controls.items():
                for field in (
                    "snn_false_accept", "matcher_false_accept",
                    "raw_neural_false_accept",
                ):
                    if type(values.get(field)) is not bool:
                        raise ValueError(f"{prefix}.{name}.{field}: bool")
                if (values["snn_false_accept"]
                        and not values["raw_neural_false_accept"]):
                    raise ValueError(
                        f"{prefix}.{name}: guard accepted without raw label"
                    )
            checks.truth(stage.get("model_unchanged_during_probes"),
                         f"{prefix}.model_unchanged_during_probes")
            checks.truth(stage.get("examples_unchanged_during_probes"),
                         f"{prefix}.examples_unchanged_during_probes")
            checks.same(stage.get("model_hash_before_probes"),
                        stage.get("model_hash_after_probes"),
                        f"{prefix}.model_hash")
            checks.same(stage.get("example_fingerprint_before_probes"),
                        stage.get("example_fingerprint_after_probes"),
                        f"{prefix}.example_fingerprint")
            model_hash = stage.get("model_hash_before_probes")
            example_hash = stage.get("example_fingerprint_before_probes")
            if not isinstance(model_hash, str) or not model_hash:
                raise ValueError(f"{prefix}.model_hash: missing")
            if not isinstance(example_hash, str) or not example_hash:
                raise ValueError(f"{prefix}.example_fingerprint: missing")
            summary = stage.get("summary", {})
            examples, probe_count = SUMMARY_EXAMPLES[stage_id]
            checks.equal_number(summary.get("total_examples"), examples,
                                f"{prefix}.summary.total_examples")
            checks.equal_number(summary.get("probe_count"), probe_count,
                                f"{prefix}.summary.probe_count")
            checks.equal_number(summary.get("revision"), examples,
                                f"{prefix}.summary.revision")
            checks.same(
                {item["name"] for item in summary.get("labels", [])},
                set(known_labels), f"{prefix}.summary.labels",
            )
            checks.same(
                stage.get("examples_by_label"),
                {label: 3 for label in known_labels},
                f"{prefix}.examples_by_label",
            )
            control_count = len(expected_controls)
            recognition_queries = (
                known_queries + 12 * len(unknown_families) + control_count
            )
            _check_timings(
                checks, stage, known_queries, recognition_queries, prefix
            )

        _check_learning_and_reload(checks, run, seed)

    pooled = report.get("pooled_stages")
    if not isinstance(pooled, list) or len(pooled) != len(STAGES):
        raise ValueError("pooled_stages has the wrong length")
    pooled_by_id = {item["id"]: item for item in pooled}
    checks.same(set(pooled_by_id), {item[0] for item in STAGES},
                "pooled_stages.ids")
    for stage_id, _, _, _ in STAGES:
        expected = _pooled_from_seeds(per_seed, stage_id)
        _compare_tree(
            checks, pooled_by_id[stage_id], expected,
            f"pooled_stages.{stage_id}",
        )

    checks.equal_number(report.get("total_training_examples_per_seed"), 12,
                        "total_training_examples_per_seed")
    _check_report_time_totals(checks, report, per_seed)
    return {
        "numeric_assertions": checks.numeric_assertions,
        "model_seeds": list(SEEDS),
        "stage_known_queries_per_seed": {
            stage_id: count for stage_id, count, _, _ in STAGES
        },
        "timed_recognitions_per_seed_by_stage": {
            stage_id: (known_queries + 12 * len(unknown_families)
                       + (len(CONTROL_NAMES) if stage_id in {
                           "four_labels", "after_reload"
                       } else 0))
            for stage_id, known_queries, _, unknown_families in STAGES
        },
        "known_query_total_per_seed_across_stages": sum(
            count for _, count, _, _ in STAGES
        ),
        "known_query_total_all_seeds_across_stages": sum(
            count for _, count, _, _ in STAGES
        ) * len(SEEDS),
        "timing_scope": (
            "SNN and matcher timers cover the same known, unknown-family, "
            "and scheduled-control inputs; stage.queries counts known inputs "
            "only. SNN time includes its support guard. Paired-record time "
            "includes a second recognition and persistence. These are "
            "sequential wall-clock totals without repeated timing trials."
        ),
        "scope": (
            "Synthetic fixed feature probes only; this validates aggregate "
            "report consistency, not microphone accuracy or generalization."
        ),
    }


def _check_guard_partition(
    checks: Checks, row: dict[str, Any], prefix: str
) -> None:
    raw = row["raw_neural"]
    guarded = row["snn"]
    effect = row["support_guard_effect"]
    checks.equal_number(
        raw["correct"], effect["guard_accepted_raw_correct"]
        + effect["guard_rejected_raw_correct"],
        f"{prefix}.raw_correct_partition",
    )
    checks.equal_number(
        raw["wrong"], effect["guard_accepted_raw_wrong"]
        + effect["guard_rejected_raw_wrong"],
        f"{prefix}.raw_wrong_partition",
    )
    checks.equal_number(
        raw["rejected"], effect["guard_accepted_raw_unclassified"]
        + effect["guard_rejected_raw_unclassified"],
        f"{prefix}.raw_unclassified_partition",
    )
    checks.equal_number(
        guarded["correct"], effect["guard_accepted_raw_correct"],
        f"{prefix}.guarded_correct",
    )
    checks.equal_number(
        guarded["wrong"], effect["guard_accepted_raw_wrong"],
        f"{prefix}.guarded_wrong",
    )
    checks.equal_number(
        guarded["rejected"], sum(
            effect[key] for key in GUARD_OUTCOMES
            if key.startswith("guard_rejected_")
        ), f"{prefix}.guarded_rejected",
    )


def _check_count_range(
    checks: Checks, value: Any, maximum: int, path: str
) -> None:
    if type(value) is not int:
        raise ValueError(f"{path}: expected an integer count")
    checks.equal_number(value, value, f"{path}.numeric")
    if value < 0 or value > maximum:
        raise ValueError(f"{path}: outside 0..{maximum}")


def _check_counts(
    checks: Checks,
    counts: dict[str, Any],
    maximum: int,
    prefix: str,
    keys: tuple[str, ...],
) -> None:
    checks.same(set(counts), set(keys), f"{prefix}.outcome_keys")
    for key in keys:
        _check_count_range(checks, counts[key], maximum, f"{prefix}.{key}")
    checks.equal_number(
        _outcome_total(counts) if keys == OUTCOMES else sum(
            counts[key] for key in keys
        ), maximum, f"{prefix}.total",
    )


def _check_timings(
    checks: Checks,
    stage: dict[str, Any],
    known_queries: int,
    recognition_queries: int,
    prefix: str,
) -> None:
    for field in TIMING_FIELDS:
        value = stage.get(field)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or value < 0):
            raise ValueError(f"{prefix}.{field}: invalid duration")
        if (field == "paired_record_wall_seconds"
                and known_queries == 0):
            checks.equal_number(value, 0.0, f"{prefix}.{field}")
        else:
            checks.at_least(value, 0, f"{prefix}.{field}")
    if recognition_queries <= 0:
        raise ValueError(f"{prefix}: missing recognition inputs")


def _check_learning_and_reload(
    checks: Checks, run: dict[str, Any], seed: int
) -> None:
    prefix = f"seed_{seed}"
    metrics = run["final_learning_metrics"]
    checks.truth(metrics.get("trained"), f"{prefix}.trained")
    checks.equal_number(metrics.get("class_count"), 4,
                        f"{prefix}.class_count")
    checks.equal_number(metrics.get("examples"), 12,
                        f"{prefix}.trained_examples")
    checks.equal_number(metrics.get("epochs"), 100, f"{prefix}.epochs")
    _check_count_range(checks, metrics.get("actual_spikes"), 9216,
                       f"{prefix}.actual_spikes")
    checks.at_least(metrics.get("actual_spikes"), 1,
                    f"{prefix}.actual_spikes_nonzero")
    _check_count_range(checks, metrics.get("changed_input_parameters"),
                       1344, f"{prefix}.changed_input_parameters")
    _check_count_range(checks, metrics.get("changed_readout_parameters"),
                       260, f"{prefix}.changed_readout_parameters")
    checks.at_least(metrics.get("changed_input_parameters"), 1,
                    f"{prefix}.changed_input_parameters_nonzero")
    checks.at_least(metrics.get("changed_readout_parameters"), 1,
                    f"{prefix}.changed_readout_parameters_nonzero")
    for field in ("loss", "accuracy", "train_seconds",
                  "matcher_train_seconds"):
        value = metrics.get(field)
        if (value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or value < 0)):
            raise ValueError(f"{prefix}.{field}: invalid numeric metric")
    if metrics.get("accuracy") is not None and metrics["accuracy"] > 1:
        raise ValueError(f"{prefix}.accuracy: outside 0..1")
    reload_hashes = run["reload_hashes"]
    checks.truth(reload_hashes.get("model_hash_matches"),
                 f"{prefix}.model_hash_matches")
    checks.truth(reload_hashes.get("example_hash_matches"),
                 f"{prefix}.example_hash_matches")
    checks.same(reload_hashes.get("model_before"),
                reload_hashes.get("model_after"),
                f"{prefix}.reload_model_hash")
    checks.same(reload_hashes.get("examples_before"),
                reload_hashes.get("examples_after"),
                f"{prefix}.reload_examples_hash")
    for key in ("model_before", "model_after", "examples_before",
                "examples_after"):
        value = reload_hashes.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{prefix}.reload_hashes.{key}: missing")
    checks.same(set(run.get("old_label_retention", {})), set(LABELS[:3]),
                f"{prefix}.old_label_retention_labels")
    for label, retained in run["old_label_retention"].items():
        checks.truth(retained, f"{prefix}.{label}_retained")
    checks.same(set(run.get("reload_label_retention", {})), set(LABELS),
                f"{prefix}.reload_label_retention_labels")
    for label, retained in run["reload_label_retention"].items():
        checks.truth(retained, f"{prefix}.{label}_reload_retained")
    matches = run["reload_results_match"]
    for label in LABELS:
        for backend in BACKENDS:
            checks.truth(matches[label][backend],
                         f"{prefix}.{label}.{backend}_reload_match")
        checks.truth(matches[label]["support_guard_effect"],
                     f"{prefix}.{label}.guard_reload_match")
    checks.truth(matches["controls"], f"{prefix}.controls_reload_match")
    evidence = run["parameter_change_evidence"]
    checks.same(evidence.get("model_hash_after_reload"),
                reload_hashes.get("model_after"),
                f"{prefix}.parameter_evidence_hash")
    checks.same(evidence.get("reported_learning_metrics"), metrics,
                f"{prefix}.parameter_evidence_metrics")
    _check_forgetting_delta(checks, run, seed)


def _check_forgetting_delta(
    checks: Checks, run: dict[str, Any], seed: int
) -> None:
    stages = {stage["id"]: stage for stage in run["stages"]}
    three = stages["three_labels"]["per_class"]
    four = stages["four_labels"]["per_class"]
    for label in LABELS[:3]:
        for backend in ("snn", "matcher"):
            before = three[label][backend]["correct"]
            after = four[label][backend]["correct"]
            checks.equal_number(
                run["forgetting_delta"][label][backend][
                    "correct_count_delta"
                ], after - before,
                f"seed_{seed}.{label}.{backend}.correct_delta",
            )
            expected_accuracy = (
                after / four[label]["queries"]
                - before / three[label]["queries"]
            )
            checks.equal_number(
                run["forgetting_delta"][label][backend]["accuracy_delta"],
                expected_accuracy,
                f"seed_{seed}.{label}.{backend}.accuracy_delta",
            )


def _check_report_time_totals(
    checks: Checks,
    report: dict[str, Any],
    per_seed: dict[str, dict[str, Any]],
) -> None:
    totals = (
        ("total_teaching_wall_seconds_all_seeds", "wall_seconds"),
        ("total_snn_replay_fit_wall_seconds_all_seeds",
         "snn_replay_fit_wall_seconds"),
        ("total_matcher_and_persistence_wall_seconds_all_seeds",
         "matcher_and_persistence_wall_seconds"),
    )
    for output_key, teaching_key in totals:
        expected = sum(
            run["teaching"][teaching_key] for run in per_seed.values()
        )
        checks.equal_number(report.get(output_key), expected, output_key)
    for seed, run in per_seed.items():
        teaching = run["teaching"]
        for key in (
            "wall_seconds", "snn_replay_fit_wall_seconds",
            "matcher_and_persistence_wall_seconds",
        ):
            value = teaching.get(key)
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or value < 0):
                raise ValueError(f"seed_{seed}.teaching.{key}: invalid")
            checks.at_least(value, 0,
                            f"seed_{seed}.teaching.{key}.nonnegative")
        checks.equal_number(
            teaching["wall_seconds"],
            teaching["snn_replay_fit_wall_seconds"]
            + teaching["matcher_and_persistence_wall_seconds"],
            f"seed_{seed}.teaching_time_partition",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report", type=Path, nargs="?", default=DEFAULT_REPORT
    )
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = validate_report(report)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
