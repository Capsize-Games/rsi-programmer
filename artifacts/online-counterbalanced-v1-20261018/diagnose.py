#!/usr/bin/env python3
"""Stream descriptive persistent-SNN diagnostics from frozen bundles.

This script does not train, search, execute graph candidates, or use held
answers as inputs. It uses row acceptance only to stratify finished episodes.
All task-level repetitions are reported as repeated measurements.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, TextIO

PRIMARY_BUDGET = 144
PERSISTENT_ARM = "persistent_snn"
BASELINE_RATE = 0.05
EXPECTED_OLD_ROWS = 1440
EXPECTED_PRIMARY_ROWS = 6 * 3 * 48


def digest(path: Path) -> str:
    """Return a file's SHA-256 without retaining its bytes."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def open_rows(path: Path) -> TextIO:
    """Open a gzip JSONL stream as UTF-8 text."""
    return gzip.open(path, "rt", encoding="utf-8")


def rows(stream: Iterable[str]) -> Iterable[dict[str, Any]]:
    """Decode JSONL one row at a time."""
    for line in stream:
        yield json.loads(line)


def compare_original_order(new_raw: Path, old_raw: Path
                           ) -> dict[str, Any]:
    """Compare the unchanged first order with all original 1,440 rows."""
    matches = 0
    with open_rows(new_raw) as new_stream, open_rows(old_raw) as old_stream:
        new_iter = iter(rows(new_stream))
        old_iter = iter(rows(old_stream))
        for index, (new, old) in enumerate(zip_longest(
                (next(new_iter, None) for _ in range(EXPECTED_OLD_ROWS)),
                (next(old_iter, None) for _ in range(EXPECTED_OLD_ROWS)),
                fillvalue=None)):
            if new is None or old is None:
                raise ValueError("old-order raw comparison ended early")
            expected = (old["budget"], old["seed"], old["arm"],
                        old["task_id"])
            actual = (new["budget"], new["seed"], new["arm"],
                      new["task_id"])
            if actual != expected or new["order_id"] != "0-1-2":
                raise ValueError(
                    f"old-order row identity mismatch at index {index}")
            for field in ("attempts", "candidate_evaluations", "accepted",
                          "observed_solved", "held_correct",
                          "polynomial_correct", "weight_before",
                          "weight_after"):
                if new.get(field) != old.get(field):
                    raise ValueError(
                        f"old-order semantic mismatch at row {index}: "
                        f"{field}")
            matches += 1
        if next(old_iter, None) is not None:
            raise ValueError("original bundle has more than 1,440 rows")
    return {
        "rows_compared": matches,
        "expected_rows": EXPECTED_OLD_ROWS,
        "semantic_fields": [
            "attempts", "candidate_evaluations", "accepted",
            "observed_solved", "held_correct", "polynomial_correct",
            "weight_before", "weight_after",
        ],
        "timings_and_new_diagnostic_fields_excluded": True,
        "status": "exact_match",
    }


def _empty_outcome() -> dict[str, Any]:
    """Return counters for accepted or failed primary episodes."""
    return {
        "episodes": 0,
        "attempts": 0,
        "updates": 0,
        "nonexact_updates": 0,
        "positive_advantages": 0,
        "positive_reward_wrong_root": 0,
        "wrong_root_attempts": 0,
    }


def _cold_group() -> dict[str, float | int]:
    """Return cold diagnostic accumulators."""
    return {"episodes": 0, "max_probability_sum": 0.0,
            "entropy_sum": 0.0}


def _add_cold(group: dict[str, float | int],
              diagnostic: dict[str, Any]) -> None:
    """Accumulate a cold probability and entropy sample."""
    group["episodes"] += 1
    group["max_probability_sum"] += diagnostic["max_probability"]
    group["entropy_sum"] += diagnostic["entropy_nats"]


def _finish_cold(group: dict[str, float | int]) -> dict[str, Any]:
    """Return means for a nonempty cold-diagnostic group."""
    count = int(group["episodes"])
    if count == 0:
        return {"episodes": 0, "mean_max_probability": None,
                "mean_entropy_nats": None}
    return {
        "episodes": count,
        "mean_max_probability": float(group["max_probability_sum"]) / count,
        "mean_entropy_nats": float(group["entropy_sum"]) / count,
    }


def _finish_outcome(group: dict[str, int], total_updates: int
                    ) -> dict[str, Any]:
    """Return episode and update shares for one acceptance stratum."""
    episodes = group["episodes"]
    updates = group["updates"]
    attempts = group["attempts"]
    return {
        **group,
        "mean_attempts_per_episode": attempts / episodes if episodes else None,
        "share_of_all_updates": updates / total_updates if total_updates else None,
        "nonexact_update_fraction": (
            group["nonexact_updates"] / updates if updates else None),
        "positive_advantage_fraction": (
            group["positive_advantages"] / updates if updates else None),
        "positive_reward_fraction_among_wrong_root_attempts": (
            group["positive_reward_wrong_root"]
            / group["wrong_root_attempts"]
            if group["wrong_root_attempts"] else None),
    }


def _close(actual: float, expected: float, label: str) -> None:
    """Reject drift in the recomputed scalar baseline chain."""
    if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f"baseline chain mismatch: {label}")


def diagnose(counterbalanced_raw: Path, original_raw: Path
             ) -> dict[str, Any]:
    """Read persistent-SNN primary-budget rows and aggregate diagnostics."""
    old_order = compare_original_order(counterbalanced_raw, original_raw)
    outcomes = {"accepted": _empty_outcome(), "failed": _empty_outcome()}
    cold_position: dict[int, dict[str, float | int]] = defaultdict(_cold_group)
    cold_root: dict[str, dict[str, float | int]] = defaultdict(_cold_group)
    cold_order: dict[str, dict[str, float | int]] = defaultdict(_cold_group)
    feature_vectors: set[tuple[Any, ...]] = set()
    feature_rows = 0
    baseline_expected: dict[tuple[str, int], float] = {}
    baseline_chain_mismatches = 0
    primary_rows = 0
    total_updates = 0
    positive_advantages = 0
    wrong_root_attempts = 0
    positive_wrong_root_rewards = 0
    nonexact_updates = 0
    orders_seen: set[str] = set()

    with open_rows(counterbalanced_raw) as stream:
        for row in rows(stream):
            if (row.get("budget") != PRIMARY_BUDGET
                    or row.get("arm") != PERSISTENT_ARM):
                continue
            primary_rows += 1
            order_id = row["order_id"]
            seed = int(row["seed"])
            orders_seen.add(order_id)
            status = "accepted" if row["accepted"] else "failed"
            group = outcomes[status]
            attempts = row["attempts"]
            update_count = int(row["policy_work"].get("updates", 0))
            if update_count != len(attempts):
                raise ValueError("persistent SNN update/attempt count differs")
            diagnostic = row["diagnostic_before"]
            baseline = float(diagnostic["baseline"])
            chain_key = order_id, seed
            if chain_key in baseline_expected:
                try:
                    _close(baseline, baseline_expected[chain_key],
                           f"{order_id}/{seed}/{row['task_id']}")
                except ValueError:
                    baseline_chain_mismatches += 1
            elif not math.isclose(baseline, 0.0, abs_tol=1e-12):
                baseline_chain_mismatches += 1

            _add_cold(cold_position[int(row["phase_position"])], diagnostic)
            _add_cold(cold_root[str(row["root"])], diagnostic)
            _add_cold(cold_order[order_id], diagnostic)
            feature_rows += 1
            if attempts:
                feature_vectors.add(tuple(attempts[0]["features"]))

            group["episodes"] += 1
            group["attempts"] += len(attempts)
            group["updates"] += update_count
            total_updates += update_count
            for attempt in attempts:
                reward = float(attempt["reward"])
                advantage = reward - baseline
                if advantage > 0.0:
                    group["positive_advantages"] += 1
                    positive_advantages += 1
                if not attempt["observed_exact"]:
                    group["nonexact_updates"] += 1
                    nonexact_updates += 1
                if attempt["action"] != row["root"]:
                    group["wrong_root_attempts"] += 1
                    wrong_root_attempts += 1
                    if reward > 0.0:
                        group["positive_reward_wrong_root"] += 1
                        positive_wrong_root_rewards += 1
                baseline += BASELINE_RATE * advantage
            baseline_expected[chain_key] = baseline

    if primary_rows != EXPECTED_PRIMARY_ROWS:
        raise ValueError(
            f"expected {EXPECTED_PRIMARY_ROWS} primary persistent rows, "
            f"found {primary_rows}")
    if len(orders_seen) != 6:
        raise ValueError("expected all six phase orders")
    if baseline_chain_mismatches:
        raise ValueError(
            f"found {baseline_chain_mismatches} baseline chain mismatches")
    if sum(group["updates"] for group in outcomes.values()) != total_updates:
        raise ValueError("update aggregation mismatch")

    by_acceptance = {
        name: _finish_outcome(group, total_updates)
        for name, group in outcomes.items()
    }
    cold_strata = {
        "phase_position": {
            str(key): _finish_cold(value)
            for key, value in sorted(cold_position.items())},
        "root_family": {
            key: _finish_cold(value)
            for key, value in sorted(cold_root.items())},
        "order": {
            key: _finish_cold(value)
            for key, value in sorted(cold_order.items())},
    }
    return {
        "scope": {
            "arm": PERSISTENT_ARM,
            "budget": PRIMARY_BUDGET,
            "episodes": primary_rows,
            "orders": sorted(orders_seen),
            "task_stream": "same 48 functions repeated across orders/seeds; "
                           "not independent task draws",
            "held_answers_used_as_input": False,
            "causal_or_significance_claim": False,
        },
        "old_order_semantic_match": old_order,
        "first_features": {
            "rows": feature_rows,
            "distinct_vectors": len(feature_vectors),
            "expected_vector": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "constant": (
                feature_rows == primary_rows and len(feature_vectors) == 1
                and tuple(next(iter(feature_vectors), ()))
                == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
            "only_vector": (list(next(iter(feature_vectors)))
                            if len(feature_vectors) == 1 else None),
        },
        "by_acceptance": by_acceptance,
        "overall_work": {
            "attempts": sum(group["attempts"]
                            for group in outcomes.values()),
            "updates": total_updates,
            "nonexact_updates": nonexact_updates,
            "nonexact_update_fraction": (
                nonexact_updates / total_updates if total_updates else None),
            "positive_advantages": positive_advantages,
            "positive_advantage_fraction": (
                positive_advantages / total_updates if total_updates else None),
            "wrong_root_attempts": wrong_root_attempts,
            "positive_reward_wrong_root_attempts": (
                positive_wrong_root_rewards),
            "positive_reward_fraction_among_wrong_root_attempts": (
                positive_wrong_root_rewards / wrong_root_attempts
                if wrong_root_attempts else None),
            "baseline_ema_rate": BASELINE_RATE,
            "baseline_chain_mismatches": baseline_chain_mismatches,
            "advantage_computation": (
                "per attempt, reward - diagnostic baseline; then update "
                "baseline += 0.05 * advantage; no held values read"),
        },
        "cold_diagnostics": cold_strata,
    }


def main() -> int:
    """Parse paths and write one exact JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counterbalanced_bundle", type=Path)
    parser.add_argument("original_bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output path already exists")
    new_raw = args.counterbalanced_bundle / "episodes.jsonl.gz"
    old_raw = args.original_bundle / "episodes.jsonl.gz"
    if not new_raw.is_file() or not old_raw.is_file():
        parser.error("both bundles must contain episodes.jsonl.gz")
    result = diagnose(new_raw, old_raw)
    result["input_hashes"] = {
        "counterbalanced_raw_sha256": digest(new_raw),
        "original_raw_sha256": digest(old_raw),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
