#!/usr/bin/env python3
"""Independently verify the frozen pre-evaluation benchmark bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rsi_programmer.probe import interpret, validate_graph

PROTOCOL_PATH = ROOT / "fixtures/benchmark_pre_eval_v1.json"
CORPUS_PATH = ROOT / "fixtures/benchmark_pre_eval_corpus_v1.json"
GENERATOR_PATH = ROOT / "scripts/generate_benchmark_pre_eval_corpus.js"
CONTROLS = (
    "streaming", "heuristic", "seeded_random", "untrained_snn",
    "trained_snn", "mlp", "recurrent",
)
SPLITS = ("train", "eval")
PROTOCOL_SHA256 = (
    "75ef7e327b02d8e7665ddba10edcdecaaed034f403b55050e44d58f229d07e84"
)
CORPUS_SHA256 = (
    "81dba9b0d683e725e4598dd19c16da55c1b3d8b8c73000df2aba54df8df4b5c7"
)


def sha256(data: bytes) -> str:
    """Return a lowercase SHA256 digest."""
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def environment_without_git(value: dict[str, Any]) -> dict[str, Any]:
    """Return runtime fields compared by the benchmark runner."""
    result = dict(value)
    result.pop("git", None)
    return result


def close(actual: Any, expected: Any) -> bool:
    """Compare reported floats with small serialization slack."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected),
                            rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


def polynomial(node: dict[str, Any]
               ) -> dict[tuple[int, int], int]:
    """Interpret a generator AST as an exact integer bivariate polynomial."""
    if not isinstance(node, dict):
        raise ValueError("malformed generator expression")
    operator = node.get("op")
    if operator == "input":
        name = node.get("name")
        if name == "x":
            return {(1, 0): 1}
        if name == "y":
            return {(0, 1): 1}
        raise ValueError("unknown generator input")
    if operator not in {"add", "sub", "mul"}:
        raise ValueError("unknown generator operator")
    left = polynomial(node.get("left"))
    right = polynomial(node.get("right"))
    result: dict[tuple[int, int], int] = defaultdict(int)
    if operator in {"add", "sub"}:
        sign = 1 if operator == "add" else -1
        for powers, coefficient in left.items():
            result[powers] += coefficient
        for powers, coefficient in right.items():
            result[powers] += sign * coefficient
    else:
        for (lx, ly), left_value in left.items():
            for (rx, ry), right_value in right.items():
                result[(lx + rx, ly + ry)] += left_value * right_value
    return {powers: value for powers, value in result.items() if value}


def polynomial_key(value: dict[tuple[int, int], int]
                   ) -> tuple[tuple[int, int, int], ...]:
    """Make an exact, sorted key for one polynomial."""
    return tuple((x_power, y_power, coefficient)
                 for (x_power, y_power), coefficient in sorted(value.items()))


def semantic_counts(corpus: dict[str, Any]) -> dict[str, Any]:
    """Count semantic duplicates without treating them as an audit failure."""
    keys: dict[str, list[tuple[tuple[int, int, int], ...]]] = {
        name: [] for name in SPLITS
    }
    for split in SPLITS:
        for task in corpus["splits"][split]:
            tree = task["generator"]["program"]
            keys[split].append(polynomial_key(polynomial(tree)))
    train_set = set(keys["train"])
    eval_set = set(keys["eval"])
    all_keys = keys["train"] + keys["eval"]
    return {
        "train_tasks": len(keys["train"]),
        "train_unique_semantics": len(train_set),
        "train_duplicate_tasks": len(keys["train"]) - len(train_set),
        "eval_tasks": len(keys["eval"]),
        "eval_unique_semantics": len(eval_set),
        "eval_duplicate_tasks": len(keys["eval"]) - len(eval_set),
        "all_unique_semantics": len(set(all_keys)),
        "train_eval_semantic_overlap": len(train_set & eval_set),
        "overlap_is_scope_limitation_only": True,
    }


def current_source_manifest(manifest: dict[str, Any]
                            ) -> dict[str, str]:
    """Recompute runner source hashes without importing its helpers."""
    before = manifest["before"]
    source_expected = before["source_sha256"]
    module_path = Path(before["environment"]["spikeforge_path"])
    if not module_path.is_absolute():
        raise ValueError("SpikeForge module path is not absolute")
    paths = set((ROOT / "rsi_programmer").rglob("*.py"))
    paths.update({
        ROOT / "scripts/benchmark_pre_eval.py",
        GENERATOR_PATH,
        PROTOCOL_PATH,
        CORPUS_PATH,
        ROOT / "requirements.txt",
    })
    lock_path = ROOT / "uv.lock"
    if lock_path.exists():
        paths.add(lock_path)
    paths.update(module_path.resolve().parent.rglob("*.py"))
    result: dict[str, str] = {}
    for path in sorted(paths):
        label = (str(path.relative_to(ROOT)) if ROOT in path.parents
                 else str(path))
        result[label] = sha256(path.read_bytes())
    if set(result) != set(source_expected):
        raise ValueError("current source manifest file set differs")
    return result


def verify_input_hashes(bundle: Path, manifest: dict[str, Any],
                        protocol: dict[str, Any]) -> dict[str, bool]:
    """Verify frozen inputs and output file hashes against their bytes."""
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    corpus_bytes = CORPUS_PATH.read_bytes()
    generator_bytes = GENERATOR_PATH.read_bytes()
    protocol_hash = sha256(protocol_bytes)
    corpus_hash = sha256(corpus_bytes)
    generator_hash = sha256(generator_bytes)
    return {
        "protocol_sha256": (protocol_hash == PROTOCOL_SHA256
                            == manifest.get("protocol_sha256")),
        "corpus_sha256": (corpus_hash == manifest.get("corpus_sha256")
                          == CORPUS_SHA256
                          == protocol["corpus"]["sha256"]),
        "generator_sha256": (
            generator_hash
            == protocol["corpus"]["generator_source"]["sha256"]
        ),
        "raw_output_sha256": (
            sha256((bundle / "raw.jsonl").read_bytes())
            == manifest.get("raw_jsonl_sha256")
        ),
        "training_output_sha256": (
            sha256((bundle / "training.json").read_bytes())
            == manifest.get("training_json_sha256")
        ),
        "aggregate_output_sha256": (
            sha256((bundle / "aggregate.json").read_bytes())
            == manifest.get("aggregate_json_sha256")
        ),
    }


def verify_sources(manifest: dict[str, Any]
                   ) -> dict[str, bool]:
    """Check before/after manifests and current source bytes."""
    before = manifest.get("before", {}).get("source_sha256", {})
    after = manifest.get("after", {}).get("source_sha256", {})
    current = current_source_manifest(manifest)
    same = before == after == current
    return {
        "source_manifest_unchanged": before == after,
        "current_source_matches_run": before == current,
        "source_frozen_flag": manifest.get("source_frozen") is True,
        "source_manifest_valid": same,
    }


def row_key(row: dict[str, Any]) -> tuple[int, str, str, int]:
    """Return the complete benchmark matrix key."""
    return (row["seed"], row["control"], row["task_id"], row["budget"])


def validate_graph_rows(graph: Any, task: dict[str, Any]
                        ) -> tuple[bool, bool, str]:
    """Validate a graph and independently execute every fixed task row."""
    if graph is None:
        return False, False, "no-graph"
    try:
        validate_graph(graph)
        train_valid = all(
            interpret(graph, x, y) == expected
            for x, y, expected in task["train"]
        )
        held_valid = all(
            interpret(graph, x, y) == expected
            for x, y, expected in task["held_out"]
        )
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        return False, False, type(exc).__name__ + ": " + str(exc)
    if not train_valid:
        return False, held_valid, "training-row-mismatch"
    if not held_valid:
        return True, False, "held-out-mismatch"
    return True, True, ""


def verify_raw(rows: list[dict[str, Any]],
               protocol: dict[str, Any], corpus: dict[str, Any],
               training: dict[str, Any]
               ) -> tuple[dict[str, bool], dict[str, Any]]:
    """Recheck raw matrix, graph validity, work, and state immutability."""
    tasks = {
        task["id"]: task for split in SPLITS
        for task in corpus["splits"][split]
    }
    eval_tasks = {task["id"] for task in corpus["splits"]["eval"]}
    seeds = protocol["training"]["seeds"]
    budgets = protocol["evaluation"]["candidate_scan_budgets"]
    expected = {
        (seed, control, task_id, budget)
        for seed in seeds for control in CONTROLS
        for task_id in eval_tasks for budget in budgets
    }
    keys = [row_key(row) for row in rows]
    unique_keys = set(keys)
    records = {
        (row["seed"], row["control"], row["task_id"], row["budget"]): row
        for row in rows
    }
    checks = {
        "row_count": len(rows) == len(expected),
        "unique_matrix_keys": len(unique_keys) == len(keys),
        "complete_matrix": unique_keys == expected,
    }
    graph_checks = 0
    acceptance_checks = 0
    counter_checks = 0
    state_checks = 0
    graph_programs = 0
    counter_failures: list[dict[str, Any]] = []
    counter_failure_reasons: Counter[str] = Counter()
    summary: dict[str, Any] = {
        "raw_rows": len(rows),
        "expected_rows": len(expected),
        "unique_keys": len(unique_keys),
        "graph_rows_checked": 0,
        "graphs_reexecuted": 0,
        "acceptance_rows_checked": 0,
        "work_rows_checked": 0,
        "state_rows_checked": 0,
        "work_counter_mismatch_count": 0,
        "work_counter_mismatch_reasons": {},
        "work_counter_mismatch_samples": [],
    }
    fit_records = {
        item["seed"]: item["controls"]
        for item in training["seeds"]
    }
    checks["training_seed_coverage"] = set(fit_records) == set(seeds)
    checks["all_returned_graphs_checked"] = True
    checks["acceptance_recomputed"] = True
    checks["work_counters_valid"] = True
    checks["weights_frozen_per_seed"] = True
    checks["heuristic_fixed_across_seeds"] = True
    previous_by_task: dict[tuple[int, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        task = tasks.get(row["task_id"])
        if task is None:
            checks["all_returned_graphs_checked"] = False
            continue
        if row["task_id"] not in eval_tasks:
            checks["all_returned_graphs_checked"] = False
            continue
        graph = row.get("returned_graph")
        solved = row.get("solved") is True
        train_valid = False
        held_valid = False
        if graph is not None:
            train_valid, held_valid, _error = validate_graph_rows(graph, task)
            graph_checks += 1
            graph_programs += 1
            graph_validation = row.get("validation", {})
            if (graph_validation.get("train_valid") is not train_valid
                    or graph_validation.get("held_out_valid") is not
                    held_valid):
                checks["acceptance_recomputed"] = False
        expected_acceptance = solved and train_valid and held_valid
        if row.get("accepted") is not expected_acceptance:
            checks["acceptance_recomputed"] = False
        if row.get("train_valid") is not train_valid:
            checks["acceptance_recomputed"] = False
        if row.get("held_out_valid") is not held_valid:
            checks["acceptance_recomputed"] = False
        validation = row.get("validation", {})
        if validation.get("train_valid") is not train_valid:
            checks["acceptance_recomputed"] = False
        if validation.get("held_out_valid") is not held_valid:
            checks["acceptance_recomputed"] = False
        acceptance_checks += 1

        search = row.get("search", {})
        tries = search.get("tried")
        signatures = search.get("signature_evaluations")
        node_rows = search.get("node_row_operations")
        budget = row.get("budget")
        correct_work = (
            type(tries) is int and type(signatures) is int
            and type(node_rows) is int and tries == signatures
            and 0 <= signatures <= budget
            and node_rows == signatures * len(task["train"])
        )
        policy = row["control"]
        features = search.get("feature_builds")
        scored = search.get("scored_descriptors")
        generated = search.get("generated_descriptors")
        lookups = search.get("operand_vector_lookups")
        if policy == "streaming":
            correct_work = correct_work and features == 0 and scored == 0
            lookahead = (generated - signatures
                         if type(generated) is int
                         and type(signatures) is int else None)
            streaming_generation_ok = (
                lookahead == 0
                or (lookahead == 1
                    and search.get("budget_exhausted") is True
                    and signatures == budget
                    and search.get("signature_budget_exhausted") is False
                    and row.get("solved") is not True)
            )
            correct_work = (correct_work and type(generated) is int
                            and streaming_generation_ok)
            want_lookups = 2 * signatures
        else:
            correct_work = correct_work and features == scored == generated
            correct_work = correct_work and type(features) is int
            want_lookups = 2 * (signatures + (features or 0))
        correct_work = correct_work and lookups == want_lookups
        steps_by_control = {
            "streaming": 0, "heuristic": 1, "seeded_random": 1,
            "untrained_snn": 6, "trained_snn": 6,
            "mlp": 1, "recurrent": 6,
        }
        policy_steps = steps_by_control[policy]
        batches = search.get("policy_batches")
        ticks = search.get("policy_batch_ticks")
        row_steps = search.get("policy_row_steps")
        snn_steps = search.get("snn_steps")
        wanted_snn_steps = (policy_steps * batches
                            if policy in {"trained_snn", "untrained_snn"}
                            else 0)
        correct_work = correct_work and (
            type(batches) is int and type(ticks) is int
            and type(row_steps) is int and type(snn_steps) is int
            and ticks == policy_steps * batches
            and row_steps == policy_steps * scored
            and snn_steps == wanted_snn_steps
        )
        counter_checks += 1
        if not correct_work:
            checks["work_counters_valid"] = False
            reasons = []
            if tries != signatures:
                reasons.append("tried_vs_signature_evaluations")
            if (type(signatures) is not int or signatures < 0
                    or signatures > budget):
                reasons.append("signature_budget")
            if node_rows != signatures * len(task["train"]):
                reasons.append("candidate_row_operations")
            if policy == "streaming" and not streaming_generation_ok:
                reasons.append("streaming_descriptor_pull_count")
            if policy != "streaming" and not (
                    features == scored == generated
                    and type(features) is int):
                reasons.append("ranked_descriptor_counts")
            if lookups != want_lookups:
                reasons.append("operand_vector_lookups")
            if not (
                    type(batches) is int and type(ticks) is int
                    and type(row_steps) is int and type(snn_steps) is int
                    and ticks == policy_steps * batches
                    and row_steps == policy_steps * scored
                    and snn_steps == wanted_snn_steps):
                reasons.append("policy_step_counts")
            counter_failure_reasons.update(reasons or ["unspecified"])
            if len(counter_failures) < 25:
                counter_failures.append({
                    "seed": row.get("seed"),
                    "control": policy,
                    "task_id": row.get("task_id"),
                    "budget": budget,
                    "reasons": reasons,
                    "tried": tries,
                    "signature_evaluations": signatures,
                    "candidate_row_operations": node_rows,
                    "feature_builds": features,
                    "scored_descriptors": scored,
                    "generated_descriptors": generated,
                    "operand_vector_lookups": lookups,
                })

        seed_controls = fit_records.get(row["seed"], {})
        fit = seed_controls.get(policy, {})
        before_state = row.get("state_fingerprint_before_eval")
        after_state = row.get("state_fingerprint_after_eval")
        before_weights = row.get("weights_before_eval_sha256")
        after_weights = row.get("weights_after_eval_sha256")
        frozen = row.get("state_frozen_during_eval") is True
        if policy == "streaming":
            correct_state = (
                frozen and before_state == "streaming-order-none"
                and after_state is None and before_weights == "none"
                and after_weights is None
            )
        else:
            correct_state = (frozen and before_state == after_state
                             and before_weights == after_weights)
        if policy not in {"streaming", "seeded_random"}:
            correct_state = correct_state and (
                before_state == fit.get("state_fingerprint_after_fit")
                and before_weights == fit.get("weights_after_fit_sha256")
            )
        state_checks += 1
        if not correct_state:
            checks["weights_frozen_per_seed"] = False
        previous_by_task[(row["seed"], row["control"],
                          row["task_id"])].append(row)

    random_states: dict[tuple[int, str], set[tuple[Any, Any]]] = (
        defaultdict(set)
    )
    for row in rows:
        if row.get("control") == "seeded_random":
            random_states[(row["seed"], row["task_id"])].add((
                row.get("state_fingerprint_before_eval"),
                row.get("weights_before_eval_sha256"),
            ))
    if any(len(states) != 1 for states in random_states.values()):
        checks["weights_frozen_per_seed"] = False

    for seed, controls in fit_records.items():
        heuristic = controls.get("heuristic", {})
        if not heuristic:
            checks["heuristic_fixed_across_seeds"] = False
        for control in ("trained_snn", "untrained_snn", "mlp", "recurrent"):
            record = controls.get(control, {})
            if not record.get("weights_frozen_during_eval", False):
                checks["weights_frozen_per_seed"] = False
        expected_meta = {
            "trained_snn": (225, 6, 0.05, "12->16_leaky->1"),
            "untrained_snn": (225, 6, 0.05, "12->16_leaky->1"),
            "mlp": (225, 1, 0.05, "12->16_relu->1"),
            "recurrent": (
                497, 6, 0.05, "12->16_tanh_rnn->mean_readout_1",
            ),
        }
        for control, (params, steps, learning_rate, architecture) in (
                expected_meta.items()):
            metadata = controls.get(control, {}).get("metadata", {})
            if (metadata.get("trainable_parameters") != params
                    or metadata.get("simulation_steps") != steps
                    or metadata.get("architecture") != architecture
                    or metadata.get("feature_dimension") != 12
                    or not close(metadata.get("learning_rate", learning_rate),
                                 learning_rate)
                    or metadata.get("fit_mode") != "adam_margin_lr_0.05"):
                checks["model_metadata_matches_protocol"] = False
        for control in ("trained_snn", "mlp", "recurrent"):
            if not controls.get(control, {}).get("fit_performed", False):
                checks["model_metadata_matches_protocol"] = False
        for control in ("untrained_snn", "heuristic"):
            if controls.get(control, {}).get("fit_performed") is not False:
                checks["model_metadata_matches_protocol"] = False
        trained = controls.get("trained_snn", {})
        untrained = controls.get("untrained_snn", {})
        if (trained.get("weights_before_fit_sha256")
                != untrained.get("weights_before_fit_sha256")
                or untrained.get("weights_after_fit_sha256")
                != untrained.get("weights_before_fit_sha256")):
            checks["model_metadata_matches_protocol"] = False
        for control in ("trained_snn", "mlp", "recurrent"):
            work = controls.get(control, {}).get("fit_work", {})
            if (work.get("requested_epochs") != protocol["training"][
                    "epochs"] or work.get("optimizer_updates") !=
                    protocol["training"]["epochs"]):
                checks["model_metadata_matches_protocol"] = False

    heuristic_states = {
        row.get("state_fingerprint_before_eval")
        for row in rows if row.get("control") == "heuristic"
    }
    heuristic_weights = {
        row.get("weights_before_eval_sha256")
        for row in rows if row.get("control") == "heuristic"
    }
    if len(heuristic_states) != 1 or len(heuristic_weights) != 1:
        checks["heuristic_fixed_across_seeds"] = False

    monotone = True
    for (_seed, _control, _task), task_rows in previous_by_task.items():
        task_rows.sort(key=lambda item: item["budget"])
        for smaller, larger in zip(task_rows, task_rows[1:]):
            small_work = smaller.get("search", {}).get(
                "signature_evaluations", -1,
            )
            large_work = larger.get("search", {}).get(
                "signature_evaluations", -1,
            )
            if (large_work < small_work
                    or (smaller.get("accepted") is True
                        and larger.get("accepted") is not True)):
                monotone = False
    checks["budget_prefix_monotone"] = monotone
    checks.setdefault("model_metadata_matches_protocol", True)
    summary.update({
        "graph_rows_checked": graph_checks,
        "graphs_reexecuted": graph_programs,
        "acceptance_rows_checked": acceptance_checks,
        "work_rows_checked": counter_checks,
        "state_rows_checked": state_checks,
        "work_counter_mismatch_count": sum(counter_failure_reasons.values()),
        "work_counter_mismatch_reasons": dict(counter_failure_reasons),
        "work_counter_mismatch_samples": counter_failures,
    })
    return checks, summary


def expected_aggregate(rows: list[dict[str, Any]],
                       protocol: dict[str, Any]
                       ) -> dict[str, Any]:
    """Recompute family/overall summaries and decision rule from raw rows."""
    families = protocol["corpus"]["families"]
    seeds = protocol["training"]["seeds"]
    budgets = protocol["evaluation"]["candidate_scan_budgets"]
    grouped: dict[tuple[str, int, str, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[(row["control"], row["budget"], row["family"],
                 row["seed"])].append(row)
    family_means: dict[tuple[str, int, str], float] = {}
    overall_means: dict[tuple[str, int], float] = {}
    family_seed_rows: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    overall_seed_rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for control in CONTROLS:
        for budget in budgets:
            for family in families:
                fractions = []
                for seed in seeds:
                    items = grouped[(control, budget, family, seed)]
                    successes = sum(row.get("accepted") is True
                                    for row in items)
                    scans = sum(row.get("search", {}).get("tried", 0)
                                for row in items)
                    wall = sum(row.get("wall_seconds_including_wrapper", 0.0)
                               for row in items)
                    snn_steps = sum(row.get("search", {}).get(
                        "snn_steps", 0) for row in items)
                    family_seed_rows[(control, budget, family, seed)] = {
                        "successes": successes,
                        "tasks": len(items),
                        "success_fraction": (successes / len(items)
                                             if items else 0.0),
                        "candidate_signatures": scans,
                        "wall_seconds": wall,
                        "snn_steps": snn_steps,
                    }
                    fractions.append(
                        successes / len(items) if items else 0.0
                    )
                family_means[(control, budget, family)] = (
                    sum(fractions) / len(fractions)
                )
            seed_fractions = []
            for seed in seeds:
                items = [row for row in rows
                         if row["control"] == control
                         and row["budget"] == budget
                         and row["seed"] == seed]
                seed_fractions.append(
                    sum(row.get("accepted") is True for row in items)
                    / len(items) if items else 0.0
                )
                successes = sum(row.get("accepted") is True
                                for row in items)
                overall_seed_rows[(control, budget, seed)] = {
                    "successes": successes,
                    "tasks": len(items),
                    "success_fraction": (successes / len(items)
                                         if items else 0.0),
                    "candidate_signatures": sum(
                        row.get("search", {}).get("tried", 0)
                        for row in items
                    ),
                    "wall_seconds": sum(
                        row.get("wall_seconds_including_wrapper", 0.0)
                        for row in items
                    ),
                    "snn_steps": sum(
                        row.get("search", {}).get("snn_steps", 0)
                        for row in items
                    ),
                }
            overall_means[(control, budget)] = (
                sum(seed_fractions) / len(seed_fractions)
            )
    matched = []
    for control in CONTROLS:
        for target in protocol["decision_rule"][
                "matched_success_targets"]:
            reaches = [budget for budget in budgets
                       if overall_means[(control, budget)] >= target]
            matched.append({
                "control": control,
                "target": target,
                "lowest_budget": min(reaches) if reaches else None,
                "unreached_censored_at": (None if reaches else max(budgets)),
            })
    advantage = []
    for budget in (512, 2048):
        best_other = max(overall_means[(name, budget)]
                         for name in CONTROLS if name != "trained_snn")
        stronger_families = sum(
            family_means[("trained_snn", budget, family)]
            > max(family_means[(name, budget, family)]
                  for name in CONTROLS if name != "trained_snn")
            for family in families
        )
        margin = overall_means[("trained_snn", budget)] - best_other
        advantage.append({
            "budget": budget,
            "trained_snn_mean_fraction": overall_means[
                ("trained_snn", budget)
            ],
            "best_other_mean_fraction": best_other,
            "overall_margin": margin,
            "families_where_trained_snn_is_higher": stronger_families,
        })
    return {
        "family_means": family_means,
        "overall_means": overall_means,
        "family_seed_rows": family_seed_rows,
        "overall_seed_rows": overall_seed_rows,
        "matched": matched,
        "advantage_checks": advantage,
        "advantage_passes": all(
            item["overall_margin"] >= 0.10
            and item["families_where_trained_snn_is_higher"] >= 2
            for item in advantage
        ),
    }


def compare_aggregate(rows: list[dict[str, Any]],
                      aggregate: dict[str, Any],
                      protocol: dict[str, Any]) -> dict[str, bool]:
    """Compare published aggregate rates/rules to independent raw counts."""
    expected = expected_aggregate(rows, protocol)
    checks = {
        "family_summary_matches_raw": True,
        "overall_summary_matches_raw": True,
        "family_seed_rows_match_raw": True,
        "overall_seed_rows_match_raw": True,
        "matched_targets_match_raw": True,
        "advantage_rule_matches_raw": True,
    }
    for item in aggregate.get("family_means_across_seeds", []):
        key = (item["control"], item["budget"], item["family"])
        value = expected["family_means"].get(key)
        if value is None or not close(item.get("mean_seed_fraction"), value):
            checks["family_summary_matches_raw"] = False
    if len(aggregate.get("family_means_across_seeds", [])) != len(
            expected["family_means"]):
        checks["family_summary_matches_raw"] = False
    actual_family_rows = aggregate.get("family_seed_rows", [])
    if len(actual_family_rows) != len(expected["family_seed_rows"]):
        checks["family_seed_rows_match_raw"] = False
    for item in actual_family_rows:
        key = (item["control"], item["budget"], item["family"],
               item["seed"])
        wanted = expected["family_seed_rows"].get(key)
        if wanted is None or any(not close(item.get(name), value)
                                 for name, value in wanted.items()):
            checks["family_seed_rows_match_raw"] = False
    for item in aggregate.get("overall_means_across_seeds", []):
        key = (item["control"], item["budget"])
        value = expected["overall_means"].get(key)
        if value is None or not close(item.get("mean_seed_fraction"), value):
            checks["overall_summary_matches_raw"] = False
    if len(aggregate.get("overall_means_across_seeds", [])) != len(
            expected["overall_means"]):
        checks["overall_summary_matches_raw"] = False
    actual_overall_rows = aggregate.get("overall_seed_rows", [])
    if len(actual_overall_rows) != len(expected["overall_seed_rows"]):
        checks["overall_seed_rows_match_raw"] = False
    for item in actual_overall_rows:
        key = (item["control"], item["budget"], item["seed"])
        wanted = expected["overall_seed_rows"].get(key)
        if wanted is None or any(not close(item.get(name), value)
                                 for name, value in wanted.items()):
            checks["overall_seed_rows_match_raw"] = False
    published_matched = aggregate.get("matched_success_budgets", [])
    if len(published_matched) != len(expected["matched"]):
        checks["matched_targets_match_raw"] = False
    for actual, wanted in zip(published_matched, expected["matched"]):
        if any(not close(actual.get(key), wanted[key]) for key in (
                "control", "target", "lowest_budget",
                "unreached_censored_at")):
            checks["matched_targets_match_raw"] = False
    advantage = aggregate.get("trained_snn_advantage", {})
    actual_checks = advantage.get("checks", [])
    if len(actual_checks) != len(expected["advantage_checks"]):
        checks["advantage_rule_matches_raw"] = False
    for actual, wanted in zip(actual_checks, expected["advantage_checks"]):
        for key, value in wanted.items():
            if not close(actual.get(key), value):
                checks["advantage_rule_matches_raw"] = False
    if advantage.get("rule_would_pass") is not expected["advantage_passes"]:
        checks["advantage_rule_matches_raw"] = False
    infra = advantage.get("infrastructure_valid") is True
    supported = expected["advantage_passes"] and infra
    if advantage.get("supported_by_frozen_rule") is not supported:
        checks["advantage_rule_matches_raw"] = False
    return checks


def run(bundle: Path) -> dict[str, Any]:
    """Audit a completed bundle and return machine-readable checks."""
    protocol = read_json(PROTOCOL_PATH)
    corpus = read_json(CORPUS_PATH)
    training_path = bundle / "training.json"
    aggregate_path = bundle / "aggregate.json"
    manifest_path = bundle / "manifest.json"
    training = read_json(training_path)
    aggregate = read_json(aggregate_path)
    manifest = read_json(manifest_path)
    rows = [json.loads(line) for line in
            (bundle / "raw.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    input_checks = verify_input_hashes(bundle, manifest, protocol)
    source_checks = verify_sources(manifest)
    raw_checks, counts = verify_raw(rows, protocol, corpus, training)
    aggregate_checks = compare_aggregate(rows, aggregate, protocol)
    checks = {
        **input_checks,
        **source_checks,
        **raw_checks,
        **aggregate_checks,
        "manifest_grid_valid": manifest.get("grid_valid") is True,
        "manifest_infrastructure_valid": (
            manifest.get("infrastructure_valid") is True
        ),
        "aggregate_infrastructure_valid": (
            aggregate.get("trained_snn_advantage", {}).get(
                "infrastructure_valid"
            ) is True
        ),
        "manifest_row_counts_match_raw": (
            manifest.get("row_count") == len(rows)
            and manifest.get("expected_row_count") == len(rows)
        ),
        "run_environment_unchanged": (
            environment_without_git(manifest.get("before", {}).get(
                "environment", {}))
            == environment_without_git(manifest.get("after", {}).get(
                "environment", {}))
        ),
    }
    counts["semantic_counts"] = semantic_counts(corpus)
    counts["semantic_overlap_scope_note"] = (
        "Polynomial overlap is reported as a corpus scope limitation; "
        "the frozen split guarantees syntactic, not semantic, separation."
    )
    result = {
        "audit_version": 1,
        "bundle": str(bundle),
        "protocol_id": protocol.get("protocol_id"),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "counts": counts,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    """Write audit.json and fail if any required verification fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path,
        default=ROOT / "artifacts/benchmark-pre-eval-v1-20261018",
        help="completed benchmark output directory",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "audit.json",
        help="path for the audit result JSON",
    )
    args = parser.parse_args(argv)
    try:
        result = run(args.bundle.resolve())
    except Exception as exc:
        result = {
            "audit_version": 1,
            "bundle": str(args.bundle.resolve()),
            "all_checks_pass": False,
            "error": type(exc).__name__ + ": " + str(exc),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)
                           + "\n", encoding="utf-8")
    print(json.dumps({
        "all_checks_pass": result["all_checks_pass"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0 if result["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
