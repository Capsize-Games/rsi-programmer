#!/usr/bin/env python3
"""Independently audit the paired, counterbalanced total-work run bundle.

Artifact-only: does not import the runtime, fit policies, or run a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "fixtures/benchmark_total_work_v1.json"
CORPUS = ROOT / "fixtures/benchmark_pre_eval_corpus_v2.json"
EXPECTED_PROTOCOL_SHA256 = (
    "b5399f0ab7f90df5191a2b72c1dd3a17706bb4d64538e616e44512cbe76a1130"
)
EXPECTED_CORPUS_SHA256 = (
    "a8fd6add90b34da732da8dc96f2c0fafcabc2f017a17cdef4b31303eec4ba921"
)
ENGINES = ("baseline", "optimized")
CONTROLS = (
    "streaming",
    "heuristic",
    "seeded_random",
    "untrained_snn",
    "trained_snn",
    "mlp",
    "recurrent",
)
LEARNED = ("trained_snn", "mlp", "recurrent")
SEEDS = (0, 1, 2)
BUDGETS = (512, 2048, 8192, 32768)
REPEATS = (0, 1)
COUNTERS = (
    "signature_evaluations",
    "generated_descriptors",
    "scored_descriptors",
    "feature_builds",
    "operand_signature_evaluations",
    "retained_operand_vectors",
    "operand_vector_lookups",
    "policy_batches",
    "policy_batch_ticks",
    "policy_row_steps",
    "snn_steps",
    "node_row_operations",
    "graph_validation_calls",
    "oracle_calls",
)
CPU_FIELDS = (
    "search_cpu_ns",
    "attempt_cpu_ns_including_hashes_and_validation",
)
WALL_FIELDS = (
    "search_wall_ns",
    "attempt_wall_ns_including_hashes_and_validation",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def numeric_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if type(left) not in (int, float) or type(right) not in (int, float):
        return left == right
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1.0)


def polynomial(node: Any) -> dict[tuple[int, int], int]:
    if not isinstance(node, dict):
        raise ValueError("malformed expression")
    op = node.get("op")
    if op == "input":
        name = node.get("name")
        if name == "x":
            return {(1, 0): 1}
        if name == "y":
            return {(0, 1): 1}
        raise ValueError("unknown expression input")
    if op not in {"add", "sub", "mul"}:
        raise ValueError("unknown expression operator")
    left = polynomial(node.get("left"))
    right = polynomial(node.get("right"))
    if op in {"add", "sub"}:
        out = dict(left)
        sign = 1 if op == "add" else -1
        for power, value in right.items():
            out[power] = out.get(power, 0) + sign * value
    else:
        out: dict[tuple[int, int], int] = defaultdict(int)
        for (lx, ly), lv in left.items():
            for (rx, ry), rv in right.items():
                out[(lx + rx, ly + ry)] += lv * rv
    return {power: value for power, value in out.items() if value}


def graph_polynomial(graph: Any) -> dict[tuple[int, int], int]:
    if not isinstance(graph, dict) or set(graph) != {
        "input_types",
        "nodes",
        "output",
    }:
        raise ValueError("malformed graph")
    if graph["input_types"] != {"x": "Int", "y": "Int"}:
        raise ValueError("unexpected graph input types")
    values: list[dict[tuple[int, int], int]] = []
    for index, node in enumerate(graph["nodes"]):
        if (
            not isinstance(node, dict)
            or set(node) != {"op", "type", "args"}
            or node["type"] != "Int"
            or node["op"] not in {"add", "sub", "mul"}
        ):
            raise ValueError("unsupported graph node")
        args = node["args"]
        if not isinstance(args, list) or len(args) != 2:
            raise ValueError("graph node must have two args")
        operands = []
        for ref in args:
            if ref == "x":
                operands.append({(1, 0): 1})
            elif ref == "y":
                operands.append({(0, 1): 1})
            elif type(ref) is int and 0 <= ref < index:
                operands.append(values[ref])
            else:
                raise ValueError("invalid or forward graph reference")
        left, right = operands
        if node["op"] in {"add", "sub"}:
            out = dict(left)
            sign = 1 if node["op"] == "add" else -1
            for power, value in right.items():
                out[power] = out.get(power, 0) + sign * value
        else:
            out = defaultdict(int)
            for (lx, ly), lv in left.items():
                for (rx, ry), rv in right.items():
                    out[(lx + rx, ly + ry)] += lv * rv
        values.append({power: value for power, value in out.items() if value})
    output = graph["output"]
    if output == "x":
        return {(1, 0): 1}
    if output == "y":
        return {(0, 1): 1}
    if type(output) is int and 0 <= output < len(values):
        return values[output]
    raise ValueError("invalid graph output")


def graph_tree_operation_count(graph: dict[str, Any]) -> int:
    """Count expanded operators reachable from the graph output."""

    def count(ref: Any) -> int:
        if ref in ("x", "y"):
            return 0
        if type(ref) is not int or not 0 <= ref < len(graph["nodes"]):
            raise ValueError(
                "invalid graph reference while counting operators"
            )
        node = graph["nodes"][ref]
        return 1 + sum(count(child) for child in node["args"])

    return count(graph["output"])


def eval_graph(graph: dict[str, Any], x: int, y: int) -> int:
    values = []
    for node in graph["nodes"]:
        args = [
            (x if ref == "x" else y if ref == "y" else values[ref])
            for ref in node["args"]
        ]
        if node["op"] == "add":
            values.append(args[0] + args[1])
        elif node["op"] == "sub":
            values.append(args[0] - args[1])
        else:
            values.append(args[0] * args[1])
    output = graph["output"]
    return x if output == "x" else y if output == "y" else values[output]


def row_key(row: dict[str, Any]) -> tuple[int, str, int, str, str, int]:
    return (
        row["repeat"],
        row["engine"],
        row["seed"],
        row["control"],
        row["task_id"],
        row["budget"],
    )


def expected_keys(
    tasks: list[dict[str, Any]],
) -> set[tuple[int, str, int, str, str, int]]:
    return {
        (repeat, engine, seed, control, task["id"], budget)
        for repeat in REPEATS
        for engine in ENGINES
        for seed in SEEDS
        for control in CONTROLS
        for task in tasks
        for budget in BUDGETS
    }


def expected_order(
    seed: int, control: str, task_id: str, budget: int, repeat: int
) -> list[str]:
    payload = f"83080:{seed}:{task_id}:{budget}:{control}".encode()
    first_baseline = hashlib.sha256(payload).digest()[0] & 1 == 0
    if repeat == 1:
        first_baseline = not first_baseline
    return list(ENGINES) if first_baseline else list(reversed(ENGINES))


def load_bundle(
    directory: Path,
    corpus: dict[str, Any],
    protocol_hash: str,
    corpus_hash: str,
    protocol: dict[str, Any],
) -> tuple[
    dict[tuple[int, str, int, str, str, int], dict[str, Any]], dict[str, Any]
]:
    errors: list[str] = []
    manifest_path = directory / "manifest.json"
    raw_path = directory / "raw.jsonl"
    training_path = directory / "training.json"
    aggregate_path = directory / "aggregate.json"
    for path in (manifest_path, raw_path, training_path, aggregate_path):
        if not path.is_file():
            errors.append(f"missing {path.name}")
    if errors:
        return {}, {"directory": str(directory.resolve()), "errors": errors}
    manifest = json.loads(manifest_path.read_text())
    raw = raw_path.read_bytes()
    training = json.loads(training_path.read_text())
    aggregate_bytes = aggregate_path.read_bytes()
    aggregate = json.loads(aggregate_bytes)
    tasks = corpus["splits"]["eval"]
    by_id = {task["id"]: task for task in tasks}
    target = {
        task["id"]: polynomial(task["generator"]["program"]) for task in tasks
    }
    expected = expected_keys(tasks)
    if training.get("status") != "complete":
        errors.append("training checkpoint is not complete")
    if manifest.get("raw_jsonl_sha256") != sha256(raw):
        errors.append("raw JSONL hash mismatch")
    if manifest.get("training_json_sha256") != sha256(
        training_path.read_bytes()
    ):
        errors.append("training JSON hash mismatch")
    if manifest.get("aggregate_json_sha256") != sha256(aggregate_bytes):
        errors.append("aggregate JSON hash mismatch")
    if manifest.get("protocol_sha256") != protocol_hash:
        errors.append("protocol hash mismatch")
    if manifest.get("corpus_sha256") != corpus_hash:
        errors.append("corpus hash mismatch")
    if (
        manifest.get("expected_row_count") != len(expected)
        or manifest.get("row_count") != len(expected)
        or not manifest.get("grid_valid")
    ):
        errors.append("manifest does not attest the complete 16,128-cell grid")
    if not manifest.get("source_frozen") or not manifest.get(
        "infrastructure_valid"
    ):
        errors.append("manifest reports source or infrastructure failure")
    for name in (
        "grid_valid",
        "paired_behavior_identical",
        "repeats_behavior_identical",
        "model_states_frozen",
        "fit_reference_valid",
        "training_groups_valid",
        "no_row_errors",
        "environment_frozen",
        "infrastructure_valid",
    ):
        if manifest.get(name) is not True or aggregate.get(name) is not True:
            errors.append(f"manifest/aggregate does not attest {name}")
    if (
        manifest.get("baseline_sha256")
        != protocol["comparison"]["baseline_source"]["sha256"]
    ):
        errors.append(
            "manifest baseline source hash differs from frozen protocol"
        )
    if manifest.get("baseline_revision") != protocol["baseline_revision"]:
        errors.append(
            "manifest baseline revision differs from frozen protocol"
        )
    if manifest.get("before", {}).get("source_sha256") != manifest.get(
        "after", {}
    ).get("source_sha256"):
        errors.append("before/after source hash manifests differ")
    source_hashes = manifest.get("before", {}).get("source_sha256", {})
    source_files = source_hashes.get("source_files_sha256", {})
    if not source_files or source_files != manifest.get("after", {}).get(
        "source_sha256", {}
    ).get("source_files_sha256"):
        errors.append("nested source file hashes are absent or changed")
    for source_name, expected_digest in source_files.items():
        source_path = Path(source_name)
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        try:
            actual_digest = sha256(source_path.read_bytes())
        except OSError as exc:
            errors.append(f"cannot read frozen source {source_name}: {exc}")
        else:
            if actual_digest != expected_digest:
                errors.append(
                    f"current source differs from manifest: {source_name}"
                )
    expected_source_fields = {
        "baseline_source_sha256": protocol["comparison"]["baseline_source"][
            "sha256"
        ],
        "protocol_sha256": protocol_hash,
        "corpus_sha256": corpus_hash,
        "model_reference_sha256": protocol["comparison"]["model_reference"][
            "sha256"
        ],
    }
    for field, value in expected_source_fields.items():
        if source_hashes.get(field) != value:
            errors.append(f"source manifest {field} differs from frozen input")
    rows: dict[tuple[int, str, int, str, str, int], dict[str, Any]] = {}
    expected_weights = protocol["comparison"]["model_reference"][
        "expected_weights_after_fit"
    ]
    expected_seeds = {str(seed) for seed in SEEDS}
    observed_seeds = {
        str(seed.get("seed")) for seed in training.get("seeds", [])
    }
    if observed_seeds != expected_seeds:
        errors.append(
            "training report does not contain exactly the frozen seeds"
        )
    for seed_record in training.get("seeds", []):
        seed_text = str(seed_record.get("seed"))
        controls = seed_record.get("controls", {})
        for control, expected_hash in expected_weights.get(
            seed_text, {}
        ).items():
            actual_hash = controls.get(control, {}).get(
                "weights_after_fit_sha256"
            )
            if actual_hash != expected_hash:
                errors.append(
                    f"post-fit model hash mismatch for seed {seed_text} {control}"
                )
        if seed_record.get("all_model_states_frozen") is not True:
            errors.append(
                f"models not frozen after evaluation for seed {seed_text}"
            )
        for control, checkpoint in seed_record.get(
            "state_after_all_evaluation", {}
        ).items():
            if (
                checkpoint.get("state_matches_after_fit") is not True
                or checkpoint.get("weights_match_after_fit") is not True
            ):
                errors.append(
                    f"post-evaluation state changed for seed {seed_text} {control}"
                )
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(line)
            key = row_key(row)
            if key in rows:
                errors.append(f"duplicate row key at line {line_number}")
                continue
            rows[key] = row
            repeat, engine, seed, control, task_id, budget = key
            if key not in expected:
                errors.append(f"unexpected matrix cell at line {line_number}")
            task_id = row["task_id"]
            if row.get("protocol_id") != "benchmark-total-work-v1":
                errors.append(f"protocol ID mismatch at line {line_number}")
            if row.get("corpus_sha256") != corpus_hash:
                errors.append(f"corpus digest mismatch at line {line_number}")
            fingerprint = row.get("behavior_fingerprint")
            if (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)
            ):
                errors.append(
                    f"invalid behavior fingerprint at line {line_number}"
                )
            if row.get("error"):
                errors.append(
                    f"runner error at line {line_number}: {row['error']}"
                )
            if row.get("state_frozen_during_eval") is not True:
                errors.append(
                    f"policy state changed during evaluation at line {line_number}"
                )
            if engine not in ENGINES or repeat not in REPEATS:
                errors.append(f"invalid engine/repeat at line {line_number}")
            if row.get("engine_order") != expected_order(
                seed, control, task_id, budget, repeat
            ):
                errors.append(f"engine order mismatch at line {line_number}")
            if (
                engine == "optimized"
                and row.get("paired_behavior_matches_baseline") is not True
            ):
                errors.append(
                    f"optimized row lacks paired behavior attestation at line {line_number}"
                )
            if (
                engine == "baseline"
                and row.get("paired_behavior_matches_baseline") is not None
            ):
                errors.append(
                    f"baseline row has invalid pair attestation at line {line_number}"
                )
            if (
                control not in {"streaming", "seeded_random"}
                and row.get("weights_before_eval_sha256")
                != row.get("expected_weights_after_fit_sha256")
            ):
                errors.append(
                    f"pre-evaluation weights differ from fit checkpoint at line {line_number}"
                )
            if row.get("weights_match_after_fit_reference") is False:
                errors.append(
                    f"weights differ from frozen fit reference at line {line_number}"
                )
            for field in CPU_FIELDS + WALL_FIELDS:
                value = row.get(field)
                if type(value) is not int or value < 0:
                    errors.append(
                        f"invalid/missing {field} at line {line_number}"
                    )
            search = row.get("search") or {}
            for field in COUNTERS:
                value = search.get(field)
                if field not in search:
                    errors.append(f"missing {field} at line {line_number}")
                if type(value) is not int or value < 0:
                    errors.append(f"invalid {field} at line {line_number}")
            if search.get("tried") != search.get("signature_evaluations"):
                errors.append(
                    f"tried/signature count mismatch at line {line_number}"
                )
            if search.get("signature_evaluations", 0) > budget:
                errors.append(
                    f"signature budget exceeded at line {line_number}"
                )
            if row.get("root_calls") != search.get("signature_evaluations"):
                errors.append(
                    f"root_calls/signature count mismatch at line {line_number}"
                )
            graph = row.get("returned_graph")
            if graph is not None:
                if (
                    graph_tree_operation_count(graph)
                    > protocol["evaluation"]["tree_bound"]
                ):
                    errors.append(
                        f"returned graph exceeds tree bound at line {line_number}"
                    )
                if graph_polynomial(graph) != target[task_id]:
                    errors.append(
                        f"returned graph has wrong exact polynomial at line {line_number}"
                    )
                for split in ("train", "held_out"):
                    for x, y, expected_value in by_id[task_id][split]:
                        if eval_graph(graph, x, y) != expected_value:
                            errors.append(
                                f"returned graph fails {split} rows at line {line_number}"
                            )
                            break
            if row.get("accepted") and graph is None:
                errors.append(
                    f"accepted row has no graph at line {line_number}"
                )
            work = row.get("work") or {}
            observed_count = len(by_id[task_id]["train"])
            feature_builds = search.get("feature_builds", 0)
            if engine == "baseline":
                expected_work = {
                    "reference_feature_target_range_elements_scanned": (
                        4 * observed_count * feature_builds
                    ),
                    "reference_feature_operand_elements_scanned": (
                        4 * observed_count * feature_builds
                    ),
                    "reference_gap_elements_scanned": None,
                }
            else:
                cache_hits = search.get("feature_stat_cache_hits", 0)
                cache_misses = search.get("feature_stat_cache_misses", 0)
                cache_lookups = search.get("feature_stat_cache_lookups", 0)
                expected_work = {
                    "feature_target_range_elements_scanned": (
                        2 * observed_count if feature_builds else 0
                    ),
                    "feature_operand_elements_scanned": (
                        2 * observed_count * cache_misses
                    ),
                    "feature_stat_cache_lookups": cache_lookups,
                    "feature_stat_cache_hits": cache_hits,
                    "feature_stat_cache_misses": cache_misses,
                    "reference_feature_target_range_elements_scanned": (
                        4 * observed_count * feature_builds
                    ),
                    "reference_feature_operand_elements_scanned": (
                        4 * observed_count * feature_builds
                    ),
                    "reference_gap_elements_scanned": (
                        observed_count * (cache_lookups - 2 * feature_builds)
                    ),
                }
                if cache_lookups != cache_hits + cache_misses:
                    errors.append(
                        f"cache lookup partition mismatch at line {line_number}"
                    )
            for field, expected_value in expected_work.items():
                if work.get(field) != expected_value:
                    errors.append(
                        f"{field} work invariant failed at line {line_number}: "
                        f"expected {expected_value}, got {work.get(field)}"
                    )
        except Exception as exc:
            errors.append(
                f"malformed row {line_number}: {type(exc).__name__}: {exc}"
            )
    if set(rows) != expected:
        errors.append(
            f"grid incomplete: observed {len(rows)}, expected {len(expected)}"
        )
    training_seconds = 0
    for seed in training.get("seeds", []):
        training_seconds += int(seed.get("training_group_cpu_ns", 0))
        training_seconds += sum(
            int(value.get("cpu_ns", 0))
            for value in seed.get("control_construction", {}).values()
        )
        training_seconds += sum(
            int(value.get("fit_cpu_ns", 0))
            for value in seed.get("controls", {}).values()
        )
    model_states_frozen = all(
        seed.get("all_model_states_frozen") is True
        for seed in training.get("seeds", [])
    )
    if not model_states_frozen:
        errors.append(
            "training report does not attest frozen post-evaluation states"
        )
    details = {
        "directory": str(directory.resolve()),
        "row_count": len(rows),
        "raw_sha256": sha256(raw),
        "training_sha256": sha256(training_path.read_bytes()),
        "manifest_sha256": sha256(manifest_path.read_bytes()),
        "errors": errors,
        "whole_run_cpu_ns": manifest.get(
            "total_cpu_ns_through_raw_and_summary_output"
        ),
        "whole_run_wall_ns": manifest.get(
            "total_wall_ns_through_raw_and_summary_output"
        ),
        "optimization_supported": manifest.get("optimization_supported"),
        "training_setup_and_fit_cpu_ns": training_seconds,
        "training": training,
        "aggregate": aggregate,
    }
    for name in ("whole_run_cpu_ns", "whole_run_wall_ns"):
        value = details[name]
        if type(value) is not int or value < 0:
            errors.append(f"manifest missing or has invalid {name}")
    return rows, details


def audit(directory: Path, output: Path) -> dict[str, Any]:
    protocol_bytes = PROTOCOL.read_bytes()
    corpus_bytes = CORPUS.read_bytes()
    protocol, corpus = json.loads(protocol_bytes), json.loads(corpus_bytes)
    protocol_hash, corpus_hash = sha256(protocol_bytes), sha256(corpus_bytes)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(
            "total-work protocol bytes differ from the pinned audit hash"
        )
    if corpus_hash != EXPECTED_CORPUS_SHA256:
        raise ValueError(
            "benchmark corpus bytes differ from the pinned audit hash"
        )
    rows, bundle = load_bundle(
        directory, corpus, protocol_hash, corpus_hash, protocol
    )
    errors = list(bundle["errors"])
    tasks = corpus["splits"]["eval"]
    expected_pairs = {
        (repeat, seed, control, task["id"], budget)
        for repeat in REPEATS
        for seed in SEEDS
        for control in CONTROLS
        for task in tasks
        for budget in BUDGETS
    }
    paired_count = 0
    outcome_mismatches = []
    root_count_mismatches = []
    counter_mismatches = []
    paired_attempt_cpu = defaultdict(lambda: defaultdict(list))
    paired_search_cpu = defaultdict(lambda: defaultdict(list))
    paired_attempt_wall = defaultdict(lambda: defaultdict(list))
    counter_totals = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    work_totals = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    by_repeat_cpu = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    matched_success = []
    for repeat, seed, control, task_id, budget in sorted(expected_pairs):
        base_key = (repeat, "baseline", seed, control, task_id, budget)
        opt_key = (repeat, "optimized", seed, control, task_id, budget)
        base = rows.get(base_key)
        opt = rows.get(opt_key)
        if base is None or opt is None:
            continue
        paired_count += 1
        for field in ("accepted", "solved"):
            if base.get(field) != opt.get(field):
                outcome_mismatches.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "control": control,
                        "task_id": task_id,
                        "budget": budget,
                        "field": field,
                    }
                )
        for field in ("train_valid", "held_out_valid", "exact_function_match"):
            if base.get("validation", {}).get(field) != opt.get(
                "validation", {}
            ).get(field):
                outcome_mismatches.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "control": control,
                        "task_id": task_id,
                        "budget": budget,
                        "field": "validation." + field,
                    }
                )
        bg, og = base.get("returned_graph"), opt.get("returned_graph")
        if (bg is None) != (og is None):
            outcome_mismatches.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "control": control,
                    "task_id": task_id,
                    "budget": budget,
                    "field": "graph_presence",
                }
            )
        elif bg is not None and (
            bg != og or graph_polynomial(bg) != graph_polynomial(og)
        ):
            outcome_mismatches.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "control": control,
                    "task_id": task_id,
                    "budget": budget,
                    "field": "returned_graph",
                }
            )
        if base.get("search", {}).get("signature_evaluations") != opt.get(
            "search", {}
        ).get("signature_evaluations"):
            root_count_mismatches.append(
                [repeat, seed, control, task_id, budget]
            )
        if opt.get("paired_behavior_matches_baseline") is not True:
            outcome_mismatches.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "control": control,
                    "task_id": task_id,
                    "budget": budget,
                    "field": "behavior_attestation",
                }
            )
        if base.get("behavior_fingerprint") != opt.get("behavior_fingerprint"):
            outcome_mismatches.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "control": control,
                    "task_id": task_id,
                    "budget": budget,
                    "field": "behavior_fingerprint",
                }
            )
        for counter in COUNTERS:
            left = (base.get("search") or {}).get(counter)
            right = (opt.get("search") or {}).get(counter)
            if left != right:
                counter_mismatches.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "control": control,
                        "task_id": task_id,
                        "budget": budget,
                        "counter": counter,
                        "baseline": left,
                        "optimized": right,
                    }
                )
        base_work = base.get("work") or {}
        opt_work = opt.get("work") or {}
        if set(base_work) != set(opt_work):
            counter_mismatches.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "control": control,
                    "task_id": task_id,
                    "budget": budget,
                    "counter": "work.keys",
                    "baseline": sorted(base_work),
                    "optimized": sorted(opt_work),
                }
            )
        shared_work = set(base_work) & set(opt_work)
        ignored_work = {
            "reference_feature_target_range_elements_scanned",
            "reference_feature_operand_elements_scanned",
            "reference_gap_elements_scanned",
            "feature_stat_cache_lookups",
            "feature_stat_cache_hits",
            "feature_stat_cache_misses",
            "feature_target_range_elements_scanned",
            "feature_operand_elements_scanned",
        }
        for counter in sorted(shared_work - ignored_work):
            left = base_work.get(counter)
            right = opt_work.get(counter)
            if left != right:
                counter_mismatches.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "control": control,
                        "task_id": task_id,
                        "budget": budget,
                        "counter": "work." + counter,
                        "baseline": left,
                        "optimized": right,
                    }
                )
        group = (control, budget)
        for name, row in (("baseline", base), ("optimized", opt)):
            paired_attempt_cpu[group][name].append(
                row["attempt_cpu_ns_including_hashes_and_validation"]
            )
            paired_search_cpu[group][name].append(row["search_cpu_ns"])
            paired_attempt_wall[group][name].append(
                row["attempt_wall_ns_including_hashes_and_validation"]
            )
            by_repeat_cpu[repeat][group][name].append(
                row["attempt_cpu_ns_including_hashes_and_validation"]
            )
            for counter in COUNTERS:
                counter_totals[group][name][counter] += int(
                    (row.get("search") or {}).get(counter, 0)
                )
            for counter, value in (row.get("work") or {}).items():
                if type(value) is int:
                    work_totals[group][name][counter] += value
    repeat_mismatches = []
    for seed in SEEDS:
        for control in CONTROLS:
            for task in tasks:
                for budget in BUDGETS:
                    for engine in ENGINES:
                        key0 = (0, engine, seed, control, task["id"], budget)
                        key1 = (1, engine, seed, control, task["id"], budget)
                        first, second = rows.get(key0), rows.get(key1)
                        if first is None or second is None:
                            continue
                        if (
                            first.get("accepted"),
                            first.get("solved"),
                            first.get("returned_graph"),
                        ) != (
                            second.get("accepted"),
                            second.get("solved"),
                            second.get("returned_graph"),
                        ):
                            repeat_mismatches.append(
                                [engine, seed, control, task["id"], budget]
                            )
                        if first.get("behavior_fingerprint") != second.get(
                            "behavior_fingerprint"
                        ):
                            repeat_mismatches.append(
                                [
                                    engine,
                                    seed,
                                    control,
                                    task["id"],
                                    budget,
                                    "behavior_fingerprint",
                                ]
                            )
                        for counter in COUNTERS:
                            if (first.get("search") or {}).get(counter) != (
                                second.get("search") or {}
                            ).get(counter):
                                repeat_mismatches.append(
                                    [
                                        engine,
                                        seed,
                                        control,
                                        task["id"],
                                        budget,
                                        counter,
                                    ]
                                )
    cpu_summary = {}
    for group in sorted(paired_attempt_cpu):
        baseline = paired_attempt_cpu[group]["baseline"]
        optimized = paired_attempt_cpu[group]["optimized"]
        mean_base, mean_opt = (
            statistics.mean(baseline),
            statistics.mean(optimized),
        )
        cpu_summary[f"{group[0]}:{group[1]}"] = {
            "attempts_per_engine": len(baseline),
            "mean_attempt_cpu_ns": {
                "baseline": mean_base,
                "optimized": mean_opt,
            },
            "paired_mean_attempt_cpu_reduction": (mean_base - mean_opt)
            / mean_base
            if mean_base
            else None,
            "mean_search_cpu_ns": {
                "baseline": statistics.mean(
                    paired_search_cpu[group]["baseline"]
                ),
                "optimized": statistics.mean(
                    paired_search_cpu[group]["optimized"]
                ),
            },
            "mean_attempt_wall_ns": {
                "baseline": statistics.mean(
                    paired_attempt_wall[group]["baseline"]
                ),
                "optimized": statistics.mean(
                    paired_attempt_wall[group]["optimized"]
                ),
            },
            "repeat_means_attempt_cpu_ns": {
                str(repeat): {
                    engine: statistics.mean(
                        by_repeat_cpu[repeat][group][engine]
                    )
                    for engine in ENGINES
                }
                for repeat in REPEATS
            },
            "counter_totals": counter_totals[group],
            "work_axis_totals": work_totals[group],
            "work_axis_deltas_optimized_minus_baseline": {
                key: work_totals[group]["optimized"].get(key, 0)
                - work_totals[group]["baseline"].get(key, 0)
                for key in (
                    set(work_totals[group]["baseline"])
                    | set(work_totals[group]["optimized"])
                )
            },
        }
    primary = {}
    primary_pass = True
    for control in LEARNED:
        stats = cpu_summary.get(f"{control}:8192")
        reduction = (
            stats["paired_mean_attempt_cpu_reduction"] if stats else None
        )
        passed = reduction is not None and reduction >= 0.20
        primary[control] = {
            "reduction": reduction,
            "minimum": 0.20,
            "passes": passed,
        }
        primary_pass = primary_pass and passed
    aggregate = bundle.get("aggregate", {})
    decision = protocol["decision_rule"]
    gate_ratios = {}
    paired_ratio_mismatches = []
    aggregate_ratios = {
        (item.get("control"), item.get("budget"), item.get("repeat")): item
        for item in aggregate.get("paired_cost_ratios", [])
    }
    for control in CONTROLS:
        for budget in BUDGETS:
            for repeat in (*REPEATS, None):
                sums = {
                    engine: sum(
                        row.get(
                            "attempt_cpu_ns_including_hashes_and_validation", 0
                        )
                        for key, row in rows.items()
                        if key[1] == engine
                        and key[3] == control
                        and key[5] == budget
                        and (repeat is None or key[0] == repeat)
                    )
                    for engine in ENGINES
                }
                ratio = (
                    sums["optimized"] / sums["baseline"]
                    if sums["baseline"]
                    else None
                )
                if budget == decision["primary_budget"]:
                    gate_ratios.setdefault(control, {})[str(repeat)] = ratio
                saved = aggregate_ratios.get((control, budget, repeat))
                if (
                    saved is None
                    or saved.get("baseline_cpu_ns") != sums["baseline"]
                    or saved.get("optimized_cpu_ns") != sums["optimized"]
                    or saved.get("optimized_over_baseline_cpu_ratio") != ratio
                ):
                    paired_ratio_mismatches.append(
                        {
                            "control": control,
                            "budget": budget,
                            "repeat": repeat,
                            "baseline_cpu_ns": sums["baseline"],
                            "optimized_cpu_ns": sums["optimized"],
                            "ratio": ratio,
                            "saved": saved,
                        }
                    )
    if paired_ratio_mismatches:
        errors.append("aggregate paired CPU ratios differ from raw rows")
    gate_passes = all(
        gate_ratios.get(control, {}).get("None") is not None
        and gate_ratios[control]["None"]
        <= 1.0 - decision["minimum_cpu_reduction"]
        for control in decision["required_controls"]
    )
    saved_gate = aggregate.get("optimization_gate", {})
    all_gate_ratios = {
        control: gate_ratios.get(control, {}).get("None")
        for control in CONTROLS
    }
    if (
        saved_gate.get("budget") != decision["primary_budget"]
        or saved_gate.get("minimum_cpu_reduction")
        != decision["minimum_cpu_reduction"]
        or saved_gate.get("required_controls") != decision["required_controls"]
        or saved_gate.get("paired_cpu_ratios") != all_gate_ratios
        or saved_gate.get("supported") is not gate_passes
    ):
        errors.append(
            "aggregate optimization gate differs from independent raw-row calculation"
        )
    if bundle.get("optimization_supported") is not saved_gate.get("supported"):
        errors.append(
            "manifest optimization result differs from aggregate gate"
        )
    matched_success = matched_success_costs(rows, expected_pairs)
    training_amortization = amortized_training_costs(
        rows,
        bundle.get("training", {}),
        protocol["amortization"]["task_counts"],
    )
    summary_mismatches = []
    saved_points = {
        (
            item.get("engine"),
            item.get("control"),
            item.get("budget"),
            item.get("repeat"),
        ): item
        for item in aggregate.get("points", [])
    }
    for engine in ENGINES:
        for control in CONTROLS:
            for budget in BUDGETS:
                for repeat in (*REPEATS, None):
                    selected = [
                        row
                        for key, row in rows.items()
                        if key[1] == engine
                        and key[3] == control
                        and key[5] == budget
                        and (repeat is None or key[0] == repeat)
                    ]
                    saved = saved_points.get((engine, control, budget, repeat))
                    expected_point = {
                        "attempts": len(selected),
                        "accepted": sum(
                            bool(row.get("accepted")) for row in selected
                        ),
                        "success_fraction": (
                            sum(bool(row.get("accepted")) for row in selected)
                            / len(selected)
                            if selected
                            else None
                        ),
                        "attempt_cpu_ns_mean": (
                            statistics.mean(
                                row[
                                    "attempt_cpu_ns_including_hashes_and_validation"
                                ]
                                for row in selected
                            )
                            if selected
                            else None
                        ),
                        "attempt_wall_ns_mean": (
                            statistics.mean(
                                row[
                                    "attempt_wall_ns_including_hashes_and_validation"
                                ]
                                for row in selected
                            )
                            if selected
                            else None
                        ),
                        "censored_attempts": sum(
                            bool(row.get("censored")) for row in selected
                        ),
                        "work_totals_by_axis": sum_work(selected),
                    }
                    if saved is None or any(
                        (
                            not numeric_equal(saved.get(field), value)
                            if field
                            in {
                                "success_fraction",
                                "attempt_cpu_ns_mean",
                                "attempt_wall_ns_mean",
                            }
                            else saved.get(field) != value
                        )
                        for field, value in expected_point.items()
                    ):
                        summary_mismatches.append(
                            {
                                "kind": "point",
                                "engine": engine,
                                "control": control,
                                "budget": budget,
                                "repeat": repeat,
                                "expected": expected_point,
                                "saved": saved,
                            }
                        )
    saved_matched = {
        (
            item.get("engine"),
            item.get("control"),
            item.get("repeat"),
            item.get("target"),
        ): item
        for item in aggregate.get("matched_success_cpu_cost", [])
    }
    for item in matched_success:
        repeat = None if item["repeat"] == "pooled" else item["repeat"]
        key = (
            item["engine"],
            item["control"],
            repeat,
            item["target_success_fraction"],
        )
        saved = saved_matched.get(key)
        expected_matched = {
            "status": "censored" if item["censored"] else "observed",
            "budget": item["budget"],
            "censored_at_budget": max(BUDGETS) if item["censored"] else None,
            "success_fraction": item["attained_success_fraction"],
            "overshoot": (
                None
                if item["censored"]
                else item["attained_success_fraction"]
                - item["target_success_fraction"]
            ),
            "mean_attempt_cpu_ns": item["mean_attempt_cpu_ns"],
            "mean_attempt_wall_ns": item["mean_attempt_wall_ns"],
            "work_totals_by_axis": item["work_totals_by_axis"],
        }
        if saved is None or any(
            (
                not numeric_equal(saved.get(field), value)
                if field
                in {
                    "success_fraction",
                    "overshoot",
                    "mean_attempt_cpu_ns",
                    "mean_attempt_wall_ns",
                }
                else saved.get(field) != value
            )
            for field, value in expected_matched.items()
        ):
            summary_mismatches.append(
                {
                    "kind": "matched_success",
                    "key": key,
                    "expected": expected_matched,
                    "saved": saved,
                }
            )
    saved_amortization = {
        (
            item.get("engine"),
            item.get("control"),
            item.get("budget"),
            item.get("independent_task_count"),
        ): item
        for item in aggregate.get("amortization", [])
    }
    amortization_fields = (
        "training_cpu_ns_mean_seed_including_full_setup",
        "training_wall_ns_mean_seed_including_full_setup",
        "training_cpu_ns_per_task",
        "mean_attempt_cpu_ns",
        "mean_attempt_wall_ns",
        "amortized_cpu_ns_per_task",
        "amortized_wall_ns_per_task",
        "training_cost_included",
    )
    for item in training_amortization:
        key = (
            item["engine"],
            item["control"],
            item["budget"],
            item["independent_task_count"],
        )
        saved = saved_amortization.get(key)
        if saved is None or any(
            (
                not numeric_equal(saved.get(field), item.get(field))
                if field != "training_cost_included"
                else saved.get(field) is not item.get(field)
            )
            for field in amortization_fields
        ):
            summary_mismatches.append(
                {
                    "kind": "training_amortization",
                    "key": key,
                    "expected": {
                        field: item.get(field) for field in amortization_fields
                    },
                    "saved": saved,
                }
            )
    if summary_mismatches:
        errors.append(
            "aggregate points, matched-success costs, or amortization differ from raw-row recalculation"
        )
    report = {
        "auditor": "total-work-audit-v2",
        "protocol_sha256": protocol_hash,
        "corpus_sha256": corpus_hash,
        "expected_rows": len(expected_keys(tasks)),
        "observed_rows": len(rows),
        "paired_cells": paired_count,
        "bundle": {
            key: value for key, value in bundle.items() if key != "training"
        },
        "paired_outcome_equivalent": not outcome_mismatches,
        "outcome_mismatches": outcome_mismatches,
        "root_signature_count_mismatches": root_count_mismatches,
        "shared_counter_mismatches": counter_mismatches,
        "repeat_determinism_mismatches": repeat_mismatches,
        "primary_attempt_cpu_gate": {
            "checks": primary,
            "passes_all": primary_pass,
        },
        "optimization_gate_passes": primary_pass,
        "independent_optimization_gate": {
            "budget": decision["primary_budget"],
            "minimum_cpu_reduction": decision["minimum_cpu_reduction"],
            "required_controls": decision["required_controls"],
            "paired_cpu_ratios": {
                control: gate_ratios.get(control, {}).get("None")
                for control in CONTROLS
            },
            "repeat_specific_cpu_ratios": {
                control: {
                    str(repeat): gate_ratios.get(control, {}).get(str(repeat))
                    for repeat in REPEATS
                }
                for control in decision["required_controls"]
            },
            "supported": gate_passes,
            "aggregate_ratio_mismatches": paired_ratio_mismatches,
        },
        "cpu_and_counter_summaries_by_control_budget": cpu_summary,
        "aggregate_summary_mismatches": summary_mismatches,
        "matched_success_costs": matched_success,
        "training_amortization": training_amortization,
        "valid": (
            not errors
            and not outcome_mismatches
            and not counter_mismatches
            and not repeat_mismatches
            and not root_count_mismatches
            and paired_count == len(expected_pairs)
        ),
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing audit report: {output}"
        )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def matched_success_costs(
    rows: dict[Any, dict[str, Any]],
    expected: set[tuple[int, int, str, str, int]],
) -> list[dict[str, Any]]:
    """Report censored cost to each frozen success target, by engine/control."""
    result = []
    for engine in ENGINES:
        for control in CONTROLS:
            for repeat in (0, 1, "pooled"):
                budget_cells = []
                for budget in BUDGETS:
                    selected = [
                        row
                        for key, row in rows.items()
                        if key[1] == engine
                        and key[3] == control
                        and key[5] == budget
                        and (repeat == "pooled" or key[0] == repeat)
                    ]
                    successes = sum(
                        bool(row.get("accepted")) for row in selected
                    )
                    cpu = sum(
                        row.get(
                            "attempt_cpu_ns_including_hashes_and_validation", 0
                        )
                        for row in selected
                    )
                    wall = sum(
                        row.get(
                            "attempt_wall_ns_including_hashes_and_validation",
                            0,
                        )
                        for row in selected
                    )
                    budget_cells.append(
                        {
                            "budget": budget,
                            "successes": successes,
                            "attempts": len(selected),
                            "cpu_ns": cpu,
                            "wall_ns": wall,
                            "work_totals_by_axis": sum_work(selected),
                        }
                    )
                for target in (0.25, 0.50, 0.75, 1.0):
                    hit = next(
                        (
                            cell
                            for cell in budget_cells
                            if cell["attempts"]
                            and cell["successes"] / cell["attempts"] >= target
                        ),
                        None,
                    )
                    result.append(
                        {
                            "engine": engine,
                            "control": control,
                            "repeat": repeat,
                            "target_success_fraction": target,
                            "budget": hit["budget"] if hit else None,
                            "attempts": hit["attempts"] if hit else None,
                            "successes": hit["successes"] if hit else None,
                            "attained_success_fraction": (
                                hit["successes"] / hit["attempts"]
                                if hit
                                else None
                            ),
                            "total_cpu_ns": hit["cpu_ns"] if hit else None,
                            "total_wall_ns": hit["wall_ns"] if hit else None,
                            "mean_attempt_cpu_ns": (
                                hit["cpu_ns"] / hit["attempts"]
                                if hit
                                else None
                            ),
                            "mean_attempt_wall_ns": (
                                hit["wall_ns"] / hit["attempts"]
                                if hit
                                else None
                            ),
                            "work_totals_by_axis": (
                                hit["work_totals_by_axis"] if hit else None
                            ),
                            "censored": hit is None,
                        }
                    )
    return result


def sum_work(rows: list[dict[str, Any]]) -> dict[str, int | None]:
    if not rows:
        return {}
    totals: dict[str, int | None] = {}
    for key in rows[0].get("work", {}):
        values = [
            row.get("work", {}).get(key)
            for row in rows
            if row.get("work", {}).get(key) is not None
        ]
        totals[key] = sum(values) if values else None
    return totals


def amortized_training_costs(
    rows: dict[Any, dict[str, Any]],
    training: dict[str, Any],
    task_counts: list[int],
) -> list[dict[str, Any]]:
    """Charge shared training-group setup once and seed-mean model work."""
    seeds = training.get("seeds", [])
    group_cpu = int(training.get("training_group_cpu_ns", 0))
    group_wall = int(training.get("training_group_wall_ns", 0))
    result = []
    for control in CONTROLS:
        setup_cpu = []
        setup_wall = []
        construction_cpu = []
        construction_wall = []
        for seed in seeds:
            constructor = seed.get("control_construction", {}).get(control, {})
            fit = seed.get("controls", {}).get(control, {})
            construction_cpu.append(int(constructor.get("cpu_ns", 0)))
            construction_wall.append(int(constructor.get("wall_ns", 0)))
            setup_cpu.append(
                int(constructor.get("cpu_ns", 0))
                + int(fit.get("fit_cpu_ns", 0))
            )
            setup_wall.append(
                int(constructor.get("wall_ns", 0))
                + int(fit.get("fit_wall_ns", 0))
            )
        is_learned = control in LEARNED
        model_cpu = statistics.mean(setup_cpu) if setup_cpu else 0.0
        model_wall = statistics.mean(setup_wall) if setup_wall else 0.0
        fixed_construction_cpu = (
            statistics.mean(construction_cpu) if construction_cpu else 0.0
        )
        fixed_construction_wall = (
            statistics.mean(construction_wall) if construction_wall else 0.0
        )
        for engine in ENGINES:
            for budget in BUDGETS:
                selected = [
                    row
                    for key, row in rows.items()
                    if key[1] == engine
                    and key[3] == control
                    and key[5] == budget
                ]
                attempt_cpu = (
                    statistics.mean(
                        [
                            row[
                                "attempt_cpu_ns_including_hashes_and_validation"
                            ]
                            for row in selected
                        ]
                    )
                    if selected
                    else None
                )
                attempt_wall = (
                    statistics.mean(
                        [
                            row[
                                "attempt_wall_ns_including_hashes_and_validation"
                            ]
                            for row in selected
                        ]
                    )
                    if selected
                    else None
                )
                for count in task_counts:
                    total_cpu = group_cpu + model_cpu if is_learned else 0
                    total_wall = group_wall + model_wall if is_learned else 0
                    result.append(
                        {
                            "engine": engine,
                            "control": control,
                            "budget": budget,
                            "independent_task_count": count,
                            "training_cpu_ns_mean_seed_including_full_setup": total_cpu,
                            "training_wall_ns_mean_seed_including_full_setup": total_wall,
                            "training_cpu_ns_per_task": total_cpu / count,
                            "training_cost_included": is_learned,
                            "fixed_control_model_construction_cpu_ns_mean_seed": (
                                None if is_learned else fixed_construction_cpu
                            ),
                            "fixed_control_model_construction_wall_ns_mean_seed": (
                                None if is_learned else fixed_construction_wall
                            ),
                            "mean_attempt_cpu_ns": attempt_cpu,
                            "mean_attempt_wall_ns": attempt_wall,
                            "amortized_cpu_ns_per_task": (
                                None
                                if attempt_cpu is None
                                else total_cpu / count + attempt_cpu
                            ),
                            "amortized_wall_ns_per_task": (
                                None
                                if attempt_wall is None
                                else total_wall / count + attempt_wall
                            ),
                        }
                    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.bundle, args.output)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "errors": report["errors"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
