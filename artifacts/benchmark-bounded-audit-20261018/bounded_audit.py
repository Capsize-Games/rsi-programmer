#!/usr/bin/env python3
"""Independent auditor for the bounded-candidate benchmark bundle.

This file deliberately imports no project runtime, trainer, runner, or summary
module. It checks captured rows against the frozen protocol and recomputes all
published aggregates from raw data.
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
PROTOCOL = ROOT / "fixtures/benchmark_bounded_candidates_v1.json"
CORPUS = ROOT / "fixtures/benchmark_pre_eval_corpus_v2.json"
PROTOCOL_SHA256 = (
    "c235994867676059311cc05ac876060da0aa1517a6e3fa53cd73b80fabd3472d"
)
CORPUS_SHA256 = (
    "a8fd6add90b34da732da8dc96f2c0fafcabc2f017a17cdef4b31303eec4ba921"
)
REPEATS = (0, 1)
SEEDS = (0, 1, 2)
RANKED_ENGINES = ("full_cached", "bounded")
RANKED_CONTROLS = ("heuristic", "trained_snn", "mlp", "recurrent")
COUNTERS = (
    "signature_evaluations",
    "tried",
    "generated_descriptors",
    "scored_descriptors",
    "feature_builds",
    "operand_vector_lookups",
    "operand_signature_evaluations",
    "retained_operand_vectors",
    "policy_batches",
    "policy_batch_ticks",
    "policy_row_steps",
    "snn_steps",
    "node_row_operations",
    "graph_validation_calls",
    "oracle_calls",
    "equivalent_alternatives_dropped",
    "equivalent_alternatives_retained",
)
CPU_FIELD = "attempt_cpu_ns_including_hashes_and_validation"
WALL_FIELD = "attempt_wall_ns_including_hashes_and_validation"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def close(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a is b
    if a is None or b is None:
        return a is b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if type(a) is int or type(b) is int:
        return type(a) is type(b) and a == b
    if type(a) is float and type(b) is float:
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
    return type(a) is type(b) and a == b


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
    left, right = polynomial(node.get("left")), polynomial(node.get("right"))
    if op in {"add", "sub"}:
        out = dict(left)
        sign = 1 if op == "add" else -1
        for power, value in right.items():
            out[power] = out.get(power, 0) + sign * value
    else:
        out: dict[tuple[int, int], int] = defaultdict(int)
        for (lx, ly), lv in left.items():
            for (rx, ry), rv in right.items():
                out[lx + rx, ly + ry] += lv * rv
    return {key: value for key, value in out.items() if value}


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
    for i, node in enumerate(graph["nodes"]):
        if (
            not isinstance(node, dict)
            or set(node) != {"op", "type", "args"}
            or node["type"] != "Int"
            or node["op"] not in {"add", "sub", "mul"}
        ):
            raise ValueError("unsupported graph node")
        args = node["args"]
        if not isinstance(args, list) or len(args) != 2:
            raise ValueError("graph node must have two arguments")
        operands = []
        for ref in args:
            if ref == "x":
                operands.append({(1, 0): 1})
            elif ref == "y":
                operands.append({(0, 1): 1})
            elif type(ref) is int and 0 <= ref < i:
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
                    out[lx + rx, ly + ry] += lv * rv
        values.append({key: value for key, value in out.items() if value})
    output = graph["output"]
    if output == "x":
        return {(1, 0): 1}
    if output == "y":
        return {(0, 1): 1}
    if type(output) is int and 0 <= output < len(values):
        return values[output]
    raise ValueError("invalid graph output")


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


def tree_ops(graph: dict[str, Any]) -> int:
    def count(ref: Any) -> int:
        if ref in ("x", "y"):
            return 0
        if type(ref) is not int or not 0 <= ref < len(graph["nodes"]):
            raise ValueError("invalid graph ref")
        return 1 + sum(count(child) for child in graph["nodes"][ref]["args"])

    return count(graph["output"])


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        row[k]
        for k in ("repeat", "engine", "seed", "control", "task_id", "budget")
    )


def expected_order(
    seed: int, task_id: str, budget: int, control: str, repeat: int
) -> list[str]:
    payload = f"81081:{seed}:{task_id}:{budget}:{control}".encode()
    first = RANKED_ENGINES[hashlib.sha256(payload).digest()[0] % 2]
    if repeat == 1:
        first = RANKED_ENGINES[1 - RANKED_ENGINES.index(first)]
    return [first, RANKED_ENGINES[1 - RANKED_ENGINES.index(first)]]


def expected_rows(
    tasks: list[dict[str, Any]], protocol: dict[str, Any]
) -> set[tuple[Any, ...]]:
    ev = protocol["evaluation"]
    keys = set()
    for seed in SEEDS:
        for task in tasks:
            for budget in ev["streaming_budgets"]:
                for repeat in REPEATS:
                    keys.add(
                        (
                            repeat,
                            "streaming",
                            seed,
                            "streaming",
                            task["id"],
                            budget,
                        )
                    )
            for control in RANKED_CONTROLS:
                for budget in ev["ranked_budgets"]:
                    for repeat in REPEATS:
                        for engine in RANKED_ENGINES:
                            keys.add(
                                (
                                    repeat,
                                    engine,
                                    seed,
                                    control,
                                    task["id"],
                                    budget,
                                )
                            )
    return keys


def expected_sequence(
    tasks: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[tuple[Any, ...]]:
    """Build the frozen timed write order, including adjacent engine pairs."""
    ev = protocol["evaluation"]
    sequence = []
    for seed in SEEDS:
        for task in tasks:
            for budget in ev["streaming_budgets"]:
                for repeat in REPEATS:
                    sequence.append(
                        (
                            repeat,
                            "streaming",
                            seed,
                            "streaming",
                            task["id"],
                            budget,
                        )
                    )
            for control in RANKED_CONTROLS:
                for budget in ev["ranked_budgets"]:
                    for repeat in REPEATS:
                        for engine in expected_order(
                            seed, task["id"], budget, control, repeat
                        ):
                            sequence.append(
                                (
                                    repeat,
                                    engine,
                                    seed,
                                    control,
                                    task["id"],
                                    budget,
                                )
                            )
    return sequence


def expected_memory_sequence(
    tasks: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[tuple[Any, ...]]:
    """Build memory write order from the protocol's mode/task/control plan."""
    sequence = []
    budget = protocol["memory"]["budget"]
    controls = [("streaming", "streaming")] + [
        (engine, control)
        for control in RANKED_CONTROLS
        for engine in RANKED_ENGINES
    ]
    for mode in protocol["memory"]["modes"]:
        for task in tasks:
            for engine, control in controls:
                sequence.append((mode, engine, 0, control, task["id"], budget))
    return sequence


def work_sum(rows: list[dict[str, Any]]) -> dict[str, int | None]:
    axes: set[str] = set()
    for row in rows:
        axes.update((row.get("work") or {}).keys())
    totals = {}
    for axis in sorted(axes):
        values = [(row.get("work") or {}).get(axis) for row in rows]
        totals[axis] = (
            None
            if any(value is None for value in values)
            else sum(int(value) for value in values)
        )
    return totals


def training_charge(
    training: dict[str, Any], control: str, axis: str
) -> float:
    if control not in {"trained_snn", "mlp", "recurrent"}:
        return 0.0
    seeds = training["seeds"]
    own = sum(
        seed["control_construction"][control][axis + "_ns"]
        + seed["controls"][control]["fit_" + axis + "_ns"]
        for seed in seeds
    ) / len(seeds)
    return training["training_group_" + axis + "_ns"] + own


def recompute_summary(
    rows: list[dict[str, Any]],
    training: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    ev = protocol["evaluation"]
    configs = [("streaming", "streaming", ev["streaming_budgets"])]
    configs += [
        (engine, control, ev["ranked_budgets"])
        for control in RANKED_CONTROLS
        for engine in RANKED_ENGINES
    ]
    points, matched, amort, ratios = [], [], [], []
    for engine, control, budgets in configs:
        cpu_charge = training_charge(training, control, "cpu")
        wall_charge = training_charge(training, control, "wall")
        for repeat in (0, 1, None):
            curve = []
            for budget in budgets:
                subset = [
                    row
                    for row in rows
                    if row["engine"] == engine
                    and row["control"] == control
                    and row["budget"] == budget
                    and (repeat is None or row["repeat"] == repeat)
                ]
                if not subset:
                    continue
                work = work_sum(subset)
                cpu = statistics.mean(row[CPU_FIELD] for row in subset)
                wall = statistics.mean(row[WALL_FIELD] for row in subset)
                count = sum(bool(row["accepted"]) for row in subset)
                point = {
                    "engine": engine,
                    "control": control,
                    "budget": budget,
                    "repeat": repeat,
                    "attempts": len(subset),
                    "accepted": count,
                    "success_fraction": count / len(subset),
                    "censored_attempts": sum(
                        bool(row.get("censored")) for row in subset
                    ),
                    "attempt_cpu_ns_mean": cpu,
                    "attempt_wall_ns_mean": wall,
                    "work_totals_by_axis": work,
                    "scan_amplification": (
                        work.get("scored_descriptors", 0)
                        / work.get("signature_evaluations", 0)
                        if work.get("signature_evaluations", 0)
                        else None
                    ),
                }
                points.append(point)
                curve.append(point)
                if repeat is None:
                    for n in protocol["amortization"]["task_counts"]:
                        amort.append(
                            {
                                "engine": engine,
                                "control": control,
                                "budget": budget,
                                "independent_task_count": n,
                                "mean_attempt_cpu_ns": cpu,
                                "mean_attempt_wall_ns": wall,
                                "training_cpu_ns_mean_seed_"
                                "including_full_setup": cpu_charge,
                                "training_wall_ns_mean_seed_"
                                "including_full_setup": wall_charge,
                                "amortized_cpu_ns_per_task": cpu
                                + cpu_charge / n,
                                "amortized_wall_ns_per_task": wall
                                + wall_charge / n,
                            }
                        )
            for target in protocol["aggregation"]["matched_success_targets"]:
                p = next(
                    (p for p in curve if p["success_fraction"] >= target), None
                )
                matched.append(
                    {
                        "engine": engine,
                        "control": control,
                        "repeat": repeat,
                        "target": target,
                        "status": "observed" if p else "censored",
                        "budget": p["budget"] if p else None,
                        "censored_at_budget": None if p else max(budgets),
                        "success_fraction": p["success_fraction"]
                        if p
                        else None,
                        "overshoot": p["success_fraction"] - target
                        if p
                        else None,
                        "mean_attempt_cpu_ns": p["attempt_cpu_ns_mean"]
                        if p
                        else None,
                        "mean_attempt_wall_ns": p["attempt_wall_ns_mean"]
                        if p
                        else None,
                        "work_totals_by_axis": p["work_totals_by_axis"]
                        if p
                        else None,
                        "training_amortized_cpu_ns_per_task_by_N": {
                            str(n): p["attempt_cpu_ns_mean"] + cpu_charge / n
                            for n in protocol["amortization"]["task_counts"]
                        }
                        if p
                        else {},
                        "training_amortized_wall_ns_per_task_by_N": {
                            str(n): p["attempt_wall_ns_mean"] + wall_charge / n
                            for n in protocol["amortization"]["task_counts"]
                        }
                        if p
                        else {},
                    }
                )
    for control in RANKED_CONTROLS:
        for budget in ev["ranked_budgets"]:
            for repeat in (0, 1, None):
                pair = {
                    p["engine"]: p
                    for p in points
                    if p["control"] == control
                    and p["budget"] == budget
                    and p["repeat"] == repeat
                }
                if set(pair) != set(RANKED_ENGINES):
                    continue
                entry: dict[str, Any] = {
                    "control": control,
                    "budget": budget,
                    "repeat": repeat,
                }
                for axis in ("cpu", "wall"):
                    field = f"attempt_{axis}_ns_mean"
                    for name in RANKED_ENGINES:
                        entry[f"{name}_{axis}_ns"] = (
                            pair[name][field] * pair[name]["attempts"]
                        )
                    entry[f"bounded_over_full_cached_{axis}_ratio"] = (
                        pair["bounded"][field] / pair["full_cached"][field]
                    )
                ratios.append(entry)
    by_full = {
        (p["engine"], p["control"]): p
        for p in matched
        if p["repeat"] is None and p["target"] == 1.0
    }
    decisions = []
    for control in protocol["decision_rule"]["required_controls"]:
        costs = {}
        for engine in RANKED_ENGINES:
            p = by_full[engine, control]
            costs[engine] = p["training_amortized_cpu_ns_per_task_by_N"].get(
                "48"
            )
        ratio = (
            costs["bounded"] / costs["full_cached"]
            if all(v is not None for v in costs.values())
            else None
        )
        decisions.append(
            {
                "control": control,
                "first_full_success_N48_cpu_ns": costs,
                "bounded_over_full_cached_cpu_ratio": ratio,
                "supported": ratio is not None and ratio <= 0.8,
            }
        )
    return {
        "points": points,
        "matched_success_cpu_cost": matched,
        "amortization": amort,
        "paired_cost_ratios": ratios,
        "decision_controls": decisions,
        "decision_supported_on_cost": all(d["supported"] for d in decisions),
    }


def compare_summary(
    saved: dict[str, Any], actual: dict[str, Any], errors: list[str]
) -> None:
    for name in (
        "points",
        "matched_success_cpu_cost",
        "amortization",
        "paired_cost_ratios",
        "decision_controls",
    ):
        left, right = saved.get(name), actual[name]
        if not isinstance(left, list) or len(left) != len(right):
            errors.append(f"aggregate {name} has wrong row count")
            continue
        for i, (a, b) in enumerate(zip(left, right)):
            if set(a) != set(b):
                errors.append(f"aggregate {name}[{i}] field set differs")
                continue
            for key in b:
                if not close(a[key], b[key]):
                    errors.append(f"aggregate {name}[{i}].{key} differs")
                    break
    if (
        saved.get("decision_supported_on_cost")
        is not actual["decision_supported_on_cost"]
    ):
        errors.append("aggregate decision_supported_on_cost differs")


def audit(bundle_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    protocol_bytes, corpus_bytes = PROTOCOL.read_bytes(), CORPUS.read_bytes()
    if sha256(protocol_bytes) != PROTOCOL_SHA256:
        raise ValueError("frozen bounded protocol hash mismatch")
    if sha256(corpus_bytes) != CORPUS_SHA256:
        raise ValueError("frozen evaluation corpus hash mismatch")
    protocol, corpus = json.loads(protocol_bytes), json.loads(corpus_bytes)
    paths = {
        name: bundle_dir / name
        for name in (
            "manifest.json",
            "raw.jsonl",
            "memory.jsonl",
            "training.json",
            "aggregate.json",
        )
    }
    for name, path in paths.items():
        if not path.is_file():
            errors.append(f"missing {name}")
    if errors:
        return {"bundle": str(bundle_dir.resolve()), "errors": errors}
    manifest = json.loads(paths["manifest.json"].read_text())
    training_bytes, raw_bytes, memory_bytes, aggregate_bytes = (
        paths[name].read_bytes()
        for name in (
            "training.json",
            "raw.jsonl",
            "memory.jsonl",
            "aggregate.json",
        )
    )
    training, aggregate = (
        json.loads(training_bytes),
        json.loads(aggregate_bytes),
    )
    expected_hashes = {
        "raw_jsonl_sha256": sha256(raw_bytes),
        "memory_jsonl_sha256": sha256(memory_bytes),
        "training_json_sha256": sha256(training_bytes),
        "aggregate_json_sha256": sha256(aggregate_bytes),
        "protocol_sha256": PROTOCOL_SHA256,
        "corpus_sha256": CORPUS_SHA256,
    }
    file_hashes = manifest.get("files_sha256", {})
    for filename, value in (
        ("raw.jsonl", expected_hashes["raw_jsonl_sha256"]),
        ("memory.jsonl", expected_hashes["memory_jsonl_sha256"]),
        ("training.json", expected_hashes["training_json_sha256"]),
        ("aggregate.json", expected_hashes["aggregate_json_sha256"]),
    ):
        if file_hashes.get(filename) != value:
            errors.append(f"manifest files_sha256[{filename}] mismatch")
    if manifest.get("protocol_sha256") != PROTOCOL_SHA256:
        errors.append("manifest protocol_sha256 mismatch")
    tasks = corpus["splits"]["eval"]
    task_by_id = {task["id"]: task for task in tasks}
    targets = {
        task["id"]: polynomial(task["generator"]["program"]) for task in tasks
    }
    expected = expected_rows(tasks, protocol)
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    raw_sequence: list[tuple[Any, ...]] = []
    for n, line in enumerate(raw_bytes.splitlines(), 1):
        try:
            row = json.loads(line)
            key = row_key(row)
            raw_sequence.append(key)
            if key in rows:
                errors.append(f"duplicate raw key at line {n}")
                continue
            rows[key] = row
            if key not in expected:
                errors.append(f"unexpected raw row key at line {n}")
            repeat, engine, seed, control, task_id, budget = key
            if row.get("protocol_id") != protocol["protocol_id"]:
                errors.append(f"protocol ID mismatch at line {n}")
            if row.get("corpus_sha256") != CORPUS_SHA256:
                errors.append(f"corpus hash mismatch at line {n}")
            if row.get("error"):
                errors.append(f"runner error at line {n}: {row['error']}")
            if row.get("state_frozen_during_eval") is not True:
                errors.append(f"model state not frozen at line {n}")
            expected_weight = protocol["model_reference"][
                "expected_weights_after_fit"
            ][str(seed)][control]
            if row.get("expected_weights_after_fit_sha256") != expected_weight:
                errors.append(
                    f"expected weight reference mismatch at line {n}"
                )
            if row.get("weights_before_eval_sha256") != expected_weight:
                errors.append(f"pre-evaluation weights mismatch at line {n}")
            if row.get("weights_match_after_fit_reference") is not True:
                errors.append(f"weight checkpoint mismatch at line {n}")
            if row.get("weights_before_eval_sha256") != row.get(
                "weights_after_eval_sha256"
            ):
                errors.append(f"weights changed during attempt at line {n}")
            if row.get("state_fingerprint_before_eval") != row.get(
                "state_fingerprint_after_eval"
            ):
                errors.append(
                    f"policy state changed during attempt at line {n}"
                )
            if engine == "streaming":
                expected_engine_order = ["streaming"]
            else:
                expected_engine_order = expected_order(
                    seed, task_id, budget, control, repeat
                )
            if row.get("engine_order") != expected_engine_order:
                errors.append(f"engine order mismatch at line {n}")
            for field in (
                CPU_FIELD,
                WALL_FIELD,
                "search_cpu_ns",
                "search_wall_ns",
            ):
                if type(row.get(field)) is not int or row[field] < 0:
                    errors.append(f"invalid {field} at line {n}")
            search, work = row.get("search") or {}, row.get("work") or {}
            if type(row.get("solved")) is not bool or row.get(
                "solved"
            ) is not search.get("solved"):
                errors.append(f"solved flag differs from search at line {n}")
            if type(row.get("censored")) is not bool or row.get(
                "censored"
            ) is not bool(
                search.get("budget_exhausted")
                or search.get("signature_budget_exhausted")
            ):
                errors.append(f"censored flag differs from search at line {n}")
            for counter in COUNTERS:
                if type(search.get(counter)) is not int or search[counter] < 0:
                    errors.append(
                        f"invalid/missing search.{counter} at line {n}"
                    )
            for axis, value in work.items():
                if value is not None and (type(value) is not int or value < 0):
                    errors.append(f"invalid work.{axis} at line {n}")
            cap = (
                protocol["evaluation"]["streaming_budgets"]
                if engine == "streaming"
                else protocol["evaluation"]["ranked_budgets"]
            )
            if budget not in cap:
                errors.append(f"unexpected budget at line {n}")
            if search.get("signature_evaluations", 0) > budget:
                errors.append(f"root signature budget exceeded at line {n}")
            if row.get("root_calls") != search.get("signature_evaluations"):
                errors.append(f"root_calls mismatch at line {n}")
            if engine == "bounded":
                method = protocol["method"]
                if search.get("window_size") != method["window_size"]:
                    errors.append(f"bounded window mismatch at line {n}")
                descriptor_peak = search.get("peak_live_descriptors")
                feature_peak = search.get("peak_live_features")
                if type(descriptor_peak) is not int or descriptor_peak < 0:
                    errors.append(
                        "bounded descriptor peak missing or invalid at "
                        f"line {n}"
                    )
                elif descriptor_peak > method["window_size"]:
                    errors.append(
                        f"bounded descriptor peak exceeded at line {n}"
                    )
                if type(feature_peak) is not int or feature_peak < 0:
                    errors.append(
                        f"bounded feature peak missing or invalid at line {n}"
                    )
                elif feature_peak > method["window_size"]:
                    errors.append(f"bounded feature peak exceeded at line {n}")
                if work.get("scored_descriptors") != search.get(
                    "scored_descriptors"
                ):
                    errors.append(
                        f"scored descriptor counter mismatch at line {n}"
                    )
                if work.get("scored_unexecuted_descriptors") != search.get(
                    "scored_unexecuted_descriptors"
                ):
                    errors.append(
                        f"unexecuted descriptor counter mismatch at line {n}"
                    )
                if work.get("peak_live_descriptors") != search.get(
                    "peak_live_descriptors"
                ):
                    errors.append(
                        f"descriptor peak counter mismatch at line {n}"
                    )
                if work.get("peak_live_features") != search.get(
                    "peak_live_features"
                ):
                    errors.append(f"feature peak counter mismatch at line {n}")
                if search.get("signature_evaluations") != work.get(
                    "signature_evaluations"
                ):
                    errors.append(
                        f"bounded signature counter mismatch at line {n}"
                    )
                if search.get("scored_descriptors", 0) < search.get(
                    "signature_evaluations", 0
                ):
                    errors.append(
                        "scored descriptors below executed signatures at "
                        f"line {n}"
                    )
                if search.get("scored_unexecuted_descriptors") != (
                    search.get("scored_descriptors", 0)
                    - search.get("signature_evaluations", 0)
                ):
                    errors.append(
                        f"scored/executed partition mismatch at line {n}"
                    )
                if search.get("feature_builds") != search.get(
                    "scored_descriptors"
                ):
                    errors.append(
                        f"feature/scoring count mismatch at line {n}"
                    )
                if work.get("skipped_descriptors") != search.get(
                    "skipped_descriptors"
                ):
                    errors.append(
                        f"skipped descriptor counter mismatch at line {n}"
                    )
                if sum(
                    search.get("layer_generated_descriptors", [])
                ) != search.get("generated_descriptors"):
                    errors.append(
                        f"bounded generated layer totals mismatch at line {n}"
                    )
                if sum(
                    search.get("layer_executed_candidates", [])
                ) != search.get("signature_evaluations"):
                    errors.append(
                        f"bounded execution layer totals mismatch at line {n}"
                    )
                if not (
                    len(search.get("layer_widths", []))
                    == len(search.get("layer_generated_descriptors", []))
                    == len(search.get("layer_executed_candidates", []))
                ):
                    errors.append(
                        f"bounded layer ledger lengths differ at line {n}"
                    )
                for width, generated, executed in zip(
                    search.get("layer_widths", []),
                    search.get("layer_generated_descriptors", []),
                    search.get("layer_executed_candidates", []),
                ):
                    if not (
                        0 <= generated <= width and 0 <= executed <= generated
                    ):
                        errors.append(
                            f"bounded layer ledger violates width at line {n}"
                        )
            graph = row.get("returned_graph")
            graph_checks = {
                "train_valid": False,
                "held_out_valid": False,
                "exact_function_match": False,
            }
            within_bound = False
            if graph is not None:
                try:
                    graph_checks["exact_function_match"] = (
                        graph_polynomial(graph) == targets[task_id]
                    )
                except (TypeError, ValueError, KeyError, IndexError):
                    pass
                for split, flag in (
                    ("train", "train_valid"),
                    ("held_out", "held_out_valid"),
                ):
                    try:
                        graph_checks[flag] = all(
                            eval_graph(graph, x, y) == value
                            for x, y, value in task_by_id[task_id][split]
                        )
                    except (TypeError, ValueError, KeyError, IndexError):
                        graph_checks[flag] = False
                try:
                    within_bound = (
                        tree_ops(graph) <= protocol["evaluation"]["tree_bound"]
                    )
                except (TypeError, ValueError, KeyError, IndexError):
                    within_bound = False
            reported_checks = {
                flag: row.get("validation", {}).get(flag)
                for flag in graph_checks
            }
            if any(
                type(reported_checks[flag]) is not bool
                or reported_checks[flag] is not graph_checks[flag]
                for flag in graph_checks
            ):
                errors.append(
                    "reported graph checks differ from independent checks at "
                    f"line {n}"
                )
            expected_accept = bool(
                row.get("solved")
                and graph is not None
                and within_bound
                and all(graph_checks.values())
            )
            if (
                type(row.get("accepted")) is not bool
                or row.get("accepted") is not expected_accept
            ):
                errors.append(
                    f"accepted flag inconsistent with checks at line {n}"
                )
            if row.get("accepted") and graph is None:
                errors.append(f"accepted row lacks graph at line {n}")
            fingerprint = row.get("behavior_fingerprint")
            if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                errors.append(f"invalid behavior fingerprint at line {n}")
        except Exception as exc:
            errors.append(
                f"malformed raw row {n}: {type(exc).__name__}: {exc}"
            )
    scheduled_raw = expected_sequence(tasks, protocol)
    if raw_sequence != scheduled_raw:
        mismatch = next(
            (
                i
                for i, (actual, scheduled) in enumerate(
                    zip(raw_sequence, scheduled_raw)
                )
                if actual != scheduled
            ),
            min(len(raw_sequence), len(scheduled_raw)),
        )
        errors.append(
            f"raw row order differs from frozen schedule at index {mismatch}"
        )
    if set(rows) != expected:
        errors.append(
            f"raw grid incomplete: {len(rows)} observed, "
            f"{len(expected)} expected"
        )
    if manifest.get("row_count") != len(expected) or manifest.get(
        "expected_row_count"
    ) != len(expected):
        errors.append("manifest timed row count mismatch")

    # Repeat determinism is checked independently within each engine.
    def deterministic_search(search: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in search.items()
            if not key.endswith("_time_seconds") and key != "elapsed_seconds"
        }

    deterministic_fields = (
        "accepted",
        "solved",
        "returned_graph",
        "behavior_fingerprint",
        "work",
        "state_fingerprint_before_eval",
        "state_fingerprint_after_eval",
        "weights_before_eval_sha256",
        "weights_after_eval_sha256",
    )
    repeat_mismatches = []
    for engine, control, budgets in [
        ("streaming", "streaming", protocol["evaluation"]["streaming_budgets"])
    ] + [
        (engine, control, protocol["evaluation"]["ranked_budgets"])
        for control in RANKED_CONTROLS
        for engine in RANKED_ENGINES
    ]:
        for seed in SEEDS:
            for task in tasks:
                for budget in budgets:
                    a = rows.get(
                        (0, engine, seed, control, task["id"], budget)
                    )
                    b = rows.get(
                        (1, engine, seed, control, task["id"], budget)
                    )
                    if a is None or b is None:
                        continue
                    for field in deterministic_fields:
                        if a.get(field) != b.get(field):
                            repeat_mismatches.append(
                                [
                                    engine,
                                    seed,
                                    control,
                                    task["id"],
                                    budget,
                                    field,
                                ]
                            )
                    if deterministic_search(
                        a.get("search", {})
                    ) != deterministic_search(b.get("search", {})):
                        repeat_mismatches.append(
                            [
                                engine,
                                seed,
                                control,
                                task["id"],
                                budget,
                                "search",
                            ]
                        )
                    flags = (
                        "train_valid",
                        "held_out_valid",
                        "exact_function_match",
                        "validation_calls",
                        "node_row_operations",
                        "error",
                    )
                    if {f: a.get("validation", {}).get(f) for f in flags} != {
                        f: b.get("validation", {}).get(f) for f in flags
                    }:
                        repeat_mismatches.append(
                            [
                                engine,
                                seed,
                                control,
                                task["id"],
                                budget,
                                "validation",
                            ]
                        )
    if repeat_mismatches:
        errors.append(
            f"repeat deterministic mismatches: {len(repeat_mismatches)}"
        )
    # Verify the two isolated memory passes against timed repeat-zero behavior.
    memory_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    memory_sequence: list[tuple[Any, ...]] = []
    modes = protocol["memory"]["modes"]
    for n, line in enumerate(memory_bytes.splitlines(), 1):
        try:
            row = json.loads(line)
            key = (
                row["memory_mode"],
                row["engine"],
                row["seed"],
                row["control"],
                row["task_id"],
                row["budget"],
            )
            memory_sequence.append(key)
            if key in memory_rows:
                errors.append(f"duplicate memory key at line {n}")
                continue
            memory_rows[key] = row
            if (
                row.get("repeat") != 0
                or row.get("protocol_id") != protocol["protocol_id"]
            ):
                errors.append(f"memory row key metadata mismatch at line {n}")
            if row.get("error"):
                errors.append(
                    f"memory child error at line {n}: {row['error']}"
                )
            if row.get("instrumented_timing_not_used") is not True:
                errors.append(f"memory row not marked non-timing at line {n}")
            for field in (
                "peak_rss_kib",
                "initial_rss_kib",
                "peak_rss_delta_kib",
            ):
                if type(row.get(field)) is not int or row[field] < 0:
                    errors.append(f"invalid memory {field} at line {n}")
            if row.get("peak_rss_delta_kib") != row.get(
                "peak_rss_kib", 0
            ) - row.get("initial_rss_kib", 0):
                errors.append(f"memory RSS delta mismatch at line {n}")
            if row.get("memory_mode") == "tracemalloc":
                for field in ("traced_peak_bytes", "traced_current_bytes"):
                    if type(row.get(field)) is not int or row[field] < 0:
                        errors.append(f"invalid traced {field} at line {n}")
                if row.get("traced_current_bytes", 0) > row.get(
                    "traced_peak_bytes", 0
                ):
                    errors.append(f"traced current exceeds peak at line {n}")
            elif row.get("memory_mode") == "rss":
                if (
                    row.get("traced_peak_bytes") is not None
                    or row.get("traced_current_bytes") is not None
                ):
                    errors.append(
                        f"RSS row unexpectedly has traced values at line {n}"
                    )
        except Exception as exc:
            errors.append(
                f"malformed memory row {n}: {type(exc).__name__}: {exc}"
            )
    mem_expected = set()
    for mode in modes:
        for engine, control, _budgets in [("streaming", "streaming", [])] + [
            (engine, control, [])
            for control in RANKED_CONTROLS
            for engine in RANKED_ENGINES
        ]:
            for task in tasks:
                mem_expected.add(
                    (
                        mode,
                        engine,
                        0,
                        control,
                        task["id"],
                        protocol["memory"]["budget"],
                    )
                )
    scheduled_memory = expected_memory_sequence(tasks, protocol)
    if memory_sequence != scheduled_memory:
        mismatch = next(
            (
                i
                for i, (actual, scheduled) in enumerate(
                    zip(memory_sequence, scheduled_memory)
                )
                if actual != scheduled
            ),
            min(len(memory_sequence), len(scheduled_memory)),
        )
        errors.append(
            "memory row order differs from frozen schedule at "
            f"index {mismatch}"
        )
    if set(memory_rows) != mem_expected:
        errors.append(
            f"memory grid mismatch: {len(memory_rows)} observed, "
            f"{len(mem_expected)} expected"
        )
    memory_mismatches = []
    for key, mem in memory_rows.items():
        mode, engine, seed, control, task_id, budget = key
        timed = rows.get((0, engine, seed, control, task_id, budget))
        if timed is None:
            memory_mismatches.append([key, "missing_timed_cell"])
            continue
        for field in (
            "behavior_fingerprint",
            "accepted",
            "solved",
            "returned_graph",
            "work",
            "state_fingerprint_before_eval",
            "state_fingerprint_after_eval",
            "weights_before_eval_sha256",
            "weights_after_eval_sha256",
        ):
            if mem.get(field) != timed.get(field):
                memory_mismatches.append([key, field])
        if deterministic_search(mem.get("search", {})) != deterministic_search(
            timed.get("search", {})
        ):
            memory_mismatches.append([key, "search"])
        flags = (
            "train_valid",
            "held_out_valid",
            "exact_function_match",
            "validation_calls",
            "node_row_operations",
            "error",
        )
        if {f: mem.get("validation", {}).get(f) for f in flags} != {
            f: timed.get("validation", {}).get(f) for f in flags
        }:
            memory_mismatches.append([key, "validation"])
    if memory_mismatches:
        errors.append(f"memory behavior mismatches: {len(memory_mismatches)}")
    if manifest.get("memory_row_count") != len(mem_expected):
        errors.append("manifest memory row count mismatch")
    # Frozen training hashes and state assertions.
    expected_weights = protocol["model_reference"][
        "expected_weights_after_fit"
    ]
    reference_path = ROOT / protocol["model_reference"]["path"]
    if (
        not reference_path.is_file()
        or sha256(reference_path.read_bytes())
        != protocol["model_reference"]["sha256"]
    ):
        errors.append("frozen model reference file hash mismatch")
    if {str(seed.get("seed")) for seed in training.get("seeds", [])} != {
        "0",
        "1",
        "2",
    }:
        errors.append("training seeds differ from frozen seeds")
    for seed in training.get("seeds", []):
        seed_id = str(seed.get("seed"))
        for control, digest in expected_weights.get(seed_id, {}).items():
            actual = (
                seed.get("controls", {})
                .get(control, {})
                .get("weights_after_fit_sha256")
            )
            if actual != digest:
                errors.append(
                    f"post-fit weights mismatch seed={seed_id} "
                    f"control={control}"
                )
        if seed.get("all_model_states_frozen") is not True:
            errors.append(f"model states not frozen seed={seed_id}")
        for control, state in seed.get(
            "state_after_all_evaluation", {}
        ).items():
            if (
                state.get("state_matches_after_fit") is not True
                or state.get("weights_match_after_fit") is not True
            ):
                errors.append(
                    f"model state changed after evaluation seed={seed_id} "
                    f"control={control}"
                )
    if training.get("status") != "complete":
        errors.append("training record is incomplete")
    # Recompute published summaries without importing production code.
    recomputed = recompute_summary(list(rows.values()), training, protocol)
    compare_summary(aggregate, recomputed, errors)
    if (
        aggregate.get("decision_supported_on_cost")
        is not recomputed["decision_supported_on_cost"]
    ):
        errors.append(
            "aggregate decision_supported_on_cost differs from "
            "raw-row calculation"
        )
    should_support = (
        manifest.get("infrastructure_valid") is True
        and recomputed["decision_supported_on_cost"]
    )
    if aggregate.get("decision_supported") is not should_support:
        errors.append(
            "aggregate decision_supported differs from audited result"
        )
    if manifest.get("decision_supported") is not should_support:
        errors.append(
            "manifest decision_supported differs from audited result"
        )
    full_success = {
        (item["engine"], item["control"]): item
        for item in recomputed["matched_success_cpu_cost"]
        if item["repeat"] is None and item["target"] == 1.0
    }
    required_snn = protocol["decision_rule"]["required_controls"]
    all_snn_success = (
        all(
            full_success[("bounded", control)]["status"] == "observed"
            for control in required_snn
        )
        and full_success[("streaming", "streaming")]["status"] == "observed"
    )
    snn_cost = full_success[("bounded", "trained_snn")][
        "training_amortized_cpu_ns_per_task_by_N"
    ].get("48")
    comparators = {
        "bounded_mlp": full_success[("bounded", "mlp")][
            "training_amortized_cpu_ns_per_task_by_N"
        ].get("48"),
        "bounded_recurrent": full_success[("bounded", "recurrent")][
            "training_amortized_cpu_ns_per_task_by_N"
        ].get("48"),
        "streaming": full_success[("streaming", "streaming")][
            "training_amortized_cpu_ns_per_task_by_N"
        ].get("48"),
    }
    snn_ratios = {
        name: (snn_cost / value if snn_cost is not None and value else None)
        for name, value in comparators.items()
    }
    snn_claim_supported = (
        all_snn_success
        and snn_cost is not None
        and all(
            value is not None and snn_cost <= 0.90 * value
            for value in comparators.values()
        )
    )
    if (
        aggregate.get("snn_efficiency_supported_on_cost")
        is not snn_claim_supported
    ):
        errors.append("aggregate SNN efficiency cost claim differs from audit")
    # Check source pins and before/after integrity where provided.
    before = manifest.get("before", {}).get("source_sha256", {})
    after = manifest.get("after", {}).get("source_sha256", {})
    if before != after:
        errors.append("before/after source hashes differ")
    legacy_protocol = ROOT / "fixtures/benchmark_total_work_v1.json"
    legacy_protocol_sha = (
        sha256(legacy_protocol.read_bytes())
        if legacy_protocol.is_file()
        else None
    )
    if before.get("protocol_sha256") != legacy_protocol_sha:
        errors.append("source snapshot reference protocol hash differs")
    if before.get("corpus_sha256") != CORPUS_SHA256:
        errors.append("source snapshot input hashes differ from frozen inputs")
    if (
        before.get("baseline_source_sha256")
        != protocol["baseline_source"]["sha256"]
    ):
        errors.append("source snapshot baseline hash differs from protocol")
    if (
        before.get("model_reference_sha256")
        != protocol["model_reference"]["sha256"]
    ):
        errors.append(
            "source snapshot model reference hash differs from protocol"
        )
    source_files = before.get("source_files_sha256", {})
    if not source_files:
        errors.append("source snapshot has no source_files_sha256")
    for name, digest in source_files.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        try:
            current = sha256(path.read_bytes())
        except OSError as exc:
            errors.append(f"cannot read source snapshot member {name}: {exc}")
        else:
            if current != digest:
                errors.append(f"source snapshot member changed: {name}")
    if manifest.get("corpus_sha256") not in (None, CORPUS_SHA256):
        errors.append("manifest corpus hash differs")
    for name, expected_digest in (
        (
            protocol["baseline_source"]["path"],
            protocol["baseline_source"]["sha256"],
        ),
        (
            "rsi_programmer/pre_eval_features.py",
            protocol["baseline_source"]["helper_sha256"][
                "rsi_programmer/pre_eval_features.py"
            ],
        ),
    ):
        source = ROOT / name
        if (
            not source.is_file()
            or sha256(source.read_bytes()) != expected_digest
        ):
            errors.append(f"pinned source changed or missing: {name}")
    if (
        manifest.get("infrastructure_valid") is not True
        or manifest.get("source_frozen") is not True
    ):
        errors.append("manifest reports infrastructure/source failure")
    if manifest.get("before", {}).get("environment") != manifest.get(
        "after", {}
    ).get("environment"):
        errors.append("before/after environment differs")
    for field in (
        "grid_valid",
        "memory_valid",
        "repeats_behavior_identical",
        "model_states_frozen",
        "no_row_errors",
        "source_frozen",
        "environment_frozen",
        "bounded_peak_valid",
        "root_cap_valid",
        "infrastructure_valid",
    ):
        if manifest.get(field) is not True:
            errors.append(f"manifest does not attest {field}")
    if aggregate.get("infrastructure_valid") is not manifest.get(
        "infrastructure_valid"
    ):
        errors.append("aggregate infrastructure_valid differs from manifest")
    result = {
        "bundle": str(bundle_dir.resolve()),
        "protocol_sha256": PROTOCOL_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "row_count": len(rows),
        "expected_row_count": len(expected),
        "memory_row_count": len(memory_rows),
        "expected_memory_row_count": len(mem_expected),
        "repeat_mismatch_count": len(repeat_mismatches),
        "memory_mismatch_count": len(memory_mismatches),
        "raw_sha256": sha256(raw_bytes),
        "memory_sha256": sha256(memory_bytes),
        "aggregate_sha256": sha256(aggregate_bytes),
        "errors": errors,
        "audit_passed": not errors,
        "decision_supported_on_cost": recomputed["decision_supported_on_cost"],
        "snn_efficiency_advantage_supported": snn_claim_supported,
        "snn_full_success_all_required": all_snn_success,
        "snn_over_comparator_n48_cpu_ratios": snn_ratios,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("audit_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
