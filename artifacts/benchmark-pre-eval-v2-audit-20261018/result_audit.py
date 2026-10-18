#!/usr/bin/env python3
"""Independently audit one completed benchmark v2 result bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import audit as input_audit

CONTROL_NAMES = (
    "streaming", "heuristic", "seeded_random", "untrained_snn",
    "trained_snn", "mlp", "recurrent",
)
SEEDS = (0, 1, 2)
OP_ORDER = ("add", "sub", "mul")


def sha256(data: bytes) -> str:
    """Hash bytes with SHA256."""
    return hashlib.sha256(data).hexdigest()


def same(actual: Any, expected: Any) -> bool:
    """Compare nested JSON values, allowing tiny float serialization slack."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected),
                            rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(actual, list) and isinstance(expected, list):
        return (len(actual) == len(expected)
                and all(same(a, e) for a, e in zip(actual, expected)))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return (actual.keys() == expected.keys()
                and all(same(actual[k], expected[k]) for k in actual))
    return actual == expected


def polynomial_add(left: dict[tuple[int, int], int],
                   right: dict[tuple[int, int], int],
                   factor: int = 1) -> dict[tuple[int, int], int]:
    """Add exact polynomial coefficient maps."""
    result = dict(left)
    for power, coefficient in right.items():
        value = result.get(power, 0) + factor * coefficient
        if value:
            result[power] = value
        else:
            result.pop(power, None)
    return result


def polynomial_multiply(left: dict[tuple[int, int], int],
                        right: dict[tuple[int, int], int]
                        ) -> dict[tuple[int, int], int]:
    """Multiply exact polynomial coefficient maps."""
    result: dict[tuple[int, int], int] = {}
    for (lx, ly), lc in left.items():
        for (rx, ry), rc in right.items():
            power = (lx + rx, ly + ry)
            result[power] = result.get(power, 0) + lc * rc
    return {power: coefficient for power, coefficient in result.items()
            if coefficient}


def validate_graph_polynomial(graph: Any
                              ) -> tuple[dict[tuple[int, int], int], int, int]:
    """Validate and expand an arithmetic graph without production helpers."""
    if not isinstance(graph, dict) or set(graph) != {
            "input_types", "nodes", "output"}:
        raise ValueError("malformed graph fields")
    if graph["input_types"] != {"x": "Int", "y": "Int"}:
        raise ValueError("malformed graph inputs")
    nodes = graph["nodes"]
    if not isinstance(nodes, list):
        raise ValueError("graph node table is not a list")
    polynomials: list[dict[tuple[int, int], int]] = []
    tree_sizes: list[int] = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError("malformed graph node")
        if (set(node) != {"op", "type", "args"}
                or node.get("op") not in OP_ORDER
                or node.get("type") != "Int"):
            raise ValueError("graph is outside arithmetic DSL")
        args = node.get("args")
        if not isinstance(args, list) or len(args) != 2:
            raise ValueError("malformed arithmetic arguments")
        operands = []
        operand_sizes = []
        for ref in args:
            if isinstance(ref, str):
                if ref not in {"x", "y"}:
                    raise ValueError("unknown graph input")
                operands.append({(1, 0): 1} if ref == "x"
                                else {(0, 1): 1})
                operand_sizes.append(0)
            elif type(ref) is int and 0 <= ref < index:
                operands.append(polynomials[ref])
                operand_sizes.append(tree_sizes[ref])
            else:
                raise ValueError("forward, cyclic, or malformed graph ref")
        if node["op"] == "add":
            poly = polynomial_add(operands[0], operands[1])
        elif node["op"] == "sub":
            poly = polynomial_add(operands[0], operands[1], -1)
        else:
            poly = polynomial_multiply(operands[0], operands[1])
        polynomials.append(poly)
        tree_sizes.append(1 + sum(operand_sizes))
    output = graph["output"]
    if isinstance(output, str):
        if output not in {"x", "y"}:
            raise ValueError("unknown graph output")
        poly = {(1, 0): 1} if output == "x" else {(0, 1): 1}
        expanded_ops = 0
    elif type(output) is int and 0 <= output < len(polynomials):
        poly = polynomials[output]
        expanded_ops = tree_sizes[output]
    else:
        raise ValueError("malformed graph output")
    return poly, len(nodes), expanded_ops


def execute_graph(graph: dict[str, Any], x_value: int,
                  y_value: int) -> int:
    """Execute a graph already checked by validate_graph_polynomial."""
    values: list[int] = []
    for node in graph["nodes"]:
        operands = [({"x": x_value, "y": y_value}[ref]
                     if isinstance(ref, str) else values[ref])
                    for ref in node["args"]]
        if node["op"] == "add":
            values.append(operands[0] + operands[1])
        elif node["op"] == "sub":
            values.append(operands[0] - operands[1])
        else:
            values.append(operands[0] * operands[1])
    output = graph["output"]
    if isinstance(output, str):
        return {"x": x_value, "y": y_value}[output]
    return values[output]


def arithmetic_vector(op: str, left: tuple[int, ...],
                     right: tuple[int, ...]) -> tuple[int, ...]:
    """Apply one benchmark arithmetic operation to row vectors."""
    if op == "add":
        return tuple(a + b for a, b in zip(left, right))
    if op == "sub":
        return tuple(a - b for a, b in zip(left, right))
    return tuple(a * b for a, b in zip(left, right))


def term_size(term: Any) -> int:
    """Count operators in one independently constructed expression tree."""
    if term[0] == "input":
        return 0
    return 1 + term_size(term[1]) + term_size(term[2])


def feature_vector(op: str, left: tuple[int, ...],
                   right: tuple[int, ...], target: tuple[int, ...],
                   left_size: int, right_size: int, root_size: int,
                   bound: int, ordinal: int, width: int) -> list[float]:
    """Rebuild the twelve layer-ordinal features from observed rows."""
    one_hot = [float(op == name) for name in OP_ORDER]
    norm = float(bound) or 1.0
    features = one_hot + [left_size / norm, right_size / norm,
                          root_size / norm]
    for vector in (left, right):
        hit = sum(a == b for a, b in zip(vector, target))
        low, high = min(target), max(target)
        inside = sum(low <= value <= high for value in vector)
        features.extend([hit / len(target), inside / len(target)])
    fraction = ordinal / (width or 1)
    features.extend([fraction, 1.0 - fraction])
    return features


def independently_label_training_task(task: dict[str, Any],
                                       sampling_seed: int
                                       ) -> dict[str, Any]:
    """Rebuild the first solving layer and its capped original ordinals."""
    rows = tuple((row[0], row[1]) for row in task["train"])
    target = tuple(row[2] for row in task["train"])
    bound = int(task["max_tree_ops"])
    levels: dict[int, list[Any]] = {
        0: [("input", "x"), ("input", "y")],
    }
    vectors: dict[Any, tuple[int, ...]] = {
        ("input", "x"): tuple(row[0] for row in rows),
        ("input", "y"): tuple(row[1] for row in rows),
    }
    generated = 0
    signature_evaluations = 0
    operand_lookups = 0
    candidate_row_operations = 2 * len(rows)
    solved_layer = 0
    layer: list[tuple[str, Any, Any]] = []
    labels: list[int] = []
    for size in range(1, bound + 1):
        triples = []
        for left_size in range(size):
            right_size = size - left_size - 1
            for left in levels[left_size]:
                for right in levels[right_size]:
                    for op in OP_ORDER:
                        triples.append((op, left, right))
        seen: set[tuple[str, tuple[int, ...]]] = set()
        current = []
        layer_labels = []
        for op, left, right in triples:
            operand_lookups += 2
            output = arithmetic_vector(op, vectors[left], vectors[right])
            signature_evaluations += 1
            candidate_row_operations += len(rows)
            layer_labels.append(int(output == target))
            key = (op, output)
            if key not in seen:
                seen.add(key)
                term = (op, left, right)
                current.append(term)
                vectors[term] = output
        generated += len(triples)
        levels[size] = current
        if any(layer_labels):
            solved_layer = size
            layer = triples
            labels = layer_labels
            break
    if not solved_layer:
        raise ValueError("training function was not solved within bound")
    positives = [index for index, value in enumerate(labels) if value == 1]
    negatives = [index for index, value in enumerate(labels) if value == 0]
    task_id = str(task["id"])

    def sample(indices: list[int], label: int, cap: int) -> list[int]:
        def sample_key(index: int) -> bytes:
            payload = ("sha256-v1:seed:task-id:class:original-ordinal:"
                       f"{sampling_seed}:{task_id}:{label}:{index}")
            return hashlib.sha256(payload.encode("utf-8")).digest()
        return sorted(sorted(indices, key=lambda index:
                             (sample_key(index), index))[:cap])

    selected_pos = sample(positives, 1, 16)
    selected_neg = sample(negatives, 0, 256)
    selected = sorted(selected_pos + selected_neg)
    width = len(layer)
    records = []
    for index in selected:
        op, left, right = layer[index]
        left_vector = vectors[left]
        right_vector = vectors[right]
        output = arithmetic_vector(op, left_vector, right_vector)
        features = feature_vector(
            op, left_vector, right_vector, target, term_size(left),
            term_size(right), solved_layer, bound, index + 1, width,
        )
        records.append({
            "ordinal": index, "label": labels[index], "operator": op,
            "left_size": term_size(left),
            "right_size": term_size(right),
            "left_vector": list(left_vector),
            "right_vector": list(right_vector),
            "output_vector": list(output), "target_vector": list(target),
            "feature_vector": features, "root_cost": solved_layer,
        })
    selected_row_operations = len(rows) * len(selected)
    candidate_row_operations += selected_row_operations
    return {
        "solving_layer": solved_layer, "full_layer_width": width,
        "generated_descriptors_through_solving_layer": generated,
        "signature_evaluations": signature_evaluations,
        "candidate_row_operations": candidate_row_operations,
        "selected_vector_row_operations": selected_row_operations,
        "operand_vector_lookups": operand_lookups + 2 * len(selected),
        "feature_builds": len(selected),
        "feature_row_statistic_observations": (
            4 * len(rows) * len(selected)),
        "target_range_row_reads": 4 * len(rows) * len(selected),
        "label_counts": {"positive": len(positives),
                         "negative": len(negatives)},
        "selected_counts": {"positive": len(selected_pos),
                            "negative": len(selected_neg)},
        "selected_indices": selected,
        "selected_indices_sha256": hashlib.sha256(
            ",".join(str(index) for index in selected).encode("ascii")
        ).hexdigest(),
        "selected_indices_by_class": {
            "positive": selected_pos, "negative": selected_neg,
        },
        "selected_indices_sha256_by_class": {
            "positive": hashlib.sha256(
                ",".join(str(index) for index in selected_pos).encode("ascii")
            ).hexdigest(),
            "negative": hashlib.sha256(
                ",".join(str(index) for index in selected_neg).encode("ascii")
            ).hexdigest(),
        },
        "selected_candidates": records,
    }


def audit_training(training: dict[str, Any],
                   corpus: dict[str, Any],
                   protocol: dict[str, Any]) -> list[str]:
    """Verify capped groups, full-layer ordinals, features and model states."""
    errors: list[str] = []
    setup = training.get("training_groups", {})
    group_reports = setup.get("groups", [])
    train_tasks = corpus["splits"]["train"]
    if len(group_reports) != len(train_tasks):
        errors.append("training group report count differs from train tasks")
    sampling_seed = protocol["training"]["sampling_seed"]
    totals = defaultdict(int)
    for index, task in enumerate(train_tasks):
        if index >= len(group_reports):
            break
        report = group_reports[index]
        if report.get("task_id") != task["id"]:
            errors.append(f"training group order mismatch at {task['id']}")
            continue
        if report.get("status") != "ok" or report.get("error"):
            errors.append(f"training group failed for {task['id']}")
            continue
        expected = independently_label_training_task(task, sampling_seed)
        for key, value in expected.items():
            actual = report.get(key)
            if not same(actual, value):
                errors.append(f"{task['id']}: training field {key} differs")
        expected_pos_cap = protocol["training"]["positive_cap"]
        expected_neg_cap = protocol["training"]["negative_cap"]
        selected_counts = report.get("selected_counts", {})
        if selected_counts.get("positive", 0) > expected_pos_cap:
            errors.append(f"{task['id']}: positive cap exceeded")
        if selected_counts.get("negative", 0) > expected_neg_cap:
            errors.append(f"{task['id']}: negative cap exceeded")
        selected = report.get("selected_indices", [])
        width = report.get("full_layer_width", 0)
        if any(type(value) is not int or value < 0 or value >= width
               for value in selected):
            errors.append(f"{task['id']}: invalid full-layer ordinal")
        if report.get("sampling_seed") != sampling_seed:
            errors.append(f"{task['id']}: wrong group sampling seed")
        if report.get("positive_cap") != expected_pos_cap:
            errors.append(f"{task['id']}: wrong positive cap metadata")
        if report.get("negative_cap") != expected_neg_cap:
            errors.append(f"{task['id']}: wrong negative cap metadata")
        if report.get("selected_counts") != expected["selected_counts"]:
            errors.append(f"{task['id']}: selected class counts differ")
        recipe = "sha256-v1:seed:task-id:class:original-ordinal"
        if report.get("sampling_recipe") != recipe:
            errors.append(f"{task['id']}: wrong sampling recipe")
        for name, value in {
                "groups": 1, "successful_groups": 1, "failed_groups": 0,
                "full_layer_candidates": int(
                    report.get("full_layer_width", 0)),
                "selected_candidates": len(selected),
                "positive_labels": int(
                    report.get("label_counts", {}).get("positive", 0)),
                "negative_labels": int(
                    report.get("label_counts", {}).get("negative", 0)),
        }.items():
            totals[name] += value
    if totals != setup.get("totals", {}):
        errors.append("training group totals do not match per-task rows")
    if setup.get("label_semantics") != (
            "first-solving-layer-on-training-observed-rows"):
        errors.append("training label semantics differ")
    if training.get("status") != "complete":
        errors.append("training report is incomplete")
    seeds = training.get("seeds", [])
    if [record.get("seed") for record in seeds] != list(SEEDS):
        errors.append("training seed records differ")
    wanted_parameters = {
        "trained_snn": 225, "untrained_snn": 225,
        "mlp": 225, "recurrent": 497,
    }
    wanted_steps = {"trained_snn": 6, "untrained_snn": 6,
                    "mlp": 1, "recurrent": 6, "seeded_random": 1,
                    "heuristic": 1, "streaming": 0}
    for seed_record in seeds:
        controls = seed_record.get("controls", {})
        if set(controls) != set(CONTROL_NAMES):
            seed = seed_record.get("seed")
            errors.append(f"seed {seed}: control set differs")
        fit_work_reference = None
        selected_groups = [report.get("selected_candidates", [])
                           for report in group_reports]
        eligible_groups = [group for group in selected_groups
                           if any(item.get("label") == 1 for item in group)
                           and any(item.get("label") == 0 for item in group)]
        expected_positive = sum(
            sum(item.get("label") == 1 for item in group)
            for group in eligible_groups)
        expected_negative = sum(
            sum(item.get("label") == 0 for item in group)
            for group in eligible_groups)
        expected_pairs = sum(
            sum(item.get("label") == 1 for item in group)
            * sum(item.get("label") == 0 for item in group)
            for group in eligible_groups)
        expected_updates = (protocol["training"]["epochs"]
                            if eligible_groups else 0)
        expected_batches = (2 * protocol["training"]["epochs"]
                            * len(eligible_groups))
        for control in CONTROL_NAMES:
            summary = controls.get(control, {})
            metadata = summary.get("metadata", {})
            parameters = wanted_parameters.get(control, 0)
            if metadata.get("trainable_parameters") != parameters:
                errors.append(f"{control}: parameter count differs")
            if metadata.get("simulation_steps") != wanted_steps[control]:
                errors.append(f"{control}: simulation steps differ")
            if summary.get("requested_epochs") != protocol[
                    "training"]["epochs"]:
                errors.append(f"{control}: requested epoch count differs")
            trainable = control in {"trained_snn", "mlp", "recurrent"}
            if summary.get("fit_performed") is not trainable:
                errors.append(f"{control}: fit-performed flag differs")
            if (summary.get("weights_before_fit_sha256") !=
                    summary.get("weights_after_fit_sha256")):
                if control not in {"trained_snn", "mlp", "recurrent"}:
                    errors.append(f"{control}: non-trainable weights changed")
            if (summary.get("state_fingerprint_before_fit") !=
                    summary.get("state_fingerprint_after_fit")):
                if control not in {"trained_snn", "mlp", "recurrent"}:
                    errors.append(f"{control}: non-trainable state changed")
            if control not in {"seeded_random", "streaming"}:
                if (summary.get("state_fingerprint_after_eval") !=
                        summary.get("state_fingerprint_after_fit")):
                    errors.append(
                        f"{control}: state changed across evaluation")
                if (summary.get("weights_after_eval_sha256") !=
                        summary.get("weights_after_fit_sha256")):
                    errors.append(
                        f"{control}: weights changed across evaluation")
                if summary.get("weights_frozen_during_eval") is not True:
                    errors.append(
                        f"{control}: evaluation freeze flag is false")
            work = summary.get("fit_work", {})
            if trainable:
                expected_steps = wanted_steps[control]
                expected_work = {
                    "requested_epochs": protocol["training"]["epochs"],
                    "eligible_groups": len(eligible_groups),
                    "positive_features": expected_positive,
                    "negative_features": expected_negative,
                    "margin_pairs_per_epoch": expected_pairs,
                    "forward_batches": expected_batches,
                    "scored_feature_rows": protocol["training"]["epochs"] and
                        sum(len(group) for group in eligible_groups)
                        * protocol["training"]["epochs"],
                    "simulation_row_steps": protocol["training"]["epochs"] and
                        sum(len(group) for group in eligible_groups)
                        * protocol["training"]["epochs"] * expected_steps,
                    "optimizer_updates": expected_updates,
                }
                if not same(work, expected_work):
                    errors.append(f"{control}: fit work differs from groups")
                if fit_work_reference is None:
                    fit_work_reference = work
                else:
                    for field in ("eligible_groups", "positive_features",
                                  "negative_features",
                                  "margin_pairs_per_epoch",
                                  "forward_batches", "scored_feature_rows",
                                  "optimizer_updates"):
                        if work.get(field) != fit_work_reference.get(field):
                            errors.append(f"{control}: learned controls used "
                                          "different groups")
            else:
                if control in {"streaming", "seeded_random"}:
                    expected_work = {
                        "candidate_score_evaluations": 0,
                        "eligible_groups": 0,
                        "negative_examples": 0,
                        "optimizer_updates": 0,
                        "pairwise_margin_comparisons_per_epoch": 0,
                        "positive_examples": 0, "snn_steps": 0,
                    }
                else:
                    expected_work = {
                        "eligible_groups": len(eligible_groups),
                        "positive_features": expected_positive,
                        "negative_features": expected_negative,
                        "margin_pairs_per_epoch": expected_pairs,
                        "optimizer_updates": 0, "forward_batches": 0,
                        "scored_feature_rows": 0,
                        "simulation_row_steps": 0,
                    }
                if not same(work, expected_work):
                    errors.append(
                        f"{control}: non-trainable fit work differs")
            if "fit_changed_weights" in summary:
                if summary["fit_changed_weights"] is not (
                        summary.get("weights_before_fit_sha256") !=
                        summary.get("weights_after_fit_sha256")):
                    errors.append(
                        f"{control}: fit weight-change flag differs")
            if "fit_changed_state" in summary:
                if summary["fit_changed_state"] is not (
                        summary.get("state_fingerprint_before_fit") !=
                        summary.get("state_fingerprint_after_fit")):
                    errors.append(f"{control}: fit state-change flag differs")
    return errors


def sha256_file(path: Path) -> str:
    """Hash one file without invoking project helpers."""
    return sha256(path.read_bytes())


def current_source_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    """Rebuild the runner's declared source set from its environment record."""
    paths = set((ROOT / "rsi_programmer").rglob("*.py"))
    paths.update((ROOT / "scripts").glob("benchmark_pre_eval*.py"))
    paths.update({
        ROOT / "scripts/generate_benchmark_pre_eval_v2.py",
        ROOT / "rsi_programmer/benchmark_v2_corpus.py",
        ROOT / "fixtures/benchmark_pre_eval_v2.json",
        ROOT / "fixtures/benchmark_pre_eval_corpus_v2.json",
        ROOT / "requirements.txt",
    })
    lock = ROOT / "uv.lock"
    if lock.exists():
        paths.add(lock)
    environment = manifest.get("before", {}).get("environment", {})
    spikeforge_path = environment.get("spikeforge_path")
    if not isinstance(spikeforge_path, str):
        raise ValueError("manifest lacks SpikeForge module path")
    package_root = Path(spikeforge_path).resolve().parent
    paths.update(package_root.rglob("*.py"))
    result = {}
    for path in sorted(paths):
        label = (str(path.relative_to(ROOT)) if ROOT in path.parents
                 else str(path))
        result[label] = sha256_file(path)
    return result


def aggregate_independently(rows: list[dict[str, Any]],
                            protocol: dict[str, Any]
                            ) -> dict[str, Any]:
    """Recompute seed/family summaries, decision checks and bootstrap."""
    family_seed: dict[tuple[str, int, str, int], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    for row in rows:
        cell = family_seed[(row["control"], row["budget"], row["family"],
                            row["seed"])]
        cell[0] += int(row["accepted"])
        cell[1] += 1
        search = row.get("search", {})
        cell[2] += float(search.get("signature_evaluations", 0))
        cell[3] += float(row.get("wall_seconds_including_wrapper", 0))
        cell[4] += float(search.get("snn_steps", 0))
    family_rows = []
    family_fractions: dict[tuple[str, int, str], list[float]] = defaultdict(
        list)
    overall: dict[tuple[str, int, int], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    for (control, budget, family, seed), values in sorted(family_seed.items()):
        successes, count, scans, wall, steps = values
        fraction = successes / count if count else 0.0
        family_fractions[(control, budget, family)].append(fraction)
        family_rows.append({
            "control": control, "budget": budget, "family": family,
            "seed": seed, "successes": int(successes),
            "tasks": int(count), "success_fraction": fraction,
            "candidate_signatures": int(scans), "wall_seconds": wall,
            "snn_steps": int(steps),
        })
        total = overall[(control, budget, seed)]
        for index, value in enumerate(values):
            total[index] += value
    family_means = [{
        "control": control, "budget": budget, "family": family,
        "seed_fractions": fractions,
        "mean_seed_fraction": sum(fractions) / len(fractions),
    } for (control, budget, family), fractions in
        sorted(family_fractions.items())]
    overall_seed_rows = []
    overall_fractions: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (control, budget, seed), values in sorted(overall.items()):
        successes, count, scans, wall, steps = values
        fraction = successes / count if count else 0.0
        overall_fractions[(control, budget)].append(fraction)
        overall_seed_rows.append({
            "control": control, "budget": budget, "seed": seed,
            "successes": int(successes), "tasks": int(count),
            "success_fraction": fraction,
            "candidate_signatures": int(scans), "wall_seconds": wall,
            "snn_steps": int(steps),
        })
    overall_means = [{
        "control": control, "budget": budget,
        "seed_fractions": fractions,
        "mean_seed_fraction": sum(fractions) / len(fractions),
    } for (control, budget), fractions in sorted(overall_fractions.items())]
    controls = protocol["controls"]
    budgets = protocol["evaluation"]["candidate_scan_budgets"]
    families = protocol["corpus"]["families"]
    scores = {(item["control"], item["budget"]):
              item["mean_seed_fraction"] for item in overall_means}
    matched = []
    for control in controls:
        for target in protocol["decision_rule"]["matched_success_targets"]:
            reached = [budget for budget in budgets
                       if scores.get((control, budget), 0.0) >= target]
            matched.append({
                "control": control, "target": target,
                "lowest_budget": min(reached) if reached else None,
                "unreached_censored_at": None if reached else max(budgets),
            })
    checks = []
    for budget in protocol["decision_rule"]["primary_budgets"]:
        trained = scores.get(("trained_snn", budget), 0.0)
        best_other = max((scores.get((control, budget), 0.0)
                          for control in controls
                          if control != "trained_snn"), default=0.0)
        higher = 0
        for family in families:
            trained_family = next((item["mean_seed_fraction"]
                                   for item in family_means
                                   if item["control"] == "trained_snn"
                                   and item["budget"] == budget
                                   and item["family"] == family), 0.0)
            other_family = max((item["mean_seed_fraction"]
                                for item in family_means
                                if item["budget"] == budget
                                and item["family"] == family
                                and item["control"] != "trained_snn"),
                               default=0.0)
            higher += int(trained_family > other_family)
        checks.append({
            "budget": budget,
            "trained_snn_mean_fraction": trained,
            "best_other_mean_fraction": best_other,
            "overall_margin": trained - best_other,
            "families_where_trained_snn_is_higher": higher,
        })
    decision = protocol["decision_rule"]
    rule_pass = all(
        item["overall_margin"] >= decision["minimum_margin"]
        and item["families_where_trained_snn_is_higher"]
        >= decision["minimum_families_higher"] for item in checks)
    uncertainty = protocol["uncertainty"]
    rng = random.Random(uncertainty["seed"])
    tasks_by_family = {
        family: sorted({row["task_id"] for row in rows
                        if row["family"] == family})
        for family in families
    }
    task_scores: dict[tuple[int, str, str], float] = {}
    for budget in budgets:
        for family in families:
            for task_id in tasks_by_family[family]:
                for control in controls:
                    outcomes = [int(row["accepted"]) for row in rows
                                if row["budget"] == budget
                                and row["family"] == family
                                and row["task_id"] == task_id
                                and row["control"] == control]
                    task_scores[(budget, task_id, control)] = (
                        sum(outcomes) / len(outcomes) if outcomes else 0.0)
    comparisons = []
    for budget in decision["primary_budgets"]:
        for control in controls:
            if control == "trained_snn":
                continue
            diffs = []
            for family in families:
                diffs.extend(
                    task_scores[(budget, task_id, "trained_snn")]
                    - task_scores[(budget, task_id, control)]
                    for task_id in tasks_by_family[family])
            draws = []
            for _ in range(uncertainty["draws"]):
                sample = []
                for family in families:
                    members = tasks_by_family[family]
                    sampled = [rng.choice(members) for _ in members]
                    sample.extend(
                        task_scores[(budget, task_id, "trained_snn")]
                        - task_scores[(budget, task_id, control)]
                        for task_id in sampled)
                draws.append(sum(sample) / len(sample) if sample else 0.0)
            draws.sort()
            draws_count = uncertainty["draws"]
            comparisons.append({
                "budget": budget, "control": control,
                "mean_paired_difference": sum(diffs) / len(diffs)
                if diffs else 0.0,
                "confidence_level": uncertainty["confidence_level"],
                "confidence_interval": [
                    draws[int(0.025 * (draws_count - 1))],
                    draws[int(0.975 * (draws_count - 1))],
                ],
            })
    bootstrap = {
        "method": "family-stratified paired task bootstrap",
        "task_unit": "evaluation task; average across training seeds first",
        "draws": uncertainty["draws"], "seed": uncertainty["seed"],
        "confidence_level": uncertainty["confidence_level"],
        "descriptive_only": True, "comparisons": comparisons,
    }
    return {
        "family_seed_rows": family_rows,
        "family_means_across_seeds": family_means,
        "overall_seed_rows": overall_seed_rows,
        "overall_means_across_seeds": overall_means,
        "matched_success_budgets": matched,
        "paired_task_bootstrap": bootstrap,
        "trained_snn_advantage": {
            "rule_would_pass": rule_pass, "checks": checks,
            "scope": decision["claim_scope"],
        },
    }


def verify_raw(bundle: Path, protocol: dict[str, Any],
               corpus: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Check matrix completeness, every returned graph, and frozen states."""
    errors: list[str] = []
    manifest_path = bundle / "manifest.json"
    training_path = bundle / "training.json"
    aggregate_path = bundle / "aggregate.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    raw_path = bundle / "raw.jsonl"
    rows = [json.loads(line) for line in
            raw_path.read_text(encoding="utf-8").splitlines() if line]
    expected_protocol_hash = input_audit.PINNED_PROTOCOL_SHA256
    expected_corpus_hash = input_audit.PINNED_CORPUS_SHA256
    for label, path, expected_hash in (
            ("protocol", input_audit.PROTOCOL_PATH, expected_protocol_hash),
            ("corpus", input_audit.CORPUS_PATH, expected_corpus_hash)):
        try:
            if sha256_file(path) != expected_hash:
                errors.append(
                    f"frozen {label} bytes differ from input audit pin")
        except OSError as exc:
            errors.append(f"cannot hash frozen {label}: {exc}")
    if protocol != input_audit.load_json(input_audit.PROTOCOL_PATH):
        errors.append("parsed protocol differs from frozen protocol fixture")
    if corpus != input_audit.load_json(input_audit.CORPUS_PATH):
        errors.append("parsed corpus differs from frozen corpus fixture")
    if manifest.get("protocol_sha256") != expected_protocol_hash:
        errors.append("manifest protocol hash differs from frozen input")
    if manifest.get("corpus_sha256") != expected_corpus_hash:
        errors.append("manifest corpus hash differs from frozen input")
    output_hash_checks = {
        "raw_jsonl_sha256": sha256_file(raw_path),
        "training_json_sha256": sha256_file(training_path),
        "aggregate_json_sha256": sha256_file(aggregate_path),
    }
    for field, actual in output_hash_checks.items():
        if manifest.get(field) != actual:
            errors.append(f"manifest {field} does not match output bytes")
    expected_count = (len(protocol["training"]["seeds"])
                      * len(protocol["controls"])
                      * len(corpus["splits"]["eval"])
                      * len(protocol["evaluation"][
                          "candidate_scan_budgets"]))
    if len(rows) != expected_count or expected_count != 4032:
        errors.append(
            f"raw row count {len(rows)} != expected {expected_count}")
    if manifest.get("row_count") != len(rows):
        errors.append("manifest row count differs from JSONL")
    if manifest.get("expected_row_count") != expected_count:
        errors.append("manifest expected row count differs")
    if manifest.get("grid_valid") is not True:
        errors.append("runner manifest reports an invalid matrix")
    if manifest.get("infrastructure_valid") is not True:
        errors.append("runner manifest reports infrastructure failure")
    if manifest.get("infrastructure_errors"):
        errors.append("runner recorded infrastructure errors")
    if manifest.get("source_frozen") is not True:
        errors.append("runner source manifest changed during measurement")
    before = manifest.get("before", {})
    after = manifest.get("after", {})
    source_before = before.get("source_sha256", {})
    source_after = after.get("source_sha256", {})
    if source_before != source_after:
        errors.append("before/after source manifests differ")
    try:
        source_current = current_source_hashes(manifest)
        if source_before != source_current:
            errors.append("current source bytes differ from run manifest")
    except (OSError, ValueError) as exc:
        source_current = {}
        errors.append("cannot rebuild current source manifest: " + str(exc))
    env_before = dict(before.get("environment", {}))
    env_after = dict(after.get("environment", {}))
    env_before.pop("git", None)
    env_after.pop("git", None)
    if env_before != env_after:
        errors.append("runtime environment changed during measurement")
    if training.get("protocol_sha256") != expected_protocol_hash:
        errors.append("training report protocol hash differs")
    if training.get("corpus_sha256") != expected_corpus_hash:
        errors.append("training report corpus hash differs")
    errors.extend(audit_training(training, corpus, protocol))

    task_by_id = {task["id"]: task for task in corpus["splits"]["eval"]}
    expected_keys = {
        (seed, control, task["id"], budget)
        for seed in protocol["training"]["seeds"]
        for control in protocol["controls"]
        for task in corpus["splits"]["eval"]
        for budget in protocol["evaluation"]["candidate_scan_budgets"]
    }
    seen_keys = set()
    controls_by_seed = {
        record["seed"]: record.get("controls", {})
        for record in training.get("seeds", [])
    }
    model_hashes: dict[tuple[int, str], tuple[Any, Any]] = {}
    random_hashes: dict[tuple[int, str], tuple[Any, Any]] = {}
    graph_count = 0
    graph_errors = 0
    accepted_count = 0
    work_checks = 0
    steps_by_control = {
        "streaming": 0, "heuristic": 1, "seeded_random": 1,
        "untrained_snn": 6, "trained_snn": 6, "mlp": 1,
        "recurrent": 6,
    }
    snn_controls = {"untrained_snn", "trained_snn"}
    for row_index, row in enumerate(rows):
        try:
            key = (row["seed"], row["control"], row["task_id"],
                   row["budget"])
        except (KeyError, TypeError):
            errors.append(f"raw row {row_index}: malformed matrix key")
            continue
        if key not in expected_keys:
            errors.append(f"raw row {row_index}: unexpected matrix key {key}")
        if key in seen_keys:
            errors.append(f"raw row {row_index}: duplicated matrix key {key}")
        seen_keys.add(key)
        task = task_by_id.get(row.get("task_id"))
        if task is None:
            errors.append(f"raw row {row_index}: unknown task id")
            continue
        if (row.get("protocol_id") != protocol["protocol_id"]
                or row.get("corpus_sha256") != expected_corpus_hash):
            errors.append(f"{key}: row protocol or corpus pin differs")
        if row.get("family") != task["family"]:
            errors.append(f"{key}: row family differs from corpus")
        if row.get("bound") != protocol["evaluation"]["tree_bound"]:
            errors.append(f"{key}: row tree bound differs")
        if row.get("budget") not in protocol["evaluation"][
                "candidate_scan_budgets"]:
            errors.append(f"{key}: row budget is outside protocol")
        if row.get("error"):
            errors.append(f"{key}: attempt contains search error")
        search = row.get("search")
        if not isinstance(search, dict):
            errors.append(f"{key}: search metrics are malformed")
            search = {}
        signatures = search.get("signature_evaluations")
        if (type(signatures) is not int or signatures < 0
                or signatures > row.get("budget", -1)):
            errors.append(f"{key}: signature budget invariant failed")
        if search.get("tried") != signatures:
            errors.append(f"{key}: tried/signature count differs")
        if search.get("oracle_calls") != 0:
            errors.append(f"{key}: search called an oracle")
        if search.get("signature_budget") is not None:
            errors.append(f"{key}: undeclared signature budget was applied")
        if search.get("solved") is not row.get("solved"):
            errors.append(f"{key}: solved flag differs from search result")
        control = row.get("control")
        if control not in CONTROL_NAMES:
            errors.append(f"{key}: unknown control")
        else:
            metrics = {
                name: search.get(name, 0) for name in (
                    "policy_batches", "scored_descriptors", "feature_builds",
                    "policy_batch_ticks", "policy_row_steps", "snn_steps",
                )
            }
            try:
                if any(type(value) is not int or value < 0
                       for value in metrics.values()):
                    raise ValueError("negative or non-integer policy metric")
                steps = steps_by_control[control]
                if metrics["policy_batch_ticks"] != (
                        steps * metrics["policy_batches"]):
                    raise ValueError("batch tick count")
                if metrics["policy_row_steps"] != (
                        steps * metrics["scored_descriptors"]):
                    raise ValueError("policy row-step count")
                if control in snn_controls:
                    if metrics["snn_steps"] != 6 * metrics[
                            "policy_batches"]:
                        raise ValueError("SNN step count")
                elif metrics["snn_steps"] != 0:
                    raise ValueError("non-SNN reported SNN steps")
                if control == "streaming":
                    if any(metrics[name] for name in (
                            "policy_batches", "scored_descriptors",
                            "feature_builds", "policy_batch_ticks",
                            "policy_row_steps", "snn_steps")):
                        raise ValueError("streaming reported ranking work")
                elif metrics["scored_descriptors"] != metrics[
                        "feature_builds"]:
                    raise ValueError("feature/scored descriptor counts differ")
                work_checks += 1
            except ValueError as exc:
                errors.append(f"{key}: policy work invariant failed ({exc})")
        graph = row.get("returned_graph")
        validation = row.get("validation", {})
        train_ok = False
        held_ok = False
        exact_ok = False
        graph_valid = False
        node_count = 0
        if graph is not None:
            graph_count += 1
            try:
                actual_poly, node_count, expanded_ops = (
                    validate_graph_polynomial(graph))
                if expanded_ops > protocol["evaluation"]["tree_bound"]:
                    raise ValueError("expanded returned tree exceeds bound")
                graph_valid = True
                witness_poly = input_audit.expression_polynomial(
                    task["generator"]["program"])
                exact_ok = actual_poly == witness_poly
                train_ok = all(execute_graph(graph, x, y) == expected
                               for x, y, expected in task["train"])
                held_ok = all(execute_graph(graph, x, y) == expected
                              for x, y, expected in task["held_out"])
            except (TypeError, ValueError, KeyError, IndexError) as exc:
                graph_errors += 1
                # Unsupported or malformed DAGs cannot pass any audit gate.
                train_ok = held_ok = exact_ok = False
                if validation.get("error") in ("", "no-graph"):
                    errors.append(f"{key}: returned graph invalid: {exc}")
        expected_accept = bool(row.get("solved") and train_ok and held_ok
                               and exact_ok)
        if row.get("accepted") is not expected_accept:
            errors.append(f"{key}: accepted flag bypasses independent gates")
        accepted_count += int(expected_accept)
        for field, value in (("train_valid", train_ok),
                             ("held_out_valid", held_ok),
                             ("exact_function_match", exact_ok)):
            if validation.get(field) is not value:
                errors.append(f"{key}: validation {field} differs")
        expected_calls = 0 if graph is None else 1
        if graph_valid:
            # The runner counts one graph-level check and one call per row.
            expected_calls = 1 + len(task["train"]) + len(task["held_out"])
        if validation.get("validation_calls") != expected_calls:
            errors.append(f"{key}: validation call count differs")
        expected_node_ops = node_count * 24 if graph_valid else 0
        if validation.get("node_row_operations") != expected_node_ops:
            errors.append(f"{key}: validation node-row count differs")
        if graph is None and validation.get("error") != "no-graph":
            errors.append(f"{key}: missing graph lacks explicit failure")
        if validation.get("elapsed_seconds", -1) < 0:
            errors.append(f"{key}: invalid validation duration")
        state_before = row.get("state_fingerprint_before_eval")
        state_after = row.get("state_fingerprint_after_eval")
        weight_before = row.get("weights_before_eval_sha256")
        weight_after = row.get("weights_after_eval_sha256")
        if row.get("state_frozen_during_eval") is not True:
            errors.append(f"{key}: policy state changed during evaluation")
        if control == "streaming":
            if (state_before != "streaming-order-none"
                    or state_after is not None or weight_before != "none"
                    or weight_after is not None):
                errors.append(f"{key}: streaming state sentinels differ")
        elif state_before != state_after or weight_before != weight_after:
            errors.append(f"{key}: state or pure weights changed in attempt")
        if control not in {"streaming", "seeded_random"}:
            fitted = controls_by_seed.get(row.get("seed"), {}).get(
                control, {})
            if state_before != fitted.get("state_fingerprint_after_fit"):
                errors.append(f"{key}: state differs from fitted model")
            if weight_before != fitted.get("weights_after_fit_sha256"):
                errors.append(f"{key}: weights differ from fitted model")
            pair = (row.get("seed"), control)
            values = (state_before, weight_before)
            if pair in model_hashes and model_hashes[pair] != values:
                errors.append(f"{key}: model state differs across cells")
            model_hashes[pair] = values
        elif control == "seeded_random":
            pair = (row.get("seed"), row.get("task_id"))
            values = (state_before, weight_before)
            if pair in random_hashes and random_hashes[pair] != values:
                errors.append(
                    f"{key}: task-seeded random state differs by budget")
            random_hashes[pair] = values
    if seen_keys != expected_keys:
        errors.append("raw matrix has missing or unexpected cells")
    expected_aggregate = aggregate_independently(rows, protocol)
    expected_aggregate["infrastructure_valid"] = True
    expected_aggregate["trained_snn_advantage"][
        "supported_by_frozen_rule"] = (
            expected_aggregate["trained_snn_advantage"]["rule_would_pass"])
    if not same(aggregate, expected_aggregate):
        errors.append("aggregate or paired bootstrap differs from raw rows")
    details = {
        "raw_rows": len(rows), "expected_rows": expected_count,
        "unique_matrix_keys": len(seen_keys), "returned_graphs": graph_count,
        "graph_validation_errors": graph_errors,
        "independently_accepted_rows": accepted_count,
        "work_invariant_cells_checked": work_checks,
        "source_files_checked": len(source_current),
        "aggregate_matches_raw": same(aggregate, expected_aggregate),
        "raw_sha256": output_hash_checks["raw_jsonl_sha256"],
    }
    return errors, details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--protocol", type=Path,
                        default=input_audit.PROTOCOL_PATH)
    parser.add_argument("--corpus", type=Path,
                        default=input_audit.CORPUS_PATH)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    errors, details = verify_raw(args.bundle, protocol, corpus)
    if sha256_file(args.protocol) != input_audit.PINNED_PROTOCOL_SHA256:
        errors.append("--protocol bytes do not match the frozen protocol pin")
    if sha256_file(args.corpus) != input_audit.PINNED_CORPUS_SHA256:
        errors.append("--corpus bytes do not match the frozen corpus pin")
    report = {"ok": not errors, "errors": errors, **details}
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
