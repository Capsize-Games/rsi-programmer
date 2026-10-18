"""Train-only capped supervision for the frozen v2 benchmark."""
from __future__ import annotations

import hashlib
import time
from collections import Counter
from typing import Any

from scripts import benchmark_pre_eval as v1

from rsi_programmer.benchmark_policy import Group

CAP_POSITIVES = 16
CAP_NEGATIVES = 256
SAMPLE_RECIPE = "sha256-v1:seed:task-id:class:original-ordinal"


def _sample_key(seed: int, task_id: str, label: int,
                ordinal: int) -> bytes:
    payload = f"{SAMPLE_RECIPE}:{seed}:{task_id}:{label}:{ordinal}"
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _sample_indices(seed: int, task_id: str, label: int,
                    candidates: list[int], cap: int) -> list[int]:
    ranked = sorted(candidates, key=lambda index: (
        _sample_key(seed, task_id, label, index), index,
    ))
    return sorted(ranked[:cap])


def _indices_hash(indices: list[int]) -> str:
    payload = ",".join(str(index) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_training_groups(
        tasks: list[dict[str, Any]], *,
        positive_cap: int = CAP_POSITIVES,
        negative_cap: int = CAP_NEGATIVES,
        sampling_seed: int = 82078,
        ) -> tuple[list[Group], dict[str, Any]]:
    """Label full solving layers from observed rows, then cap feature rows.

    Candidate ordinals and feature denominators always refer to the complete
    generated layer. Held-out rows are never read by this function.
    """
    if positive_cap < 1 or negative_cap < 1:
        raise ValueError("class caps must be positive")
    groups: list[Group] = []
    task_reports: list[dict[str, Any]] = []
    aggregate = Counter()
    setup_started = time.perf_counter()
    for task in tasks:
        task_started = time.perf_counter()
        rows = tuple((row[0], row[1]) for row in task["train"])
        target = tuple(row[2] for row in task["train"])
        bound = int(task["max_tree_ops"])
        levels: dict[int, list[v1.Expr]] = {
            0: [v1.Expr("input", "x"), v1.Expr("input", "y")],
        }
        vectors = {id(term): v1.signature(term, rows)
                   for term in levels[0]}
        layer: list[tuple[str, v1.Expr, v1.Expr]] = []
        labels: list[int] = []
        generated = 0
        row_operations = len(vectors) * len(rows)
        signature_evaluations = 0
        operand_lookups = 0
        solved_layer = 0
        try:
            for size in range(1, bound + 1):
                triples = v1.layer_triples(levels, size)
                seen: set[tuple[str, tuple[int, ...]]] = set()
                current: list[v1.Expr] = []
                layer_labels: list[int] = []
                for op, left, right in triples:
                    operand_lookups += 2
                    left_sig = vectors[id(left)]
                    right_sig = vectors[id(right)]
                    if op == "add":
                        output = tuple(a + b for a, b in
                                       zip(left_sig, right_sig))
                    elif op == "sub":
                        output = tuple(a - b for a, b in
                                       zip(left_sig, right_sig))
                    else:
                        output = tuple(a * b for a, b in
                                       zip(left_sig, right_sig))
                    signature_evaluations += 1
                    row_operations += len(rows)
                    expr = v1.Expr(op, left, right, size)
                    label = int(output == target)
                    key = (op, output)
                    if key not in seen:
                        seen.add(key)
                        current.append(expr)
                        vectors[id(expr)] = output
                    layer_labels.append(label)
                generated += len(triples)
                levels[size] = current
                if any(layer_labels):
                    solved_layer = size
                    layer = triples
                    labels = layer_labels
                    break
            if not solved_layer:
                raise ValueError("training target is unreachable within bound")
            positives = [i for i, label in enumerate(labels) if label == 1]
            negatives = [i for i, label in enumerate(labels) if label == 0]
            selected_pos = _sample_indices(
                sampling_seed, str(task["id"]), 1, positives,
                positive_cap,
            )
            selected_neg = _sample_indices(
                sampling_seed, str(task["id"]), 0, negatives,
                negative_cap,
            )
            selected = sorted(selected_pos + selected_neg)
            width = len(layer)
            group: Group = []
            selected_records = []
            for index in selected:
                op, left, right = layer[index]
                operand_lookups += 2
                left_vector = vectors[id(left)]
                right_vector = vectors[id(right)]
                output = _apply_vector(op, left_vector, right_vector)
                label = labels[index]
                feature = v1.encode(
                    op, left_vector, right_vector, target,
                    left.size, right.size, solved_layer, bound,
                    index + 1, width,
                )
                group.append((feature, label))
                selected_records.append({
                    "ordinal": index,
                    "label": label,
                    "operator": op,
                    "left_size": left.size,
                    "right_size": right.size,
                    "left_vector": list(left_vector),
                    "right_vector": list(right_vector),
                    "output_vector": list(output),
                    "target_vector": list(target),
                    "feature_vector": feature,
                    "root_cost": solved_layer,
                })
            selected_vector_row_operations = len(rows) * len(selected)
            row_operations += selected_vector_row_operations
            groups.append(group)
            report = {
                "task_id": task["id"],
                "status": "ok",
                "solving_layer": solved_layer,
                "full_layer_width": width,
                "generated_descriptors_through_solving_layer": generated,
                "signature_evaluations": signature_evaluations,
                "candidate_row_operations": row_operations,
                "selected_vector_row_operations": (
                    selected_vector_row_operations),
                "operand_vector_lookups": operand_lookups,
                "feature_builds": len(selected),
                "feature_row_statistic_observations": (
                    4 * len(rows) * len(selected)),
                "target_range_row_reads": 4 * len(rows) * len(selected),
                "label_counts": {
                    "positive": len(positives),
                    "negative": len(negatives),
                },
                "selected_counts": {
                    "positive": len(selected_pos),
                    "negative": len(selected_neg),
                },
                "selected_indices": selected,
                "selected_indices_sha256": _indices_hash(selected),
                "selected_indices_by_class": {
                    "positive": selected_pos, "negative": selected_neg,
                },
                "selected_indices_sha256_by_class": {
                    "positive": _indices_hash(selected_pos),
                    "negative": _indices_hash(selected_neg),
                },
                "sampling_recipe": SAMPLE_RECIPE,
                "sampling_seed": sampling_seed,
                "positive_cap": positive_cap,
                "negative_cap": negative_cap,
                "selected_candidates": selected_records,
                "setup_seconds": time.perf_counter() - task_started,
                "error": "",
            }
        except Exception as exc:
            groups.append([])
            report = {
                "task_id": task.get("id"),
                "status": "failed",
                "solving_layer": solved_layer or None,
                "full_layer_width": len(layer),
                "generated_descriptors_through_solving_layer": generated,
                "signature_evaluations": signature_evaluations,
                "candidate_row_operations": row_operations,
                "selected_vector_row_operations": 0,
                "operand_vector_lookups": operand_lookups,
                "feature_builds": 0,
                "feature_row_statistic_observations": 0,
                "target_range_row_reads": 0,
                "label_counts": {
                    "positive": sum(labels),
                    "negative": len(labels) - sum(labels),
                },
                "selected_counts": {"positive": 0, "negative": 0},
                "selected_indices": [],
                "selected_indices_sha256": _indices_hash([]),
                "selected_indices_by_class": {
                    "positive": [], "negative": [],
                },
                "selected_indices_sha256_by_class": {
                    "positive": _indices_hash([]),
                    "negative": _indices_hash([]),
                },
                "sampling_recipe": SAMPLE_RECIPE,
                "sampling_seed": sampling_seed,
                "positive_cap": positive_cap,
                "negative_cap": negative_cap,
                "selected_candidates": [],
                "setup_seconds": time.perf_counter() - task_started,
                "error": type(exc).__name__ + ": " + str(exc),
            }
        task_reports.append(report)
        aggregate.update({
            "groups": 1,
            "successful_groups": int(report["status"] == "ok"),
            "failed_groups": int(report["status"] != "ok"),
            "full_layer_candidates": report["full_layer_width"],
            "selected_candidates": len(report["selected_indices"]),
            "positive_labels": report["label_counts"]["positive"],
            "negative_labels": report["label_counts"]["negative"],
        })
    return groups, {
        "label_semantics": "first-solving-layer-on-training-observed-rows",
        "groups": task_reports,
        "totals": dict(aggregate),
        "setup_seconds": time.perf_counter() - setup_started,
    }


def _apply_vector(op: str, left: tuple[int, ...],
                  right: tuple[int, ...]) -> tuple[int, ...]:
    if op == "add":
        return tuple(a + b for a, b in zip(left, right))
    if op == "sub":
        return tuple(a - b for a, b in zip(left, right))
    return tuple(a * b for a, b in zip(left, right))
