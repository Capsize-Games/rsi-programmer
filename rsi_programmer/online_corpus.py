"""Deterministic synthetic task corpus for online graph search."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any

OPERATORS = ("add", "sub", "mul")
CONSTANTS = (2, 3, 4, 5)
CORPUS_SEED = 83074
ROOT_QUOTAS = (
    {"add": 12, "sub": 2, "mul": 2},
    {"add": 2, "sub": 12, "mul": 2},
    {"add": 2, "sub": 2, "mul": 12},
)
PRIMARY_BUDGET = 144
SECONDARY_BUDGET = 432
TASKS_PER_PHASE = 16
Polynomial = tuple[tuple[int, int, int], ...]
PolyMap = dict[tuple[int, int], int]


def _clean(poly: PolyMap) -> PolyMap:
    """Remove zero coefficients from a polynomial map."""
    return {key: value for key, value in poly.items() if value}


def _apply(operator: str, left: PolyMap,
           right: PolyMap) -> PolyMap:
    """Add, subtract, or multiply two sparse integer polynomials."""
    result: PolyMap = defaultdict(int)
    if operator == "mul":
        for (left_x, left_y), left_value in left.items():
            for (right_x, right_y), right_value in right.items():
                key = left_x + right_x, left_y + right_y
                result[key] += left_value * right_value
    elif operator in {"add", "sub"}:
        result.update(left)
        sign = 1 if operator == "add" else -1
        for key, value in right.items():
            result[key] += sign * value
    else:
        raise ValueError("unknown polynomial operator")
    return _clean(result)


def _polynomial_key(poly: PolyMap) -> Polynomial:
    """Return an exact, sorted polynomial coefficient tuple."""
    return tuple((x_power, y_power, value)
                 for (x_power, y_power), value in sorted(poly.items())
                 if value)


def _leaf_poly(leaf: str, constant: int) -> PolyMap:
    """Return the polynomial for one symbolic leaf."""
    if leaf == "x":
        return {(1, 0): 1}
    if leaf == "y":
        return {(0, 1): 1}
    if leaf == "c":
        return {(0, 0): constant}
    raise ValueError("unknown polynomial leaf")


def _leaf_node(leaf: str, constant: int) -> dict[str, Any]:
    """Build an expression node for a symbolic leaf."""
    if leaf == "c":
        return {"op": "const", "value": constant}
    if leaf in {"x", "y"}:
        return {"input": leaf}
    raise ValueError("unknown polynomial leaf")


def _tree(operator: str, left: dict[str, Any],
          right: dict[str, Any]) -> dict[str, Any]:
    """Return a binary expression node."""
    return {"op": operator, "left": left, "right": right}


def _candidate_universe() -> tuple[
        dict[str, dict[Polynomial, dict[str, Any]]],
        dict[str, dict[Polynomial, int]], set[Polynomial], int]:
    """Enumerate grammar trees and retain unique-root polynomial witnesses."""
    roots_by_polynomial: dict[Polynomial, set[str]] = defaultdict(set)
    witnesses: dict[str, dict[Polynomial, dict[str, Any]]] = {
        operator: {} for operator in OPERATORS}
    positions: dict[str, dict[Polynomial, int]] = {
        operator: {} for operator in OPERATORS}
    lane_positions = {operator: 0 for operator in OPERATORS}

    for constant in CONSTANTS:
        for shape in ("left", "right"):
            for leaves in permutations(("x", "y", "c")):
                for inner in OPERATORS:
                    for root in OPERATORS:
                        lane_positions[root] += 1
                        first, second, third = leaves
                        first_poly = _leaf_poly(first, constant)
                        second_poly = _leaf_poly(second, constant)
                        third_poly = _leaf_poly(third, constant)
                        if shape == "left":
                            inner_poly = _apply(inner, first_poly,
                                                second_poly)
                            poly = _apply(root, inner_poly, third_poly)
                            tree = _tree(
                                root,
                                _tree(inner, _leaf_node(first, constant),
                                      _leaf_node(second, constant)),
                                _leaf_node(third, constant))
                        else:
                            inner_poly = _apply(inner, second_poly,
                                                third_poly)
                            poly = _apply(root, first_poly, inner_poly)
                            tree = _tree(
                                root, _leaf_node(first, constant),
                                _tree(inner, _leaf_node(second, constant),
                                      _leaf_node(third, constant)))
                        key = _polynomial_key(poly)
                        roots_by_polynomial[key].add(root)
                        witnesses[root].setdefault(key, tree)
                        positions[root].setdefault(
                            key, lane_positions[root])

    unique_witnesses: dict[str, dict[Polynomial, dict[str, Any]]] = {
        operator: {
            key: tree for key, tree in witnesses[operator].items()
            if roots_by_polynomial[key] == {operator}
        }
        for operator in OPERATORS
    }
    unique_positions: dict[str, dict[Polynomial, int]] = {
        operator: {
            key: positions[operator][key]
            for key in unique_witnesses[operator]
        }
        for operator in OPERATORS
    }
    return (unique_witnesses, unique_positions,
            set(roots_by_polynomial), sum(lane_positions.values()))


def evaluate_polynomial(polynomial: Polynomial, x: int, y: int) -> int:
    """Evaluate sparse polynomial coefficients with integer arithmetic."""
    return sum(coefficient * x ** x_power * y ** y_power
               for x_power, y_power, coefficient in polynomial)


def evaluate_witness(tree: dict[str, Any], x: int, y: int) -> int:
    """Evaluate a generated witness without using the search interpreter."""
    if "input" in tree:
        if tree["input"] == "x":
            return x
        if tree["input"] == "y":
            return y
        raise ValueError("unknown witness input")
    if tree.get("op") == "const":
        return int(tree["value"])
    operator = tree.get("op")
    left = evaluate_witness(tree["left"], x, y)
    right = evaluate_witness(tree["right"], x, y)
    if operator == "add":
        return left + right
    if operator == "sub":
        return left - right
    if operator == "mul":
        return left * right
    raise ValueError("unknown witness operator")


def _sample_inputs(seed: int) -> tuple[list[tuple[int, int]],
                                       list[tuple[int, int]]]:
    """Draw one shared input set independently of target labels."""
    inputs = [(x, y) for x in range(-9, 10) for y in range(-9, 10)]
    selected = random.Random(seed).sample(inputs, 38)
    return selected[:6], selected[6:]


def _assert_targets_observed_separately(
        witnesses: dict[str, dict[Polynomial, dict[str, Any]]],
        all_polynomials: set[Polynomial],
        observed_inputs: list[tuple[int, int]]) -> None:
    """Ensure no other grammar polynomial aliases a target on train rows."""
    signatures: dict[tuple[int, ...], set[Polynomial]] = defaultdict(set)
    for polynomial in all_polynomials:
        signature = tuple(evaluate_polynomial(polynomial, x, y)
                          for x, y in observed_inputs)
        signatures[signature].add(polynomial)
    for targets in witnesses.values():
        for polynomial in targets:
            signature = tuple(evaluate_polynomial(polynomial, x, y)
                              for x, y in observed_inputs)
            if signatures[signature] != {polynomial}:
                raise RuntimeError(
                    "observed rows do not separate target polynomials")


def _task_id(rng: random.Random, used: set[str]) -> str:
    """Return a deterministic opaque identifier."""
    while True:
        value = f"{rng.getrandbits(64):016x}"
        if value not in used:
            used.add(value)
            return value


def build_online_corpus(corpus_seed: int = CORPUS_SEED
                        ) -> dict[str, Any]:
    """Create the frozen 48-task, three-phase synthetic corpus."""
    if type(corpus_seed) is not int or corpus_seed < 0:
        raise ValueError("corpus seed must be a non-negative integer")
    witnesses, positions, all_polynomials, candidate_count = (
        _candidate_universe())
    if candidate_count != 432:
        raise RuntimeError("candidate grammar size changed")
    expected_counts = {"add": 16, "sub": 36, "mul": 40}
    actual_counts = {operator: len(witnesses[operator])
                     for operator in OPERATORS}
    if actual_counts != expected_counts:
        raise RuntimeError("unique-root candidate pool changed")

    train_inputs, held_inputs = _sample_inputs(corpus_seed)
    _assert_targets_observed_separately(
        witnesses, all_polynomials, train_inputs)
    target_rng = random.Random(corpus_seed + 1)
    id_rng = random.Random(corpus_seed + 2)
    chosen: dict[str, list[Polynomial]] = {}
    for operator in OPERATORS:
        keys = sorted(witnesses[operator])
        if len(keys) < TASKS_PER_PHASE:
            raise RuntimeError("not enough distinct target polynomials")
        chosen[operator] = (keys if operator == "add" else
                            target_rng.sample(keys, TASKS_PER_PHASE))
        target_rng.shuffle(chosen[operator])

    tasks: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    assigned: dict[str, int] = {operator: 0 for operator in OPERATORS}
    for phase, quotas in enumerate(ROOT_QUOTAS):
        schedule = [operator for operator in OPERATORS
                    for _ in range(quotas[operator])]
        target_rng.shuffle(schedule)
        for root in schedule:
            polynomial = chosen[root][assigned[root]]
            assigned[root] += 1
            witness = witnesses[root][polynomial]
            task_id = _task_id(id_rng, used_ids)
            observed = [[x, y, evaluate_polynomial(polynomial, x, y)]
                        for x, y in train_inputs]
            held = [[x, y, evaluate_polynomial(polynomial, x, y)]
                    for x, y in held_inputs]
            tasks.append({
                "id": task_id,
                "phase": phase,
                "root": root,
                "polynomial": [list(term) for term in polynomial],
                "candidate_index": positions[root][polynomial],
                "witness": witness,
                "observed": observed,
                "held": held,
            })

    if assigned != {operator: TASKS_PER_PHASE for operator in OPERATORS}:
        raise RuntimeError("target assignment did not consume each pool")
    root_positions = {
        operator: sorted(positions[operator].values())
        for operator in OPERATORS
    }
    return {
        "schema_version": 1,
        "protocol": {
            "name": "online-graph-v1",
            "corpus_seed": corpus_seed,
            "candidate_universe": {
                "constants": list(CONSTANTS),
                "operators": list(OPERATORS),
                "tree_shapes": ["left", "right"],
                "permuted_leaves": ["x", "y", "c"],
                "syntactic_trees": candidate_count,
                "distinct_candidate_polynomials": len(all_polynomials),
                "lane_size": candidate_count // len(OPERATORS),
                "unique_polynomials_by_root": actual_counts,
                "unique_root_pool_sizes": actual_counts,
                "unique_target_positions_by_root": root_positions,
            },
            "phase_root_quotas": [dict(quotas)
                                  for quotas in ROOT_QUOTAS],
            "tasks_per_phase": TASKS_PER_PHASE,
            "primary_budget": PRIMARY_BUDGET,
            "secondary_budget": SECONDARY_BUDGET,
            "model_seeds": [0, 1, 2],
            "arms": ["persistent_snn", "frozen_snn",
                     "episode_reset_snn", "online_nonspiking",
                     "grammar_streaming"],
            "policy": {
                "input_features": 6,
                "hidden_units": 16,
                "steps": 6,
                "learning_rate": 0.05,
                "eligibility_decay": 0.0,
                "reward_baseline": 0.05,
                "weight_clip": 5.0,
                "update_clip": 0.05,
                "membrane_beta": 0.8,
                "threshold": 1.0,
                "initialization":
                "seeded random projection; no trained encoder",
                "credit_assignment":
                "current decision only; no cross-action trace",
            },
            "observed_input_pairs": [list(pair)
                                     for pair in train_inputs],
            "observed_rows_separate_all_targets": True,
            "held_input_pairs": [list(pair)
                                 for pair in held_inputs],
            "target_selection":
            "unique exact integer polynomials, one root lane only",
            "policy_visibility":
            "only the six-value state and available-lane mask",
            "held_feedback": "never returned to the policy",
            "acceptance_gates": {
                "persistent_vs_frozen_reset": {
                    "first_lane_accuracy_margin": 0.10,
                    "positive_margin_each_model_seed": True,
                    "accepted_tasks_at_least_each_control": True,
                },
                "persistent_vs_online_nonspiking": {
                    "first_lane_accuracy_margin": 0.10,
                    "accepted_tasks_at_least_control": True,
                    "total_cpu_at_most_control": True,
                },
                "streaming_deployment": {
                    "accepted_tasks_at_least_control": True,
                    "total_cpu_at_most_control": True,
                    "budgets": [PRIMARY_BUDGET, SECONDARY_BUDGET],
                },
            },
            "memory_reporting":
            "whole-run peak RSS and serialized model size only",
        },
        "tasks": tasks,
    }


def write_online_corpus(path: Path,
                        corpus_seed: int = CORPUS_SEED) -> str:
    """Write deterministic JSON and return its SHA-256 digest."""
    corpus = build_online_corpus(corpus_seed)
    payload = json.dumps(corpus, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
