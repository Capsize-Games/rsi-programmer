"""Function-disjoint construction tasks with deterministic teacher traces."""
from __future__ import annotations

import hashlib
import itertools
import random
from typing import Any

from rsi_programmer.benchmark_v2_corpus import (
    Node, Polynomial, apply, expression_key, leaf, polynomial,
)
from rsi_programmer.construction_state import Action, ConstructionState
from rsi_programmer.probe import Expr, lower_to_dag

SEED = 20261101
OPERATORS = ("add", "sub", "mul")
BOUND = 4
OBSERVED = tuple(itertools.product(range(-3, 3), repeat=2))
QUOTAS = {
    1: {"train": 6, "dev": 1, "eval": 1},
    2: {"train": 28, "dev": 6, "eval": 7},
    3: {"train": 128, "dev": 32, "eval": 44},
    4: {"train": 384, "dev": 96, "eval": 192},
}


def _digest(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def catalog() -> dict[int, list[Node]]:
    """Enumerate functions by minimal tree size, without solver filtering."""
    levels = {0: [leaf("x"), leaf("y")]}
    seen = {polynomial(node) for node in levels[0]}
    for size in range(1, BOUND + 1):
        current: list[Node] = []
        for left_size in range(size):
            for left in levels[left_size]:
                for right in levels[size - left_size - 1]:
                    for op in OPERATORS:
                        node = apply(op, left, right)
                        identity = polynomial(node)
                        if identity not in seen:
                            seen.add(identity)
                            current.append(node)
        levels[size] = current
    return levels


def _expression(node: Node) -> Expr:
    if node["op"] == "input":
        return Expr("input", node["name"])
    left, right = _expression(node["left"]), _expression(node["right"])
    return Expr(node["op"], left, right, left.size + right.size + 1)


def _value(identity: Polynomial, x: int, y: int) -> int:
    return sum(c * x ** xp * y ** yp for xp, yp, c in identity)


def generate_corpus() -> dict[str, Any]:
    """Freeze observed/held rows and splits before model fitting."""
    levels = catalog()
    splits: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("train", "dev", "eval")
    }
    available: dict[str, int] = {}
    for size, quotas in QUOTAS.items():
        pool = [node for node in levels[size]
                if any(xp or yp for xp, yp, _ in polynomial(node))]
        pool.sort(key=lambda node: _digest(expression_key(node)))
        available[str(size)] = len(pool)
        if len(pool) < sum(quotas.values()):
            raise ValueError("insufficient function pool")
        cursor = 0
        for split, count in quotas.items():
            for node in pool[cursor:cursor + count]:
                identity = polynomial(node)
                key = _digest(str(identity))
                rows = list(itertools.product(range(-6, 7), repeat=2))
                rows = [pair for pair in rows if pair not in OBSERVED]
                random.Random(int(key, 16)).shuffle(rows)
                observed = [[x, y, _value(identity, x, y)]
                            for x, y in OBSERVED]
                held = [[x, y, _value(identity, x, y)]
                        for x, y in rows[:24]]
                splits[split].append({
                    "id": key[:16], "split": split,
                    "observed": observed, "held": held,
                    "max_tree_ops": BOUND,
                    "generator": {
                        "minimal_tree_ops": size,
                        "program": lower_to_dag(_expression(node)),
                        "polynomial": [list(term) for term in identity],
                    },
                })
            cursor += count
        random.Random(SEED + size).shuffle(splits["train"])
    return {
        "corpus_id": "construction-v1", "seed": SEED,
        "grammar": {"operators": list(OPERATORS), "terminals": ["x", "y"],
                    "max_tree_ops": BOUND},
        "generation_acceptance": (
            "First syntax witness per exact polynomial at minimal tree "
            "size; exclude constant/identity targets; hash-order quotas. "
            "No evaluated-solver success or neural scores used."
        ),
        "available_functions_by_size": available,
        "quotas": QUOTAS, "splits": splits,
    }


def teacher_steps(task: dict[str, Any]
                  ) -> list[tuple[ConstructionState, Action]]:
    """Replay the generator witness using observed execution only."""
    if task["split"] != "train":
        raise ValueError("teacher traces are restricted to training tasks")
    state = ConstructionState(tuple(tuple(row) for row in task["observed"]),
                              task["max_tree_ops"])
    examples: list[tuple[ConstructionState, Action]] = []
    for node in task["generator"]["program"]["nodes"]:
        slots = [0 if ref == "x" else 1 if ref == "y" else ref + 2
                 for ref in node["args"]]
        action: Action = {"op": node["op"],
                          "left": slots[0], "right": slots[1]}
        examples.append((state, action))
        state = state.apply(action)
    if not state.observed_match():
        raise ValueError("teacher witness does not meet its specification")
    examples.append((state, {"op": "stop"}))
    return examples
