"""Deterministic, solver-independent generation for benchmark corpus v2."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from collections.abc import Iterator
from typing import Any

OPERATORS = ("add", "sub", "mul")
TERMINALS = ("x", "y")
FAMILIES = ("balanced", "repeated_subtree", "skewed")
RESERVATION_ORDER = ("repeated_subtree", "balanced", "skewed")
FAMILY_SEEDS = {"repeated_subtree": 82071, "balanced": 82072,
                "skewed": 82073}
SPLIT_SEED = 82074
INPUT_SEED = 82075
TASK_COUNTS = {"train": 24, "eval": 48}
PER_FAMILY = {"train": 8, "eval": 16}
GRID = tuple(itertools.product(range(-32, 33), repeat=2))

Node = dict[str, Any]
Polynomial = tuple[tuple[int, int, int], ...]


def leaf(name: str) -> Node:
    return {"op": "input", "name": name}


def apply(op: str, left: Node, right: Node) -> Node:
    return {"op": op, "left": left, "right": right}


def expression_key(node: Node) -> str:
    if node["op"] == "input":
        return str(node["name"])
    return (f"{node['op']}({expression_key(node['left'])},"
            f"{expression_key(node['right'])})")


def tree_ops(node: Node) -> int:
    return 0 if node["op"] == "input" else (
        1 + tree_ops(node["left"]) + tree_ops(node["right"]))


def _poly_add(left: dict[tuple[int, int], int],
              right: dict[tuple[int, int], int], factor: int = 1
              ) -> dict[tuple[int, int], int]:
    result = dict(left)
    for monomial, coefficient in right.items():
        result[monomial] = result.get(monomial, 0) + factor * coefficient
        if result[monomial] == 0:
            del result[monomial]
    return result


def _poly_mul(left: dict[tuple[int, int], int],
              right: dict[tuple[int, int], int]
              ) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for (lx, ly), lc in left.items():
        for (rx, ry), rc in right.items():
            key = (lx + rx, ly + ry)
            result[key] = result.get(key, 0) + lc * rc
    return {key: value for key, value in result.items() if value}


def polynomial(node: Node) -> Polynomial:
    """Return exact canonical integer coefficients in x/y exponent order."""
    if node["op"] == "input":
        return ((1, 0, 1),) if node["name"] == "x" else ((0, 1, 1),)
    left = dict(((x, y), coefficient)
                for x, y, coefficient in polynomial(node["left"]))
    right = dict(((x, y), coefficient)
                 for x, y, coefficient in polynomial(node["right"]))
    if node["op"] == "add":
        result = _poly_add(left, right)
    elif node["op"] == "sub":
        result = _poly_add(left, right, -1)
    elif node["op"] == "mul":
        result = _poly_mul(left, right)
    else:
        raise ValueError(f"unsupported operation: {node['op']}")
    return tuple((x, y, value) for (x, y), value in sorted(result.items()))


def degree(poly: Polynomial) -> int:
    return max((x + y for x, y, _ in poly), default=-1)


def l1_norm(poly: Polynomial) -> int:
    return sum(abs(coefficient) for _, _, coefficient in poly)


def meets_difficulty_floor(poly: Polynomial) -> bool:
    """Conservative algebraic floor excluding every <=3-op polynomial.

    A <=3-op tree has degree at most 4 and coefficient L1 norms bounded by
    1 at degree 4, 2 at degree 3, and 4 at degree 2. This floor rejects each
    such envelope while requiring degree >=2.
    """
    deg = degree(poly)
    norm = l1_norm(poly)
    return (deg >= 2 and (deg >= 5 or
                          (deg == 4 and norm > 1) or
                          (deg == 3 and norm > 2) or
                          (deg == 2 and norm > 4)))


def _shapes(ops: int) -> Iterator[Node]:
    if ops == 0:
        yield leaf("_")
        return
    for left_ops in range(ops):
        right_ops = ops - 1 - left_ops
        for left in _shapes(left_ops):
            for right in _shapes(right_ops):
                for op in OPERATORS:
                    yield apply(op, left, right)


def _family_matches(family: str, node: Node) -> bool:
    if family == "repeated_subtree":
        counts = Counter(_nonleaf_keys(node))
        return any(count >= 2 for count in counts.values())
    if family == "balanced":
        return (node["op"] != "input" and
                tree_ops(node["left"]) == tree_ops(node["right"]))
    if family == "skewed":
        return (node["op"] != "input" and
                abs(tree_ops(node["left"]) - tree_ops(node["right"])) >= 2)
    raise ValueError(f"unknown family: {family}")


def _digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def candidate_pools() -> tuple[dict[str, list[Node]], dict[str, int]]:
    """Enumerate syntax witnesses and retain first witness per polynomial."""
    pools = {family: [] for family in FAMILIES}
    reasons = {"semantic_duplicates": 0, "difficulty_floor": 0}
    seen: set[Polynomial] = set()
    for family in RESERVATION_ORDER:
        candidates = []
        for ops in (4, 5):
            for shape in _shapes(ops):
                leaves = [n for n in _leaf_paths(shape)]
                for names in itertools.product(TERMINALS, repeat=len(leaves)):
                    node = _fill_leaves(shape, iter(names))
                    if _family_matches(family, node):
                        candidates.append(node)
        candidates.sort(key=lambda node: _digest(
            FAMILY_SEEDS[family], expression_key(node)))
        for node in candidates:
            poly = polynomial(node)
            if poly in seen:
                reasons["semantic_duplicates"] += 1
                continue
            if not meets_difficulty_floor(poly):
                reasons["difficulty_floor"] += 1
                continue
            seen.add(poly)
            pools[family].append(node)
    return pools, reasons


def _leaf_paths(node: Node) -> Iterator[None]:
    if node["op"] == "input":
        yield None
    else:
        yield from _leaf_paths(node["left"])
        yield from _leaf_paths(node["right"])


def _nonleaf_keys(node: Node) -> Iterator[str]:
    if node["op"] == "input":
        return
    yield expression_key(node)
    yield from _nonleaf_keys(node["left"])
    yield from _nonleaf_keys(node["right"])


def _fill_leaves(node: Node, names: Iterator[str]) -> Node:
    if node["op"] == "input":
        return leaf(next(names))
    return apply(node["op"], _fill_leaves(node["left"], names),
                 _fill_leaves(node["right"], names))


def _evaluate(poly: Polynomial, x: int, y: int) -> int:
    return sum(coefficient * (x ** x_power) * (y ** y_power)
               for x_power, y_power, coefficient in poly)


def generate_corpus() -> dict[str, Any]:
    pools, rejects = candidate_pools()
    for family in RESERVATION_ORDER:
        needed = sum(PER_FAMILY.values())
        if len(pools[family]) < needed:
            raise RuntimeError(
                f"insufficient {family} pool: {len(pools[family])} < {needed}")
    selected: list[tuple[str, Node]] = []
    for family in RESERVATION_ORDER:
        ranked = sorted(pools[family], key=lambda node: _digest(
            SPLIT_SEED, f"{family}:{expression_key(node)}"))
        four_ops = [node for node in ranked if tree_ops(node) == 4]
        five_ops = [node for node in ranked if tree_ops(node) == 5]
        quota = sum(PER_FAMILY.values())
        four_quota = min(8, len(four_ops))
        chosen = four_ops[:four_quota] + five_ops[:quota - four_quota]
        if len(chosen) < quota:
            chosen += four_ops[four_quota:four_quota + quota - len(chosen)]
        selected.extend((family, node) for node in chosen)
    split_order = sorted(selected, key=lambda item: _digest(
        SPLIT_SEED, f"split:{item[0]}:{expression_key(item[1])}"))
    counts = {"train": {family: 0 for family in FAMILIES},
              "eval": {family: 0 for family in FAMILIES}}
    tasks: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    input_pairs = sorted(GRID, key=lambda pair: _digest(
        INPUT_SEED, f"{pair[0]}:{pair[1]}"))
    next_input = 0
    for family, node in split_order:
        split = ("train" if counts["train"][family] <
                 PER_FAMILY["train"] else "eval")
        counts[split][family] += 1
        number = len(tasks[split]) + 1
        task_id = f"v2_{split}_{number:02d}"
        task_inputs = input_pairs[next_input:next_input + 24]
        next_input += 24
        rows = [[x, y, _evaluate(polynomial(node), x, y)]
                for x, y in task_inputs]
        tasks[split].append({
            "id": task_id,
            "family": family,
            "train": rows[:12],
            "held_out": rows[12:],
            "max_tree_ops": 5,
            "generator": {
                "seed": FAMILY_SEEDS[family],
                "program": node,
                "program_key": expression_key(node),
                "canonical_polynomial": [list(term)
                                          for term in polynomial(node)],
                "tree_ops": tree_ops(node),
                "witness_ops": tree_ops(node),
                "degree": degree(polynomial(node)),
                "coefficient_l1": l1_norm(polynomial(node)),
            },
        })
    return {
        "corpus_id": "benchmark-pre-eval-v2",
        "generator": "exact-integer-polynomial-enumerator-v2",
        "generation_acceptance": (
            "all selected functions retained; no solver or search-outcome "
            "filtering"),
        "seeds": {"families": FAMILY_SEEDS, "split": SPLIT_SEED,
                  "inputs": INPUT_SEED},
        "generation_audit": {
            "candidate_pool_counts": {key: len(value)
                                       for key, value in pools.items()},
            "rejected_counts": rejects,
            "selected_witness_ops": {
                str(size): sum(
                    task["generator"]["tree_ops"] == size
                    for split_tasks in tasks.values()
                    for task in split_tasks)
                for size in (4, 5)},
            "input_grid": [-32, 32],
            "difficulty_floor": (
                "degree>=2 and (degree>=5 or degree4 L1>1 or "
                "degree3 L1>2 or degree2 L1>4)"),
        },
        "dsl": {"operators": list(OPERATORS), "terminals": list(TERMINALS),
                "max_tree_ops": 5},
        "splits": tasks,
    }


def canonical_json(corpus: dict[str, Any]) -> str:
    return json.dumps(corpus, indent=2, sort_keys=False) + "\n"
