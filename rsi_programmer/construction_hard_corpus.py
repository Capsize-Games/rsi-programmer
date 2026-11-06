"""Hard construction bank at exact minimal tree-op counts."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any

from rsi_programmer.benchmark_v2_corpus import (
    Node, Polynomial, apply, leaf, polynomial,
)
from rsi_programmer.probe import Expr, lower_to_dag

SEED = 20261106
BOUND = 8
STRATA = (6, 7, 8)
OPERATORS = ("add", "sub", "mul")
TERMINALS = ("x", "y")
QUOTAS = {"train": 1200, "dev": 200, "eval": 300}
EARLY_STOP_MULTIPLE = 4
OBSERVED = tuple(itertools.product(range(-3, 3), repeat=2))
OBSERVED_SET = frozenset(OBSERVED)
HELD_GRID = tuple(itertools.product(range(-6, 7), repeat=2))
V1_CORPUS = "fixtures/construction_corpus_v1.json"
IDENTITY = (((1, 0, 1),), ((0, 1, 1),))
ACCEPTANCE = (
    "Exact level-by-level catalog over add/sub/mul and x/y; "
    "stratum is the minimal tree-op count of first appearance "
    "at levels 6, 7, 8; constant/bare-input and v1-corpus "
    "polynomials excluded; level 8 early-stops past a recorded "
    "multiple of quota with the seen set complete below 8. No "
    "evaluated-solver success or neural scores used."
)
Pools = dict[int, list[tuple[Polynomial, Node]]]
_LEVELS: dict[int, list[Node]] | None = None
_SEEN: set[Polynomial] | None = None


def _digest(seed: int, value: str) -> str:
    """Hash one seed-scoped value for deterministic ordering."""
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _grow(levels: dict[int, list[Node]], seen: set[Polynomial],
          size: int, target: int | None = None
          ) -> tuple[list[Node], int, bool]:
    """Grow one level; stop early once target new identities appear."""
    current: list[Node] = []
    tried = 0
    for left_size in range(size):
        for left in levels[left_size]:
            for right in levels[size - left_size - 1]:
                for op in OPERATORS:
                    tried += 1
                    node = apply(op, left, right)
                    identity = polynomial(node)
                    if identity not in seen:
                        seen.add(identity)
                        current.append(node)
                        if target is not None and len(current) >= target:
                            return current, tried, True
    return current, tried, False


def catalog_levels() -> tuple[dict[int, list[Node]], set[Polynomial]]:
    """Return cached complete levels 0..7; level 8 stays per-call."""
    global _LEVELS, _SEEN
    if _LEVELS is None or _SEEN is None:
        levels = {0: [leaf("x"), leaf("y")]}
        seen = {polynomial(node) for node in levels[0]}
        for size in range(1, BOUND):
            current, _, _ = _grow(levels, seen, size)
            levels[size] = current
        _LEVELS, _SEEN = levels, seen
    return _LEVELS, _SEEN


def load_v1_identities(path: str | Path | None) -> set[Polynomial]:
    """Load every v1 polynomial identity for exclusion, no task rows."""
    if path is None:
        return set()
    corpus = json.loads(Path(path).read_text())
    found: set[Polynomial] = set()
    for tasks in corpus["splits"].values():
        for task in tasks:
            poly = task["generator"]["polynomial"]
            found.add(tuple(tuple(term) for term in poly))
    return found


def stratum_share(total: int) -> dict[int, int]:
    """Split a quota evenly across strata, remainder lowest-first."""
    base, rest = divmod(total, len(STRATA))
    return {ops: base + (1 if pos < rest else 0)
            for pos, ops in enumerate(STRATA)}


def _pool(nodes: list[Node], v1: set[Polynomial]
          ) -> tuple[list[tuple[Polynomial, Node]], int]:
    """Keep nontrivial identities outside the v1 corpus."""
    pairs = [(polynomial(node), node) for node in nodes]
    kept = [pair for pair in pairs if pair[0]
            and pair[0] not in IDENTITY and pair[0] not in v1]
    return kept, len(pairs) - len(kept)


def _select(pool: list[tuple[Polynomial, Node]],
            seed: int) -> list[tuple[Polynomial, Node]]:
    """Order witnesses by seed digest for split assignment."""
    return sorted(pool, key=lambda item: _digest(seed, str(item[0])))


def _expression(node: Node) -> Expr:
    """Convert a catalog node to a probe expression tree."""
    if node["op"] == "input":
        return Expr("input", node["name"])
    left = _expression(node["left"])
    right = _expression(node["right"])
    return Expr(node["op"], left, right, left.size + right.size + 1)


def _value(identity: Polynomial, x: int, y: int) -> int:
    """Evaluate one exact polynomial identity at integer inputs."""
    return sum(c * x ** xp * y ** yp for xp, yp, c in identity)


def _task(identity: Polynomial, node: Node, ops: int,
          split: str, seed: int) -> dict[str, Any]:
    """Freeze observed/held rows and the minimal-size witness."""
    key = _digest(seed, str(identity))
    rng = random.Random(int(key, 16))
    rows = [pair for pair in HELD_GRID if pair not in OBSERVED_SET]
    rng.shuffle(rows)
    observed = [[x, y, _value(identity, x, y)] for x, y in OBSERVED]
    held = [[x, y, _value(identity, x, y)] for x, y in rows[:24]]
    return {
        "id": key[:16], "split": split, "observed": observed,
        "held": held, "max_tree_ops": BOUND,
        "generator": {
            "program": lower_to_dag(_expression(node)),
            "polynomial": [list(term) for term in identity],
            "minimal_tree_ops": ops,
        },
    }


def _pools(levels: dict[int, list[Node]], seen: set[Polynomial],
           v1: set[Polynomial], shares: dict[str, dict[int, int]],
           seed: int) -> tuple[Pools, dict[str, int], dict[str, int],
                               dict[str, Any]]:
    """Filter levels 6/7 fully; collect level 8 with an early stop."""
    need = sum(share[BOUND] for share in shares.values())
    nodes, tried, stopped = _grow(
        levels, set(seen), BOUND, EARLY_STOP_MULTIPLE * need)
    record = {"need": need, "multiple": EARLY_STOP_MULTIPLE,
              "examined": tried, "collected": len(nodes),
              "early_stop": stopped}
    pools: Pools = {}
    excluded: dict[str, int] = {}
    for ops in STRATA:
        kept, dropped = _pool(nodes if ops == BOUND else levels[ops], v1)
        excluded[str(ops)] = dropped
        pools[ops] = _select(kept, seed)
    sizes = {str(size): len(levels[size]) for size in range(BOUND)}
    sizes[str(BOUND)] = len(nodes)
    return pools, excluded, sizes, record


def _assign(pools: Pools, shares: dict[str, dict[int, int]],
            wanted: dict[str, int], seed: int
            ) -> tuple[dict[str, list[dict[str, Any]]],
                       dict[str, dict[str, int]]]:
    """Deal ordered witnesses to splits; record stratum shortfalls."""
    splits: dict[str, list[dict[str, Any]]] = {n: [] for n in wanted}
    shortfalls: dict[str, dict[str, int]] = {n: {} for n in wanted}
    for ops in STRATA:
        cursor = 0
        for name in wanted:
            count = shares[name][ops]
            chosen = pools[ops][cursor:cursor + count]
            cursor += count
            shortfalls[name][str(ops)] = count - len(chosen)
            splits[name].extend(_task(identity, node, ops, name, seed)
                                for identity, node in chosen)
    return splits, shortfalls


def _static(seed: int, wanted: dict[str, int],
            shares: dict[str, dict[int, int]]) -> dict[str, Any]:
    """Report the quota-independent bank description."""
    return {
        "corpus_id": "construction-hard-v1", "seed": seed,
        "grammar": {"operators": list(OPERATORS),
                    "terminals": list(TERMINALS),
                    "max_tree_ops": BOUND},
        "generation_acceptance": ACCEPTANCE,
        "strata": list(STRATA),
        "quotas": wanted,
        "stratum_quotas": {name: {str(ops): share[ops] for ops in STRATA}
                           for name, share in shares.items()},
    }


def _dynamic(pools: Pools, excluded: dict[str, int],
             sizes: dict[str, int], record: dict[str, Any],
             v1: set[Polynomial], v1_path: str | Path | None
             ) -> dict[str, Any]:
    """Report pool counts, the level census, and exclusion counts."""
    return {
        "available_functions_by_stratum": {
            str(ops): len(pool) for ops, pool in pools.items()},
        "catalog_level_sizes": sizes,
        "level_8_collection": record,
        "v1_exclusion": {
            "v1_path": None if v1_path is None else str(v1_path),
            "v1_polynomials": len(v1),
            "excluded_identities_by_stratum": excluded,
        },
    }


def generate_corpus(seed: int = SEED,
                    quotas: dict[str, int] | None = None,
                    v1_path: str | Path | None = V1_CORPUS
                    ) -> dict[str, Any]:
    """Freeze minimal-size tasks with strata-even quotas before fit."""
    wanted = dict(quotas) if quotas else dict(QUOTAS)
    shares = {name: stratum_share(total)
              for name, total in wanted.items()}
    levels, seen = catalog_levels()
    v1 = load_v1_identities(v1_path)
    pools, excluded, sizes, record = _pools(
        levels, seen, v1, shares, seed)
    splits, shortfalls = _assign(pools, shares, wanted, seed)
    return {**_static(seed, wanted, shares),
            **_dynamic(pools, excluded, sizes, record, v1, v1_path),
            "shortfalls": shortfalls, "splits": splits}
