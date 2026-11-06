"""Random hard construction bank at exact sampled tree-op counts."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any

from rsi_programmer.online_graph import Polynomial, graph_polynomial
from rsi_programmer.probe import Expr, lower_to_dag

SEED = 20261106
BOUND = 7
STRATA = (5, 6, 7)
OPERATORS = ("add", "sub", "mul")
TERMINALS = ("x", "y")
QUOTAS = {"train": 1200, "dev": 200, "eval": 300}
OBSERVED = tuple(itertools.product(range(-3, 3), repeat=2))
OBSERVED_SET = frozenset(OBSERVED)
HELD_GRID = tuple(itertools.product(range(-6, 7), repeat=2))
V1_CORPUS = "fixtures/construction_corpus_v1.json"
MAX_ATTEMPTS = 120000
IDENTITY = (((1, 0, 1),), ((0, 1, 1),))


def _digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def sample_expr(ops: int, rng: random.Random) -> Expr:
    """Sample a uniform random binary tree with exactly ops operators."""
    if ops == 0:
        return Expr("input", rng.choice(TERMINALS))
    left = sample_expr(rng.randrange(ops), rng)
    right = sample_expr(ops - 1 - left.size, rng)
    op = rng.choice(OPERATORS)
    return Expr(op, left, right, left.size + right.size + 1)


def expr_polynomial(expr: Expr) -> Polynomial:
    """Expand an Expr to its exact integer polynomial identity."""
    return graph_polynomial(lower_to_dag(expr))


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


def _value(identity: Polynomial, x: int, y: int) -> int:
    return sum(c * x ** xp * y ** yp for xp, yp, c in identity)


def _task(identity: Polynomial, expr: Expr, ops: int,
          split: str, seed: int) -> dict[str, Any]:
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
            "program": lower_to_dag(expr),
            "polynomial": [list(term) for term in identity],
            "sampled_ops": ops,
        },
    }


def _sample_stratum(ops: int, need: int, seed: int,
                    v1: set[Polynomial], taken: set[Polynomial],
                    budget: int) -> tuple[dict[Polynomial, Expr], int]:
    """Collect distinct identities outside v1 and earlier strata."""
    rng = random.Random(seed + ops)
    pool: dict[Polynomial, Expr] = {}
    excluded = 0
    attempts = 0
    while len(pool) < need and attempts < budget:
        attempts += 1
        expr = sample_expr(ops, rng)
        identity = expr_polynomial(expr)
        if identity in v1:
            excluded += 1
        elif (identity and identity not in IDENTITY
              and identity not in pool and identity not in taken):
            pool[identity] = expr
    return pool, excluded


def generate_corpus(seed: int = SEED,
                    quotas: dict[str, int] | None = None,
                    v1_path: str | Path | None = V1_CORPUS,
                    max_attempts: int = MAX_ATTEMPTS) -> dict[str, Any]:
    """Freeze random-tree tasks with strata-even quotas before fitting."""
    wanted = dict(quotas) if quotas else dict(QUOTAS)
    shares = {name: stratum_share(total)
              for name, total in wanted.items()}
    v1 = load_v1_identities(v1_path)
    splits: dict[str, list[dict[str, Any]]] = {n: [] for n in wanted}
    available: dict[str, int] = {}
    excluded: dict[str, int] = {}
    shortfalls: dict[str, dict[str, int]] = {n: {} for n in wanted}
    taken: set[Polynomial] = set()
    for ops in STRATA:
        need = sum(share[ops] for share in shares.values())
        pool, hits = _sample_stratum(ops, need, seed, v1, taken,
                                      max_attempts)
        taken.update(pool)
        excluded[str(ops)] = hits
        available[str(ops)] = len(pool)
        ordered = sorted(pool.items(),
                         key=lambda item: _digest(seed, str(item[0])))
        cursor = 0
        for name in wanted:
            count = shares[name][ops]
            chosen = ordered[cursor:cursor + count]
            cursor += count
            shortfalls[name][str(ops)] = count - len(chosen)
            splits[name].extend(_task(poly, expr, ops, name, seed)
                                for poly, expr in chosen)
    return {
        "corpus_id": "construction-hard-v1", "seed": seed,
        "grammar": {"operators": list(OPERATORS),
                    "terminals": list(TERMINALS),
                    "max_tree_ops": BOUND},
        "generation_acceptance": (
            "Random expression trees at exact sampled tree-op counts "
            "in {5, 6, 7}; dedupe and split by exact polynomial "
            "identity; constant/identity and v1-corpus polynomials "
            "excluded. No evaluated-solver success or neural scores "
            "used."
        ),
        "strata": list(STRATA),
        "quotas": wanted,
        "stratum_quotas": {name: {str(ops): share[ops] for ops in STRATA}
                           for name, share in shares.items()},
        "available_functions_by_stratum": available,
        "v1_exclusion": {
            "v1_path": None if v1_path is None else str(v1_path),
            "v1_polynomials": len(v1),
            "excluded_samples_by_stratum": excluded,
        },
        "shortfalls": shortfalls,
        "splits": splits,
    }
