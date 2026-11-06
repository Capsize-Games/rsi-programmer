"""Shape-broad scaling bank at minimal tree-op counts 6/7/8."""
from __future__ import annotations

import hashlib
import itertools
import random
from typing import Any

from rsi_programmer.benchmark_v2_corpus import (
    Node,
    Polynomial,
    apply,
    polynomial,
)
from rsi_programmer.construction_hard_corpus import (
    BOUND,
    IDENTITY,
    OPERATORS,
    TERMINALS,
    V1_CORPUS,
    catalog_levels,
    load_v1_identities,
)
from rsi_programmer.probe import Expr, lower_to_dag

SEED = 20261109
STRATA = (6, 7, 8)
QUOTAS = {"train": 4800, "eval": 12000}
OBSERVED = tuple(itertools.product(range(-3, 3), repeat=2))
OBSERVED_SET = frozenset(OBSERVED)
HELD_GRID = tuple(itertools.product(range(-6, 7), repeat=2))
ATTEMPT_CAP = 100_000
ACCEPTANCE = (
    "Complete catalog levels 6/7 digest-ordered; level 8 "
    "rejection-sampled balanced across left-subtree sizes 0..7; "
    "minimality exact because the seen set is complete below 8; "
    "exclusions by polynomial identity; no solver or neural "
    "scores used."
)
Pools = dict[int, list[tuple[Polynomial, Node]]]
Shares = dict[str, dict[int, int]]


def _digest(seed: int, value: str) -> str:
    """Hash one seed-scoped value for deterministic ordering."""
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


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


def _make_task(identity: Polynomial, node: Node, ops: int,
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


def _share(total: int) -> dict[int, int]:
    """Split a quota evenly across strata, remainder lowest-first."""
    base, rest = divmod(total, len(STRATA))
    return {ops: base + (1 if pos < rest else 0)
            for pos, ops in enumerate(STRATA)}


def _filtered(nodes: list[Node], excluded: set[Polynomial]
              ) -> tuple[list[tuple[Polynomial, Node]], int]:
    """Keep nontrivial identities outside the exclusion set."""
    pairs = [(polynomial(node), node) for node in nodes]
    kept = [pair for pair in pairs if pair[0]
            and pair[0] not in IDENTITY and pair[0] not in excluded]
    return kept, len(pairs) - len(kept)


def _ordered(pairs: list[tuple[Polynomial, Node]], seed: int
             ) -> list[tuple[Polynomial, Node]]:
    """Order witnesses by seed digest for split assignment."""
    return sorted(pairs, key=lambda item: _digest(seed, str(item[0])))


def _sample_block(k: int, quota: int, rng: random.Random,
                  taken: set[Polynomial], excluded: set[Polynomial]
                  ) -> tuple[list[tuple[Polynomial, Node]], int]:
    """Rejection-sample stratum-8 witnesses with left size k."""
    levels, _ = catalog_levels()
    lefts, rights = levels[k], levels[BOUND - 1 - k]
    found: list[tuple[Polynomial, Node]] = []
    attempts = 0
    while len(found) < quota and attempts < ATTEMPT_CAP:
        attempts += 1
        idx = rng.randrange(len(lefts) * len(rights) * len(OPERATORS))
        node = apply(OPERATORS[idx % 3], lefts[idx // 3 // len(rights)],
                     rights[idx // 3 % len(rights)])
        identity = polynomial(node)
        if (not identity or identity in IDENTITY
                or identity in taken or identity in excluded):
            continue
        taken.add(identity)
        found.append((identity, node))
    return found, attempts


def _sample_level8(need: int, seed: int, excluded: set[Polynomial]
                   ) -> tuple[list[tuple[Polynomial, Node]],
                              dict[str, Any]]:
    """Block-balanced stratum-8 pool, digest-ordered, truncated."""
    _, seen = catalog_levels()
    rng = random.Random(seed)
    taken = set(seen)
    quota = -(-need // BOUND)
    pairs: list[tuple[Polynomial, Node]] = []
    blocks: list[dict[str, int]] = []
    for k in range(BOUND):
        found, attempts = _sample_block(k, quota, rng, taken, excluded)
        pairs.extend(found)
        blocks.append({"left": k, "accepted": len(found),
                       "attempts": attempts})
    rejected = sum(b["attempts"] - b["accepted"] for b in blocks)
    record = {"need": need, "per_block_quota": quota, "blocks": blocks,
              "collected": len(pairs), "rejected": rejected,
              "kept": min(need, len(pairs))}
    return _ordered(pairs, seed)[:need], record


def _pools(seed: int, shares: Shares, excluded: set[Polynomial]
           ) -> tuple[Pools, dict[str, int], dict[str, int],
                      dict[str, Any]]:
    """Digest-ordered pools per stratum with exclusion counts."""
    levels, _ = catalog_levels()
    need8 = sum(share[BOUND] for share in shares.values())
    pools: Pools = {}
    dropped: dict[str, int] = {}
    record: dict[str, Any] = {}
    for ops in STRATA:
        if ops == BOUND:
            pools[ops], record = _sample_level8(need8, seed, excluded)
            dropped[str(ops)] = record["rejected"]
        else:
            kept, count = _filtered(levels[ops], excluded)
            pools[ops] = _ordered(kept, seed)
            dropped[str(ops)] = count
    sizes = {str(size): len(levels[size]) for size in range(BOUND)}
    sizes[str(BOUND)] = record["collected"]
    return pools, dropped, sizes, record


def _assign(pools: Pools, shares: Shares, wanted: dict[str, int],
            seed: int) -> tuple[dict[str, list[dict[str, Any]]],
                                dict[str, dict[str, int]]]:
    """Deal digest-ordered witnesses; shuffle eval across strata."""
    store: dict[str, list[dict[str, Any]]] = {n: [] for n in wanted}
    shortfalls: dict[str, dict[str, int]] = {n: {} for n in wanted}
    for ops in STRATA:
        cursor = 0
        for name in wanted:
            count = shares[name][ops]
            chosen = pools[ops][cursor:cursor + count]
            cursor += count
            shortfalls[name][str(ops)] = count - len(chosen)
            store[name].extend(_make_task(p, n, ops, name, seed)
                               for p, n in chosen)
    random.Random(seed).shuffle(store["eval"])
    return store, shortfalls


def _static(seed: int, wanted: dict[str, int],
            shares: Shares) -> dict[str, Any]:
    """Report the quota-independent bank description."""
    return {
        "corpus_id": "construction-scaling-v1", "seed": seed,
        "grammar": {"operators": list(OPERATORS),
                    "terminals": list(TERMINALS),
                    "max_tree_ops": BOUND},
        "generation_acceptance": ACCEPTANCE,
        "strata": list(STRATA), "quotas": wanted,
        "stratum_quotas": {name: {str(ops): share[ops]
                                  for ops in STRATA}
                           for name, share in shares.items()},
    }


def generate_corpus(seed: int = SEED,
                    quotas: dict[str, int] | None = None,
                    excluded: set[Polynomial] | None = None
                    ) -> dict[str, Any]:
    """Freeze shape-broad tasks with strata-even quotas before fit."""
    wanted = dict(quotas) if quotas else dict(QUOTAS)
    shares = {name: _share(total) for name, total in wanted.items()}
    if excluded is None:
        excluded = load_v1_identities(V1_CORPUS)
    pools, dropped, sizes, record = _pools(seed, shares, excluded)
    splits, shortfalls = _assign(pools, shares, wanted, seed)
    return {
        **_static(seed, wanted, shares),
        "available_functions_by_stratum": {
            str(ops): len(pool) for ops, pool in pools.items()},
        "catalog_level_sizes": sizes,
        "level_8_sampling": record,
        "exclusion_counts": dropped, "shortfalls": shortfalls,
        "excluded_identities": len(excluded), "splits": splits,
    }
