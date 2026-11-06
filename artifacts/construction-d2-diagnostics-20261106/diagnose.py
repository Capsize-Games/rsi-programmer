"""D2 diagnostic: 19,200-task third scaling point.

Development data only. Same shape-broad pools and the same
300-task broad eval identities as D1 (those identities stay
excluded from training); one seed-0 100-epoch fit on the
19,200-task nested superset, evaluated at beam widths 8 and
32 with streaming at 0.5 CPU s/task as control. Per-task rows
saved. Weights saved to /tmp. Output: results.json here.
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rsi_programmer.benchmark_v2_corpus import (  # noqa: E402
    Node, Polynomial, apply, polynomial,
)
from rsi_programmer.construction_beam import beam_construct  # noqa: E402
from rsi_programmer.construction_check import check_graph  # noqa: E402
from rsi_programmer.construction_hard_corpus import (  # noqa: E402
    BOUND, IDENTITY, OPERATORS, _digest, _pool, _task,
    catalog_levels, load_v1_identities,
)
from rsi_programmer.construction_policy import (  # noqa: E402
    ConstructionPolicy,
)
from rsi_programmer.construction_training import (  # noqa: E402
    fit_constructor,
)
from rsi_programmer.cpu_budget import (  # noqa: E402
    CpuBudget, CpuBudgetExceeded,
)
from rsi_programmer.pre_eval import pre_evaluation_search  # noqa: E402

BACKSTOP = 2_000_000
D1_SEED = 20261108
LARGE_PER = 1600
EVAL_PER = 100
XLARGE_PER = 6400
NEED8 = XLARGE_PER + EVAL_PER


def rows_of(task: dict[str, Any]) -> Any:
    return tuple(tuple(row) for row in task["observed"])


def poly_of(task: dict[str, Any]) -> Polynomial:
    return tuple(tuple(t) for t in task["generator"]["polynomial"])


def stratum_of(task: dict[str, Any]) -> int:
    return int(task["generator"]["minimal_tree_ops"])


def accepted(task: dict[str, Any], graph: Any) -> bool:
    check = check_graph(graph, rows_of(task),
                        tuple(tuple(r) for r in task["held"]),
                        poly_of(task), BOUND)
    return bool(check["accepted"])


def timed(fn: Any, cap: float) -> tuple[Any, float, bool]:
    started = time.process_time()
    try:
        with CpuBudget(cap):
            return fn(), time.process_time() - started, False
    except CpuBudgetExceeded:
        return None, time.process_time() - started, True


def beam_row(task: dict[str, Any], policy: Any, width: int,
             cap: float) -> dict[str, Any]:
    observed = rows_of(task)

    def search() -> Any:
        return beam_construct(policy, observed, bound=BOUND,
                              budget=BACKSTOP, width=width, seed=0)
    outcome, cpu, capped = timed(search, cap)
    graph = None if outcome is None else outcome["graph"]
    return {"id": task["id"], "stratum": stratum_of(task),
            "solved": accepted(task, graph), "cpu_seconds": cpu,
            "executions": 0 if outcome is None
            else outcome["executions"], "capped": capped}


def streaming_row(task: dict[str, Any],
                  cap: float) -> dict[str, Any]:
    observed = rows_of(task)

    def search() -> Any:
        return pre_evaluation_search(
            None, tuple((x, y) for x, y, _ in observed),
            tuple(w for _, _, w in observed), BOUND,
            BACKSTOP)["graph"]
    graph, cpu, capped = timed(search, cap)
    return {"id": task["id"], "stratum": stratum_of(task),
            "solved": accepted(task, graph), "cpu_seconds": cpu,
            "capped": capped}


def corpus_polys() -> set[Polynomial]:
    corpus = json.loads(
        (ROOT / "fixtures/construction_hard_corpus_v1.json")
        .read_text())
    return {poly_of(t) for split in corpus["splits"].values()
            for t in split}


def ordered_pool(ops: int, excluded: set[Polynomial],
                 v1: set[Polynomial]
                 ) -> list[tuple[Polynomial, Node]]:
    """Complete-level pool at 6/7, digest-ordered, exclusions out."""
    levels, _ = catalog_levels()
    kept, _ = _pool(levels[ops], v1)
    fresh = [pair for pair in kept if pair[0] not in excluded]
    return sorted(fresh,
                  key=lambda it: _digest(D1_SEED, str(it[0])))


def level8_block(k: int, quota: int, rng: random.Random,
                 taken: set[Polynomial], v1: set[Polynomial],
                 excluded: set[Polynomial]
                 ) -> list[tuple[Polynomial, Node]]:
    """Rejection-sample stratum-8 witnesses with left subtree k."""
    levels, _ = catalog_levels()
    lefts, rights = levels[k], levels[BOUND - 1 - k]
    total = len(lefts) * len(rights) * len(OPERATORS)
    found: list[tuple[Polynomial, Node]] = []
    attempts = 0
    while len(found) < quota and attempts < 100_000:
        attempts += 1
        idx = rng.randrange(total)
        op = OPERATORS[idx % 3]
        rest = idx // 3
        node = apply(op, lefts[rest // len(rights)],
                     rights[rest % len(rights)])
        identity = polynomial(node)
        if (not identity or identity in IDENTITY
                or identity in taken or identity in v1
                or identity in excluded):
            continue
        taken.add(identity)
        found.append((identity, node))
    return found


def level8_pool(excluded: set[Polynomial], v1: set[Polynomial],
                need: int) -> list[tuple[Polynomial, Node]]:
    """Block-balanced stratum-8 pool, digest-ordered."""
    _, seen = catalog_levels()
    rng = random.Random(D1_SEED)
    taken = set(seen)
    quota = -(-need // BOUND)
    pairs: list[tuple[Polynomial, Node]] = []
    for k in range(BOUND):
        pairs.extend(level8_block(k, quota, rng, taken, v1,
                                  excluded))
    ordered = sorted(pairs,
                     key=lambda it: _digest(D1_SEED, str(it[0])))
    return ordered[:need]


def build_sets() -> tuple[list[dict[str, Any]],
                          list[dict[str, Any]], dict[str, Any]]:
    """19,200-task superset pool; D1 eval identities stay out."""
    excluded = corpus_polys()
    v1 = load_v1_identities(
        ROOT / "fixtures/construction_corpus_v1.json")
    train: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    d1_pool8 = level8_pool(excluded, v1, LARGE_PER + EVAL_PER)
    d1_eval8 = d1_pool8[LARGE_PER:LARGE_PER + EVAL_PER]
    eval8_polys = {p for p, _ in d1_eval8}
    pools = {6: ordered_pool(6, excluded, v1),
             7: ordered_pool(7, excluded, v1),
             8: level8_pool(excluded | eval8_polys, v1, NEED8)}
    for ops in (6, 7, 8):
        pairs = pools[ops]
        meta[str(ops)] = {"pool": len(pairs)}
        picked = (pairs[:LARGE_PER]
                  + pairs[LARGE_PER:
                          LARGE_PER + (XLARGE_PER - LARGE_PER)])
        eval_pairs = (d1_eval8 if ops == BOUND
                      else pairs[LARGE_PER:
                                 LARGE_PER + EVAL_PER])
        if ops != BOUND:
            picked = (pairs[:LARGE_PER]
                      + pairs[LARGE_PER + EVAL_PER:
                              XLARGE_PER + EVAL_PER])
        train.extend(_task(p, n, ops, "train", D1_SEED)
                     for p, n in picked)
        held.extend(_task(p, n, ops, "eval", D1_SEED)
                    for p, n in eval_pairs)
    meta["sizes"] = {"train": len(train), "eval": len(held)}
    return train, held, meta


def solve_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = Counter(r["stratum"] for r in rows if r["solved"])
    return {"solved": sum(by.values()), "n": len(rows),
            "by_stratum": {str(k): v for k, v in sorted(by.items())},
            "cpu_seconds": sum(r["cpu_seconds"] for r in rows)}


def save(record: dict[str, Any]) -> None:
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(record, indent=1) + "\n")


def fit_pool(name: str, tasks: list[dict[str, Any]],
             record: dict[str, Any]) -> ConstructionPolicy:
    print(f"fit {name} ({len(tasks)} tasks, 100 epochs)...",
          flush=True)
    policy = ConstructionPolicy(seed=0)
    fit = fit_constructor(policy, tasks, 100)
    record[f"fit_{name}"] = {k: fit[k] for k in (
        "complete_cpu_seconds", "final_loss", "final_hash",
        "teacher_steps")}
    torch.save(policy.state_dict(), f"/tmp/d2-{name}.pt")
    save(record)
    return policy


def with_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {**solve_counts(rows), "rows": rows}


def main() -> int:
    record: dict[str, Any] = {"seed": 0, "d1_seed": D1_SEED}
    print("building shape-broad pools...", flush=True)
    train, held, meta = build_sets()
    record["pools"] = meta
    save(record)
    policy = fit_pool("xlarge", train, record)
    record["eval"] = {}
    for width in (8, 32):
        print(f"eval xlarge width {width}...", flush=True)
        rows = [beam_row(t, policy, width, 30.0) for t in held]
        record["eval"][f"xlarge@w{width}"] = with_rows(rows)
        save(record)
    print("streaming @0.5...", flush=True)
    rows = [streaming_row(t, 0.5) for t in held]
    record["eval"]["streaming@0.5"] = with_rows(rows)
    save(record)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
