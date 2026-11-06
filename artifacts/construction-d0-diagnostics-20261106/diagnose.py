"""D0 diagnostics on development data (see hypothesis survey).

One fresh imitation fit (seed 0, 100 epochs, hard-corpus train
pool), then: D0-a rollout acceptance per stratum; D0-b learned
and streaming solve rates on two unseen stratum-8 sets
(prefix-unseen vs shape-broad); D0-c beam-width sweep 8/32/128;
streaming cap sweep derived from one 2 s pass. Output:
results.json beside this file. Development data only.
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rsi_programmer.benchmark_v2_corpus import (  # noqa: E402
    Node, Polynomial, apply, polynomial,
)
from rsi_programmer.construction_beam import beam_construct  # noqa: E402
from rsi_programmer.construction_check import check_graph  # noqa: E402
from rsi_programmer.construction_hard_corpus import (  # noqa: E402
    BOUND, EARLY_STOP_MULTIPLE, IDENTITY, OPERATORS, SEED,
    _grow, _pool, _select, _task, catalog_levels,
    load_v1_identities,
)
from rsi_programmer.construction_policy import (  # noqa: E402
    ConstructionPolicy,
)
from rsi_programmer.construction_rollouts import (  # noqa: E402
    collect_rollouts,
)
from rsi_programmer.construction_training import (  # noqa: E402
    fit_constructor,
)
from rsi_programmer.cpu_budget import (  # noqa: E402
    CpuBudget, CpuBudgetExceeded,
)
from rsi_programmer.pre_eval import pre_evaluation_search  # noqa: E402

BACKSTOP = 2_000_000
PER_STRATUM = 40
SET_SIZE = 60
BROAD_PER_BLOCK = 8
BROAD_SEED = 20261107
NEED8 = 566


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


def prefix_unseen(corpus_polys: set[Polynomial]
                  ) -> list[dict[str, Any]]:
    """Next unseen identities in the corpus's own prefix order."""
    levels, seen = catalog_levels()
    v1 = load_v1_identities(
        ROOT / "fixtures/construction_corpus_v1.json")
    nodes, _, _ = _grow(levels, set(seen), BOUND,
                        EARLY_STOP_MULTIPLE * NEED8)
    kept, _ = _pool(nodes, v1)
    fresh = [(p, n) for p, n in _select(kept, SEED)
             if p not in corpus_polys]
    return [_task(p, n, BOUND, "train", SEED)
            for p, n in fresh[:SET_SIZE]]


def broad_block(k: int, rng: random.Random,
                taken: set[Polynomial], v1: set[Polynomial],
                corpus_polys: set[Polynomial]
                ) -> list[tuple[Polynomial, Node]]:
    """Rejection-sample stratum-8 witnesses with left subtree k."""
    levels, _ = catalog_levels()
    lefts, rights = levels[k], levels[BOUND - 1 - k]
    total = len(lefts) * len(rights) * len(OPERATORS)
    found: list[tuple[Polynomial, Node]] = []
    attempts = 0
    while len(found) < BROAD_PER_BLOCK and attempts < 4000:
        attempts += 1
        idx = rng.randrange(total)
        op = OPERATORS[idx % 3]
        rest = idx // 3
        node = apply(op, lefts[rest // len(rights)],
                     rights[rest % len(rights)])
        identity = polynomial(node)
        if (not identity or identity in IDENTITY
                or identity in taken or identity in v1
                or identity in corpus_polys):
            continue
        taken.add(identity)
        found.append((identity, node))
    return found


def broad_tasks(corpus_polys: set[Polynomial]
                ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Shape-broad stratum-8 set, even across left-subtree sizes."""
    _, seen = catalog_levels()
    v1 = load_v1_identities(
        ROOT / "fixtures/construction_corpus_v1.json")
    rng = random.Random(BROAD_SEED)
    taken = set(seen)
    tasks: list[dict[str, Any]] = []
    blocks: dict[str, int] = {}
    for k in range(BOUND):
        found = broad_block(k, rng, taken, v1, corpus_polys)
        blocks[str(k)] = len(found)
        for identity, node in found:
            task = _task(identity, node, BOUND, "train", SEED)
            task["left_block"] = k
            tasks.append(task)
    return tasks, blocks


def acceptance_by_stratum(policy: Any,
                          train: list[dict[str, Any]]
                          ) -> dict[str, Any]:
    """D0-a: rollout acceptance and match fractions per stratum."""
    rollouts, cpu = collect_rollouts(policy, train, BOUND,
                                     BACKSTOP, width=8, seed=0)
    by_id = {t["id"]: stratum_of(t) for t in train}
    stats: dict[str, Any] = {"rollout_cpu_seconds": cpu}
    for stratum in (6, 7, 8):
        part = [r for r in rollouts
                if by_id[r.task_id] == stratum]
        fracs = [r.observed_match_fraction for r in part]
        stats[str(stratum)] = {
            "n": len(part),
            "accepted": sum(1 for r in part if r.accepted),
            "mean_match_fraction": sum(fracs) / len(fracs),
            "match_ge_0.9": sum(1 for f in fracs if f >= 0.9),
            "match_eq_1": sum(1 for f in fracs if f == 1.0),
        }
    return stats


def fractions(rows: list[dict[str, Any]],
              budgets: list[float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum in (6, 7, 8, "all"):
        part = [r for r in rows
                if stratum == "all" or r["stratum"] == stratum]
        entry = {}
        for b in budgets:
            solves = sum(1 for r in part
                         if r["solved"] and r["cpu_seconds"] <= b)
            cost = sum(min(b, r["cpu_seconds"]) for r in part)
            entry[str(b)] = {
                "fraction": solves / len(part),
                "cpu_per_solve": cost / solves if solves else None,
            }
        result[str(stratum)] = entry
    return result


def solve_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = Counter(r["stratum"] for r in rows if r["solved"])
    return {"solved": sum(by.values()), "n": len(rows),
            "by_stratum": {str(k): v for k, v in sorted(by.items())},
            "cpu_seconds": sum(r["cpu_seconds"] for r in rows)}


def save(record: dict[str, Any]) -> None:
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(record, indent=1) + "\n")


def main() -> int:
    corpus = json.loads(
        (ROOT / "fixtures/construction_hard_corpus_v1.json")
        .read_text())
    train = corpus["splits"]["train"]
    corpus_polys = {poly_of(t) for split
                    in corpus["splits"].values() for t in split}
    record: dict[str, Any] = {"seed": 0, "bound": BOUND}
    print("imitation fit (100 epochs, full train)...", flush=True)
    policy = ConstructionPolicy(seed=0)
    fit = fit_constructor(policy, train, 100)
    record["fit"] = {k: fit[k] for k in (
        "complete_cpu_seconds", "final_loss", "final_hash",
        "teacher_steps")}
    save(record)
    print("D0-a rollout acceptance...", flush=True)
    record["d0a_acceptance"] = acceptance_by_stratum(policy, train)
    save(record)
    sample: list[dict[str, Any]] = []
    for stratum in (6, 7, 8):
        sample.extend(sorted(
            (t for t in train if stratum_of(t) == stratum),
            key=lambda t: t["id"])[:PER_STRATUM])
    print("D0-c width sweep...", flush=True)
    record["d0c_widths"] = {}
    for width in (8, 32, 128):
        rows = [beam_row(t, policy, width, 30.0) for t in sample]
        record["d0c_widths"][str(width)] = solve_counts(rows)
        save(record)
    print("streaming cap sweep (2 s pass)...", flush=True)
    rows = [streaming_row(t, 2.0) for t in sample]
    record["streaming_caps"] = fractions(
        rows, [0.25, 0.5, 1.0, 2.0])
    save(record)
    print("D0-b stratum-8 sets...", flush=True)
    sets: dict[str, list[dict[str, Any]]] = {
        "prefix_unseen": prefix_unseen(corpus_polys)}
    broad, blocks = broad_tasks(corpus_polys)
    sets["shape_broad"] = broad
    record["d0b"] = {"broad_blocks_sampled": blocks}
    for name, tasks in sets.items():
        learned = [beam_row(t, policy, 8, 10.0) for t in tasks]
        stream = [streaming_row(t, 0.5) for t in tasks]
        entry: dict[str, Any] = {
            "learned": solve_counts(learned),
            "streaming@0.5": solve_counts(stream)}
        if name == "shape_broad":
            block_of = {t["id"]: t["left_block"] for t in tasks}
            by = Counter(block_of[r["id"]] for r in learned
                         if r["solved"])
            entry["learned_by_left_block"] = {
                str(k): v for k, v in sorted(by.items())}
        record["d0b"][name] = entry
        save(record)
    print(json.dumps(record, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
