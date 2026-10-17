#!/usr/bin/env python3
"""Directing-vs-echoing instrumentation for the lattice gate.

No training, loss, threshold, or author changes: this module
only measures. Named-rate decomposes to commit/stash/reject;
a seeded word-shuffle ablation (lattice struct kept, words
random) separates naming signal from sweep structure; beam
rank overlap tests whether commits reproduce beam order;
per-op rates show minority-op survival.
"""
from __future__ import annotations

from typing import Any

from rsi_programmer.author import _new_ctx, _score_order
from rsi_programmer.lattice import BitLattice
from rsi_programmer.lauthor import lattice_search
from rsi_programmer.policy import SearchPolicy, layer_triples
from rsi_programmer.probe import Expr, signature, terminals
from rsi_programmer.traces import TraceRecorder

SHUFFLE_SEEDS = (0, 1, 2)
ECHO_SIZES = (1, 2)


def summarize(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Fate counts and per-op named rates from one trace."""
    fates: dict[str, int] = {}
    per_op: dict[str, list[int]] = {}
    for entry in trace:
        fates[entry["fate"]] = fates.get(entry["fate"], 0) + 1
        pair = per_op.setdefault(entry["op"], [0, 0])
        pair[1] += 1
        pair[0] += 1 if entry["named"] else 0
    rates = {op: named / total for op, (named, total)
             in per_op.items()}
    examined = len(trace)
    named = sum(1 for e in trace if e["named"])
    return {"examined": examined, "named": named,
            "named_rate": named / max(examined, 1),
            "fates": fates, "per_op": rates}


def _full_levels(size: int) -> dict[int, list[Any]]:
    """Probe-order full expression levels up to a size."""
    levels: dict[int, list[Any]] = {0: list(terminals())}
    for step in range(1, size):
        levels[step] = [Expr(op, left, right, step)
                        for op, left, right in
                        layer_triples(levels, step)]
    return levels


def echo_overlap(policy: SearchPolicy, weights: list[float],
                 rows: Any, target: Any, bound: int,
                 budget: int, named: set[Any],
                 width: int) -> float:
    """Fraction of lattice-named sigs beam would commit."""
    hits, total = 0, 0
    for size in ECHO_SIZES:
        if size > bound:
            break
        ctx = _new_ctx()
        ctx["levels"] = _full_levels(size)
        _, sigs, order, _ = _score_order(
            policy, ctx, rows, target, size, bound,
            budget, weights)
        top = {sigs[i] for i in order[:width]}
        layer = {s for s in named if s in set(sigs)}
        total += len(layer)
        hits += len(layer & top)
    return hits / max(total, 1)


def family_report(lat: BitLattice, policy: SearchPolicy,
                  weights: list[float],
                  tasks: list[dict[str, Any]],
                  recorder: TraceRecorder, budget: int,
                  width: int) -> dict[str, Any]:
    """Directing-vs-echo metrics for one task family."""
    examined = named = solved = shuffled = 0
    overlap_top, overlap_n = 0.0, 0
    per_op: dict[str, list[int]] = {}
    for task in tasks:
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        trace: list[dict[str, Any]] = []
        out = lattice_search(lat, rows, target, bound,
                             budget, str(task["id"]), width,
                             6, trace, None)
        holds = out["solved"] and recorder.oracle_holds(
            task, out["graph"])
        solved += 1 if holds else 0
        part = summarize(trace)
        examined += part["examined"]
        named += part["named"]
        for op, rate in part["per_op"].items():
            pair = per_op.setdefault(op, [0.0, 0])
            pair[0] += rate
            pair[1] += 1
        named_sigs = {e["sig"] for e in trace if e["named"]}
        overlap_top += echo_overlap(policy, weights, rows,
                                    target, bound, budget,
                                    named_sigs, width)
        overlap_n += 1
        for seed in SHUFFLE_SEEDS:
            alt = lattice_search(lat, rows, target, bound,
                                 budget, str(task["id"]),
                                 width, 6, None, seed)
            if alt["solved"] and recorder.oracle_holds(
                    task, alt["graph"]):
                shuffled += 1
                break
    return {"tasks": len(tasks), "solved": solved,
            "named_rate": named / max(examined, 1),
            "examined_tried": examined,
            "echo_overlap": overlap_top / max(overlap_n, 1),
            "shuffle_solves": shuffled,
            "shuffle_tasks": len(tasks),
            "per_op": {op: total / count
                       for op, (total, count)
                       in per_op.items()}}


def selectivity_report(
        lat: BitLattice, policy: SearchPolicy,
        weights: list[float],
        families: dict[str, tuple[list[dict[str, Any]],
                                  TraceRecorder]],
        budget: int, width: int) -> dict[str, dict[str, Any]]:
    """Per-family directing-vs-echo verdicts with prints."""
    report = {}
    for name, (tasks, recorder) in families.items():
        part = family_report(lat, policy, weights, tasks,
                             recorder, budget, width)
        report[name] = part
        print("selectivity %s: solved=%d/%d named=%.3f "
              "echo=%.3f shuffle=%d/%d per-op=%s" %
              (name, part["solved"], part["tasks"],
               part["named_rate"], part["echo_overlap"],
               part["shuffle_solves"], part["shuffle_tasks"],
               {op: round(r, 3)
                for op, r in part["per_op"].items()}))
    return report
