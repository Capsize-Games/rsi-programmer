#!/usr/bin/env python3
"""Word-gated beam author for the bit-slice lattice (v33).

Proposals come from probe-order triple layers (the lattice
drive is triple-shaped by training); authorship is word
driven: a triple commits only when the lattice emits its
opcode word with EMIT-majority. No policy scores rank
anything; fallback=False only, so misses stay misses.
"""
from __future__ import annotations

from typing import Any

from rsi_programmer.author import MAX_PASSES, _backtrack
from rsi_programmer.author import _commit, _fresh, _new_ctx
from rsi_programmer.author import _pack, open_gap
from rsi_programmer.lattice import BitLattice
from rsi_programmer.policy import encode, layer_triples
from rsi_programmer.probe import Expr, signature
from rsi_programmer.traces import TraceRecorder
from rsi_programmer.words import OPCODES, pack


def _words_and_emit(lat: BitLattice, feats: list[list[float]],
                    gap: float | None) -> tuple[list[int],
                                                list[bool]]:
    """One lattice drive; thresholded words plus struct flags."""
    rates = lat.head_rates(feats, gap)
    words = []
    for row in range(len(feats)):
        bits = [1 if rates["b%d" % i][row][0] >= 0.5 else 0
                for i in range(4)]
        words.append(pack(bits))
    struct = [row[0] >= row[1] for row in rates["st"]]
    return words, struct


def _lattice_scan(lat: BitLattice, ctx: dict[str, Any],
                  rows: Any, target: Any, size: int, bound: int,
                  budget: int, task_id: str, width: int,
                  stash: dict[int, list[Any]]) -> bool:
    """Commit triples the lattice names; rejects drop out.

    Disagreed triples are rejected, never stashed: every
    commit is word-named by construction, so misses are
    capability verdicts instead of silent enumerations.
    Only word-named overflow waits in the stash.
    """
    triples = layer_triples(ctx["levels"], size)
    if size not in ctx["levels"]:
        ctx["levels"][size] = []
    base = len(ctx["levels"][size])
    if not triples:
        return False
    lat.reset()
    sigs = [signature(Expr(op, left, right, size), rows)
            for op, left, right in triples]
    gap = open_gap(ctx["sigs"], target)
    feats = [encode(op, signature(left, rows),
                    signature(right, rows), target,
                    left.size, right.size, size, bound,
                    ctx["tried"] + i + 1, budget)
             for i, (op, left, right) in enumerate(triples)]
    words, emits = _words_and_emit(lat, feats, gap)
    for idx, (op, _left, _right) in enumerate(triples):
        if ctx["tried"] >= budget:
            return False
        if not _fresh(sigs[idx], ctx):
            continue
        ctx["examined"] = ctx.get("examined", 0) + 1
        if not (words[idx] == OPCODES[op] and emits[idx]):
            continue
        ctx["named"] = ctx.get("named", 0) + 1
        if len(ctx["levels"][size]) - base >= width:
            stash[size].append((0.0, triples[idx], sigs[idx]))
            continue
        ctx["tried"] += 1
        triple, sig = triples[idx], sigs[idx]
        if _commit(ctx, triple, sig, task_id, size,
                   target):
            return True
    return False


def _lattice_sweep(lat: BitLattice, rows: Any, target: Any,
                   bound: int, budget: int, task_id: str,
                   width: int) -> tuple[dict[str, Any], bool]:
    """Word-gated scans plus fifo backtrack, to fixpoint."""
    ctx = _new_ctx()
    for _ in range(MAX_PASSES):
        stash: dict[int, list[Any]] = {s: []
                                      for s in range(1, bound + 1)}
        for size in range(1, bound + 1):
            if ctx["tried"] >= budget:
                break
            if _lattice_scan(lat, ctx, rows, target, size,
                             bound, budget, task_id, width,
                             stash):
                out = _pack(ctx, True, ctx["graph"], "",
                            "lattice")
                out["agreed"] = True
                out["examined"] = ctx.get("examined", 0)
                out["named"] = ctx.get("named", 0)
                return out, True
        if ctx["tried"] >= budget:
            break
        known = len(ctx["sigs"])
        if _backtrack(ctx, task_id, bound, budget, stash,
                      target, "fifo"):
            out = _pack(ctx, True, ctx["graph"], "",
                        "lattice")
            out["agreed"] = True
            out["examined"] = ctx.get("examined", 0)
            out["named"] = ctx.get("named", 0)
            return out, True
        if len(ctx["sigs"]) == known:
            break
    verdict = "budget-exhausted" if ctx["tried"] >= budget \
        else "no-authored-solution"
    out = _pack(ctx, False, None, verdict, "lattice")
    out["agreed"] = False
    out["examined"] = ctx.get("examined", 0)
    out["named"] = ctx.get("named", 0)
    return out, False


def lattice_search(lat: BitLattice, rows: Any, target: Any,
                   bound: int, budget: int, task_id: str = "",
                   width: int = 3,
                   widen: int = 0) -> dict[str, Any]:
    """Word-gated sweep, optional wider retry, never fallback.

    Drives are stateless: temporal resets every scan, so the
    same levels always emit the same words (fixpoint stays
    well-defined). Cross-drive conversation is explicitly
    out; it was never trained.
    """
    out, _ = _lattice_sweep(lat, rows, target, bound, budget,
                            task_id, width)
    if not out["solved"] and widen > 0 and \
            out["tried"] < budget:
        again, _ = _lattice_sweep(lat, rows, target, bound,
                                  budget - out["tried"],
                                  task_id, widen)
        again["tried"] = out["tried"] + again["tried"]
        again["examined"] = out["examined"] + again["examined"]
        again["named"] = out["named"] + again["named"]
        out = again
    out["named_rate"] = out["named"] / max(out["examined"], 1)
    return out


def lattice_runs(lat: BitLattice, tasks: list[dict[str, Any]],
                 budget: int, width: int,
                 recorder: TraceRecorder,
                 widen: int = 6) -> list[dict[str, Any]]:
    """Author-only lattice sweep with oracle honesty check."""
    per: list[dict[str, Any]] = []
    for task in tasks:
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        out = lattice_search(lat, rows, target, bound,
                             budget, str(task["id"]), width,
                             widen)
        holds = out["solved"] and recorder.oracle_holds(
            task, out["graph"])
        per.append({"task": str(task["id"]),
                    "solved": bool(out["solved"] and holds),
                    "tried": int(out["tried"]),
                    "agreed": bool(out["agreed"]),
                    "named_rate": float(out["named_rate"]),
                    "oracle": bool(holds),
                    "path": "lattice",
                    "verdict": str(out.get("verdict", ""))})
    return per
