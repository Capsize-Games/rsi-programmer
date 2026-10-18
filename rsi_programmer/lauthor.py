#!/usr/bin/env python3
"""Word-gated beam author for the bit-slice lattice (v33).

Proposals come from probe-order triple layers (the lattice
drive is triple-shaped by training); authorship is word
driven: a triple commits only when the lattice emits its
opcode word with EMIT-majority. No policy scores rank
anything; fallback=False only, so misses stay misses.
"""
from __future__ import annotations

import random
from typing import Any

from rsi_programmer.author import MAX_PASSES, _backtrack
from rsi_programmer.author import _add_work
from rsi_programmer.author import _commit, _fresh, _new_ctx
from rsi_programmer.author import _pack, open_gap
from rsi_programmer.lattice import BitLattice
from rsi_programmer.policy import encode, layer_triples
from rsi_programmer.probe import Expr, signature
from rsi_programmer.traces import TraceRecorder
from rsi_programmer.words import OPCODES, pack


def _words_and_emit(lat: BitLattice, feats: list[list[float]],
                    gap: float | None,
                    rng: Any = None) -> tuple[list[int],
                                              list[bool]]:
    """One lattice drive; thresholded words plus struct flags.

    With rng set, words come from the seeded shuffle (struct
    stays lattice): the word-shuffle ablation for measuring
    how much solves depend on naming versus structure.
    """
    rates = lat.head_rates(feats, gap)
    if rng is None:
        words = []
        for row in range(len(feats)):
            bits = [1 if rates["b%d" % i][row][0] >=
                    lat.thresholds["b%d" % i]
                    else 0 for i in range(4)]
            words.append(pack(bits))
    else:
        words = [rng.randrange(16) for _ in feats]
    margin = lat.thresholds["st"]
    struct = [row[0] >= row[1] + margin for row in rates["st"]]
    return words, struct


def _lattice_scan(lat: BitLattice, ctx: dict[str, Any],
                  rows: Any, target: Any, size: int, bound: int,
                  budget: int, task_id: str, width: int,
                  stash: dict[int, list[Any]],
                  trace: list[dict[str, Any]] | None = None,
                  rng: random.Random | None = None) -> bool:
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
    ctx["signature_evaluations"] += len(triples)
    ctx["operand_signature_evaluations"] += 2 * len(triples)
    # Lattice packs are trained with this constant context channel.
    gap = 1.0
    feats = [encode(op, signature(left, rows),
                    signature(right, rows), target,
                    left.size, right.size, size, bound,
                    i + 1, len(triples))
             for i, (op, left, right) in enumerate(triples)]
    words, emits = _words_and_emit(lat, feats, gap, rng)
    for idx, (op, _left, _right) in enumerate(triples):
        if ctx["tried"] >= budget:
            return False
        if not _fresh(sigs[idx], ctx):
            continue
        ctx["examined"] = ctx.get("examined", 0) + 1
        if not (words[idx] == OPCODES[op] and emits[idx]):
            if trace is not None:
                trace.append({"size": size, "op": op,
                              "sig": sigs[idx],
                              "feats": feats[idx],
                              "named": False,
                              "fate": "dropped"})
            continue
        ctx["named"] = ctx.get("named", 0) + 1
        if len(ctx["levels"][size]) - base >= width:
            stash[size].append((0.0, triples[idx], sigs[idx]))
            if trace is not None:
                trace.append({"size": size, "op": op,
                              "sig": sigs[idx],
                              "feats": feats[idx],
                              "named": True,
                              "fate": "stashed"})
            continue
        ctx["tried"] += 1
        triple, sig = triples[idx], sigs[idx]
        if trace is not None:
            trace.append({"size": size, "op": op,
                          "sig": sigs[idx],
                          "feats": feats[idx],
                          "named": True, "fate": "committed"})
        if _commit(ctx, triple, sig, task_id, size,
                   target):
            return True
    return False


def _lattice_sweep(lat: BitLattice, rows: Any, target: Any,
                   bound: int, budget: int, task_id: str,
                   width: int,
                   trace: list[dict[str, Any]] | None = None,
                   shuffle: int | None = None
                   ) -> tuple[dict[str, Any], bool]:
    """Word-gated scans plus fifo backtrack, to fixpoint."""
    ctx = _new_ctx()
    rng = random.Random(shuffle) if shuffle is not None else None
    for _ in range(MAX_PASSES):
        stash: dict[int, list[Any]] = {s: []
                                      for s in range(1, bound + 1)}
        for size in range(1, bound + 1):
            if ctx["tried"] >= budget:
                break
            if _lattice_scan(lat, ctx, rows, target, size,
                             bound, budget, task_id, width,
                             stash, trace, rng):
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
                   width: int = 3, widen: int = 0,
                   trace: list[dict[str, Any]] | None = None,
                   shuffle: int | None = None) -> dict[str, Any]:
    """Word-gated sweep, optional wider retry, never fallback.

    Drives are stateless: temporal resets every scan, so the
    same levels always emit the same words (fixpoint stays
    well-defined). Cross-drive conversation is explicitly
    out; it was never trained.
    """
    out, _ = _lattice_sweep(lat, rows, target, bound, budget,
                            task_id, width, trace, shuffle)
    if not out["solved"] and widen > 0 and \
            out["tried"] < budget:
        again, _ = _lattice_sweep(lat, rows, target, bound,
                                  budget - out["tried"],
                                  task_id, widen, trace,
                                  shuffle)
        again["tried"] = out["tried"] + again["tried"]
        _add_work(again, out)
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
                    "signature_evaluations":
                    int(out["signature_evaluations"]),
                    "cost_unit": "candidate_signature_evaluations",
                    "agreed": bool(out["agreed"]),
                    "named_rate": float(out["named_rate"]),
                    "oracle": bool(holds),
                    "path": "lattice",
                    "verdict": str(out.get("verdict", ""))})
    return per
