#!/usr/bin/env python3
"""Layer teaching groups and per-head metrics for ModularPolicy.

Each layer of each task becomes one group: thirteen-dim drives with
the open-gap channel, the row's own op index, and positive/negative
row sets for the choice, op, operand, and emit terms. Reports score
each head against chance with the group counts that produced them.
"""
from __future__ import annotations

from typing import Any

from rsi_programmer.author import open_gap, solution_sigs
from rsi_programmer.modules import N_OP
from rsi_programmer.policy import OP_ORDER, Rows, Vals
from rsi_programmer.policy import encode, layer_triples
from rsi_programmer.probe import Expr, signature, terminals
from rsi_programmer.traces import TraceRecorder


def _layers(rows: Rows, target: Vals,
            bound: int) -> list[tuple[int, list]]:
    """Deduplicated Int layers mirroring the enumerator order."""
    levels: dict[int, list[Expr]] = {0: list(terminals())}
    out: list[tuple[int, list]] = []
    for size in range(1, bound + 1):
        triples = layer_triples(levels, size)
        exprs: dict[Any, Expr] = {}
        for op, left, right in triples:
            expr = Expr(op, left, right, size)
            exprs.setdefault((op, signature(expr, rows)), expr)
        levels[size] = list(exprs.values())
        out.append((size, triples))
        if any(signature(e, rows) == target
               for e in levels[size]):
            break
    return out


def modular_groups(fixture: str,
                   budget: int = 10 ** 9) -> list[dict[str, Any]]:
    """One teaching group per task layer with head labels."""
    recorder = TraceRecorder(fixture)
    groups: list[dict[str, Any]] = []
    for task in recorder.load_family()["tasks"]:
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        want = solution_sigs(rows, target, bound)
        offset = 0
        for size, triples in _layers(rows, target, bound):
            sigs = [signature(Expr(o, l, r, size), rows)
                    for o, l, r in triples]
            lefts = [signature(l, rows) for _, l, _ in triples]
            gap = open_gap(sigs, target)
            total = sum(len(t) for _, t in _layers(rows, target,
                                                   bound))
            feats = [encode(o, signature(l, rows),
                            signature(r, rows), target,
                            l.size, r.size, size, bound,
                            offset + i + 1, total)
                     for i, (o, l, r) in enumerate(triples)]
            pos = [i for i, sig in enumerate(sigs)
                   if sig in want]
            neg = [i for i in range(len(triples))
                   if i not in pos]
            ops = [OP_ORDER.index(o) for o, _, _ in triples]
            solops = {OP_ORDER.index(o) for i, (o, _, _)
                      in enumerate(triples) if i in pos}
            groups.append({"feats": feats, "gap": gap,
                           "pos": pos, "neg": neg, "ops": ops,
                           "solops": sorted(solops),
                           "sigs": [list(s) for s in sigs],
                           "lefts": [list(s) for s in lefts],
                           "want": sorted(map(list, want))})
            offset += len(triples)
    return groups


def module_report(policy: Any,
                  groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-head accuracy against chance with group counts."""
    op_hits = op_chance = 0.0
    lo_hits = lo_chance = 0.0
    counted = 0
    for group in groups:
        rates = policy.head_rates(group["feats"], group.get("gap"))
        n = len(group["feats"])
        if not n or not group["solops"]:
            continue
        counted += 1
        preds = [row.index(max(row)) for row in rates["op"]]
        op_hits += sum(1 for i, p in enumerate(preds)
                       if p == group["ops"][i]) / n
        op_chance += 1.0 / N_OP
        want = {tuple(s) for s in group["want"]}
        lo = rates["lo"]
        top = max(range(n), key=lambda i: lo[i][0])
        lo_hits += float(tuple(group["lefts"][top]) in want)
        lo_chance += 1.0 / n
    if not counted:
        return {"layers": 0}
    return {"layers": counted,
            "op_accuracy": round(op_hits / counted, 4),
            "op_chance": round(op_chance / counted, 4),
            "operand_top1": round(lo_hits / counted, 4),
            "operand_chance": round(lo_chance / counted, 4)}
