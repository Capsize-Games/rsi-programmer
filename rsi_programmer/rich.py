#!/usr/bin/env python3
"""Richer search-state features and surrogate builders (#35).

Pure helpers for the scale-up loop: four extra features beyond
the W2 twelve, surrogate training pairs under fidelity caps,
fitness ordering, and family rollups. No classes live here.
"""

from __future__ import annotations

from typing import Any

from rsi_programmer.evolve import SIGMA, guided_search
from rsi_programmer.policy import Group, Rows, SearchPolicy, Task, Vals
from rsi_programmer.policy import encode, layer_triples
from rsi_programmer.probe import Expr, signature, terminals

RICH = ("depth-frac", "imbalance", "joint-hit", "joint-inside")
RICH_LR = 1.0
NEG_CAP = 1500
SEL_CAP = 750
COMPS = ("correctness", "efficiency", "hard_benefit", "recovery")


def rich_extra(left_sig: Vals, right_sig: Vals, target: Vals,
               lsize: int, rsize: int, size: int,
               bound: int) -> list[float]:
    """Four richer search-state features beyond the W2 twelve."""
    n = len(target) or 1
    low, high = min(target), max(target)
    hits = sum(a == b for s in (left_sig, right_sig)
               for a, b in zip(s, target))
    inside = sum(1 for s in (left_sig, right_sig) for a in s
                 if low <= a <= high)
    norm = float(bound) or 1.0
    return [size / norm, abs(lsize - rsize) / norm,
            hits / (2 * n), inside / (2 * n)]


def bonus_for(rich: list[float],
              feats: list[list[float]]) -> list[float]:
    """Rich linear bonus for one batch of base features."""
    out = []
    for feat in feats:
        extra = (feat[5], abs(feat[3] - feat[4]),
                 (feat[6] + feat[8]) / 2.0,
                 (feat[7] + feat[9]) / 2.0)
        out.append(sum(w * x for w, x in zip(rich, extra)))
    return out


def neg_pair(op: str, left: Expr, right: Expr, size: int,
             rows: Rows, target: Vals, bound: int,
             k: int) -> tuple[Any, Any]:
    """One negative pair in both feature spaces."""
    left_sig = signature_of(left, rows)
    right_sig = signature_of(right, rows)
    feat = encode(op, left_sig, right_sig, target, left.size,
                  right.size, size, bound, k, NEG_CAP)
    extra = rich_extra(left_sig, right_sig, target, left.size,
                       right.size, size, bound)
    return (feat, 0), (extra, 0)


def signature_of(expr: Expr, rows: Rows) -> Vals:
    """Signature helper kept beside its only caller."""
    return signature(expr, rows)


def fit_rich(groups: list[Group]) -> list[float]:
    """Mean-difference weights for the rich features."""
    weights = [0.0, 0.0, 0.0, 0.0]
    count = 0
    for group in groups:
        pos = [f for f, y in group if y == 1]
        neg = [f for f, y in group if y == 0]
        if not pos or not neg:
            continue
        for i in range(4):
            mean_pos = sum(f[i] for f in pos) / len(pos)
            mean_neg = sum(f[i] for f in neg) / len(neg)
            weights[i] += RICH_LR * (mean_pos - mean_neg)
        count += 1
    return [v / max(1, count) for v in weights]


def dom4(a: dict[str, float], b: dict[str, float]) -> bool:
    """Pareto dominance over the four scale-up components."""
    return all(a[k] >= b[k] for k in COMPS) and any(
        a[k] > b[k] for k in COMPS)


def front4(pop: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Nondominated front of the population."""
    return [e for e in pop if not any(
        dom4(o["fitness"], e["fitness"]) for o in pop if o is not e)]


def fitkey4(entry: dict[str, Any]) -> tuple[float, ...]:
    """Order key over the four fitness components."""
    fit = entry["fitness"]
    return (fit["correctness"], fit["efficiency"],
            fit["hard_benefit"], fit["recovery"])


def fitness_of(policy: SearchPolicy, jobs: list[Any],
               sel: list[Any], rich: list[float], w1cost: int,
               seed: int) -> dict[str, float]:
    """Correctness/efficiency/benefit/recovery fitness."""
    w1 = [guided_search(policy, r, g, b, m, i)
          for r, g, b, m, i in jobs]
    noisy = SearchPolicy(seed=seed)
    noisy.restore(policy.snapshot())
    noisy.perturb(SIGMA, seed)
    fixed = list(rich)
    noisy._bonus = lambda feats: bonus_for(fixed, feats)
    rec = [guided_search(noisy, r, g, b, m, i)
           for r, g, b, m, i in jobs]
    solved = sum(r["solved"] for r in w1)
    cost = sum(r["tried"] for r in w1)
    benefit = _pairs_acc(policy, sel, rich)
    recovered = sum(r["solved"] for r in rec) / len(rec)
    return {"correctness": solved / len(w1), "recovery": recovered,
            "efficiency": max(0.0, 1.0 - cost / w1cost),
            "hard_benefit": benefit}


def _pairs_acc(policy: SearchPolicy, sel: list[Any],
               rich: list[float]) -> float:
    """Top-1 accuracy of the solver over selection negs."""
    hits = 0
    for feats in sel:
        vals = policy.scores(feats)
        scored = [v + b for v, b in zip(
            vals, bonus_for(rich, feats))]
        best = max(range(len(feats)), key=scored.__getitem__)
        hits += best == len(feats) - 1
    return hits / len(sel)


def rollup(per: list[dict[str, Any]],
           fam: str) -> dict[str, Any]:
    """Solve-rate and cost rollup for one accepted family."""
    rows = [p for p in per if p["family"] == fam]
    solved = sum(1 for p in rows if p["solved"])
    cost = sum(int(p["tried"]) for p in rows)
    base = sum(int(p["base_tried"]) for p in rows)
    return {"solve_rate": solved / len(rows), "cost": cost,
            "baseline_cost": base, "solved": solved}


def _neg_layer(levels: dict[int, list[Expr]], size: int,
               rows: Rows, target: Vals, bound: int,
               g12: Group, gx: Group) -> list[Expr]:
    """One layer of capped negatives; returns its members."""
    cur: list[Expr] = []
    for op, left, right in layer_triples(levels, size):
        cur.append(Expr(op, left, right, size))
        first, second = neg_pair(op, left, right, size, rows,
                                 target, bound, len(g12))
        g12.append(first)
        gx.append(second)
        if len(g12) >= NEG_CAP:
            break
    return cur


def hard_neg(rows: Rows, target: Vals,
             bound: int) -> tuple[Group, Group]:
    """Bounded negatives from the task's own early layers."""
    g12: Group = []
    gx: Group = []
    levels: dict[int, list[Expr]] = {0: list(terminals())}
    for size in (1, 2, 3, 4):
        levels[size] = _neg_layer(levels, size, rows, target,
                                  bound, g12, gx)
        if len(g12) >= NEG_CAP:
            break
    return g12, gx


def hard_pos(expr: Expr, rows: Rows, target: Vals,
               bound: int, tried: int) -> tuple[Any, Any]:
    """Positive pairs from the solution top triple."""
    left, right = expr.left, expr.right
    lside, rside = signature_of(left, rows), signature_of(
        right, rows)
    feat = encode(expr.op, lside, rside, target, left.size,
                  right.size, expr.size, bound, tried, tried)
    extra = rich_extra(lside, rside, target, left.size,
                       right.size, expr.size, bound)
    return (feat, 1), (extra, 1)


def family_pairs(tasks: dict[str, Task], exprs: dict[str, Any],
                 tries: dict[str, int], parts: Any) -> tuple[
                     list[Group], list[Group], list[Any]]:
    """Surrogate groups plus one disjoint selection slice."""
    g12: list[Group] = []
    gx: list[Group] = []
    sel: list[Any] = []
    for tid, expr in exprs.items():
        if expr is None:
            continue
        rows, target, bound = parts(tasks[tid])
        pos = hard_pos(expr, rows, target, bound, tries[tid])
        neg12, negx = hard_neg(rows, target, bound)
        g12.append([pos[0]] + neg12[SEL_CAP:])
        gx.append([pos[1]] + negx[SEL_CAP:])
        sel.append([f for f, _ in neg12[:SEL_CAP]] + [pos[0][0]])
    return g12, gx, sel
