#!/usr/bin/env python3
"""SNN edits: greedy author (#37), beam (#38), operands (#39)."""
from __future__ import annotations
import random
import statistics
from typing import Any
from rsi_programmer.actions import ActionSpace
from rsi_programmer.evolve import guided_search
from rsi_programmer.policy import DIM, OP_ORDER, Group, Rows, SearchPolicy, Vals
from rsi_programmer.policy import Triple, encode, layer_triples
from rsi_programmer.traces import TraceRecorder
from rsi_programmer.probe import Expr, enumerate_programs
from rsi_programmer.probe import signature, terminals
EXTRA = ("nodes_per_bound", "open_gap", "exact_flag", "self_apply")
OPDIMS = tuple(range(len(OP_ORDER) + 3, len(OP_ORDER) + 7))
TaskRow = tuple[str, Rows, Vals, int]
Scored = tuple[list[Triple], list[Vals], list[int]]
Stash = dict[int, list[tuple[Triple, Vals]]]
def open_gap(sigs: list[Vals], target: Vals) -> float:
    return 1.0 - max([sum(1 for a, b in zip(s, target) if a == b)
                       / (len(target) or 1) for s in sigs] or [0.0])
def encode_author(triple: Triple, sig: Vals, rows: Rows, target: Vals,
                  size: int, bound: int, tried: int, total: int,
                  nodes: list[Any], gap: float) -> list[float]:
    op, left, right = triple
    feats = encode(op, signature(left, rows), signature(right, rows),
                   target, left.size, right.size, size, bound, tried, total)
    exact, same = float(sig == target), float(left is right)
    return feats + [len(nodes) / (float(bound) or 1.0), gap, exact, same]
def author_scores(policy: SearchPolicy, feats: list[list[float]],
                  weights: list[float]) -> list[float]:
    base = policy.scores([f[:DIM] for f in feats])
    return [b + sum(w * e for w, e in zip(weights, f[DIM:]))
            for b, f in zip(base, feats)]
def solution_sigs(rows: Rows, target: Vals, bound: int) -> set[Vals]:
    expr, _, _ = enumerate_programs(rows, target, bound)
    assert expr is not None
    found, stack = set(), [expr]
    while stack:
        term = stack.pop()
        if isinstance(term, str) or term.op == "input": continue
        found.add(signature(term, rows))
        stack += [term.left, term.right]
    return found
def _collect_layers(rows: Rows, target: Vals,
                    bound: int) -> list[tuple[int, list[Triple]]]:
    levels: dict[int, list[Expr]] = {0: list(terminals())}
    layers: list[tuple[int, list[Triple]]] = []
    for size in range(1, bound + 1):
        triples = layer_triples(levels, size)
        layers.append((size, triples))
        exprs: dict[Any, Expr] = {}
        for op, left, right in triples:
            expr = Expr(op, left, right, size)
            exprs.setdefault((op, signature(expr, rows)), expr)
        levels[size] = list(exprs.values())
        if any(signature(e, rows) == target for e in levels[size]): break
    return layers
def _label_layer(size: int, triples: list[Triple], rows: Rows,
                 target: Vals, bound: int, offset: int,
                 budget: int, want: set[Vals]) -> Group:
    group: Group = []
    for i, (op, left, right) in enumerate(triples):
        sig = signature(Expr(op, left, right, size), rows)
        feats = encode(op, signature(left, rows), signature(right, rows),
                       target, left.size, right.size, size, bound,
                       offset + i + 1, budget)
        group.append((feats, 1 if sig in want else 0))
    return group
def _base_groups(fixture: str, budget: int = 10 ** 9) -> list[Group]:
    rec = TraceRecorder(fixture)
    frozen = {r["task"]: r for r in rec.record_all()["records"]}
    groups: list[Group] = []
    for task in rec.load_family()["tasks"]:
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        want = solution_sigs(rows, target, bound)
        layers = _collect_layers(rows, target, bound)
        total = sum(len(t) for _, t in layers)
        assert total == len(frozen[task["id"]]["choice_pairs"])
        offset = 0
        for size, triples in layers:
            groups.append(_label_layer(size, triples, rows, target,
                                       bound, offset, budget, want))
            offset += len(triples)
    return groups
def _kept(pairs: Group) -> bool:
    var = [statistics.pvariance([f[d] for f, _ in pairs]) for d in OPDIMS]
    return 0 < sum(y for _, y in pairs) < len(pairs) and max(var) > 0.0
def contrastive_groups(fixture: str, budget: int = 10 ** 9) -> list[Group]:
    out: list[Group] = []
    for group in _base_groups(fixture, budget):
        parts: dict[int, Group] = {}
        for feats, label in group:
            op = max(range(len(OP_ORDER)), key=lambda i: feats[i])
            parts.setdefault(op, []).append((feats, label))
        out += [p for p in parts.values() if _kept(p)]
    return out
def clone_groups(fixture: str, budget: int = 10 ** 9) -> list[Group]:
    groups = _base_groups(fixture, budget)
    return groups + contrastive_groups(fixture, budget)
def train_author(policy: SearchPolicy, fixture: str, epochs: int = 100) -> int:
    groups = clone_groups(fixture)
    policy.fit(groups, epochs=epochs)
    return sum(len(g) for g in groups)
def emit_edit(space: ActionSpace, graph: dict[str, Any],
              action: dict[str, Any], task_id: str, sig: Vals
              ) -> tuple[dict[str, Any], dict[str, Any], bool]:
    rec = {"task": task_id, "op": action.get("op"),
           "verdict": "signature-mismatch", "signature": list(sig)}
    try: nxt = space.apply(graph, action)
    except ValueError as exc:
        rec["verdict"] = "validator: " + str(exc)
        return graph, rec, False
    return nxt, rec, True
def _new_ctx() -> dict[str, Any]:
    levels: dict[int, list[Expr]] = {0: list(terminals())}
    graph = {"input_types": {"x": "Int", "y": "Int"},
             "nodes": [], "output": "x"}
    refs = {id(term): term.left for term in levels[0]}
    return {"space": ActionSpace(), "graph": graph, "levels": levels,
            "refs": refs, "sigs": [], "tried": 0, "log": []}
def _commit(ctx: dict[str, Any], triple: Triple, sig: Vals,
            task_id: str, size: int, target: Vals) -> bool:
    op, left, right = triple
    action = {"kind": "apply", "op": op, "left": ctx["refs"][id(left)],
              "right": ctx["refs"][id(right)]}
    ctx["graph"], rec, ok = emit_edit(ctx["space"], ctx["graph"],
                                      action, task_id, sig)
    if ok and sig == target: rec["verdict"] = ""
    ctx["log"].append(rec)
    if not ok: return False
    ctx["sigs"].append(sig)
    expr = Expr(op, left, right, size)
    ctx["refs"][id(expr)] = len(ctx["graph"]["nodes"]) - 1
    ctx["levels"][size].append(expr)
    return sig == target
def _score_order(policy: SearchPolicy, ctx: dict[str, Any],
                 rows: Rows, target: Vals, size: int, bound: int,
                 budget: int, w: list[float]) -> Scored:
    triples = layer_triples(ctx["levels"], size)
    if not triples or ctx["tried"] >= budget: return triples, [], []
    gap = open_gap(ctx["sigs"], target)
    sigs = [signature(Expr(o, l, r, size), rows) for o, l, r in triples]
    tried, nodes = ctx["tried"] + 1, ctx["graph"]["nodes"]
    feats = [encode_author(t, s, rows, target, size, bound, tried,
                           budget, nodes, gap) for t, s in zip(triples, sigs)]
    scored = author_scores(policy, feats, w)
    order = sorted(range(len(feats)), key=scored.__getitem__, reverse=True)
    return triples, sigs, order
def _author_step(policy: SearchPolicy, ctx: dict[str, Any],
                 rows: Rows, target: Vals, size: int, bound: int,
                 budget: int, task_id: str, topk: int,
                 w: list[float]) -> bool:
    triples, sigs, order = _score_order(policy, ctx, rows, target,
                                        size, bound, budget, w)
    ctx["tried"] += len(triples)
    ctx["levels"][size] = []
    for idx in order[:topk] if ctx["tried"] < budget else []:
        triple, sig = triples[idx], sigs[idx]
        if _commit(ctx, triple, sig, task_id, size, target): return True
    return False
def _pack(ctx: dict[str, Any], solved: bool, graph: dict[str, Any] | None,
    verdict: str, path: str = "author") -> dict[str, Any]:
    return {"solved": solved, "tried": ctx["tried"], "graph": graph,
            "rejected": ctx["log"], "verdict": verdict, "path": path}
def _use_fallback(policy: SearchPolicy, ctx: dict[str, Any], rows: Rows,
    target: Vals, bound: int, budget: int, task_id: str) -> dict[str, Any]:
    ranked = guided_search(policy, rows, target, bound, budget, task_id)
    ranked["path"] = "fallback"
    ranked["author_log"], ranked["author_tried"] = ctx["log"], ctx["tried"]
    return ranked
def author_search(policy: SearchPolicy, rows: Rows, target: Vals,
                  bound: int, budget: int, task_id: str = "",
                  topk: int = 3, weights: list[float] | None = None,
                  fallback: bool = True) -> dict[str, Any]:
    ctx, w = _new_ctx(), list(weights) if weights else [0.0] * len(EXTRA)
    for size in range(1, bound + 1):
        if _author_step(policy, ctx, rows, target, size, bound,
                        budget, task_id, topk, w):
            return _pack(ctx, True, ctx["graph"], "")
    if fallback:
        return _use_fallback(policy, ctx, rows, target, bound, budget,
                             task_id)
    verdict = "budget-exhausted" if ctx["tried"] >= budget else \
        "no-authored-solution"
    return _pack(ctx, False, None, verdict)
def _fitness(policy: SearchPolicy, tasks: list[TaskRow], weights: list[float],
    topk: int, budget: int) -> tuple[int, int]:
    solved, tried = 0, 0
    for tid, rows, target, bound in tasks:
        out = author_search(policy, rows, target, bound, budget, tid,
                            topk, weights, fallback=False)
        solved, tried = solved + out["solved"], tried + out["tried"]
    return (solved, -tried)
Refined = tuple[list[float], dict[str, int]]
def refine_author(policy: SearchPolicy, tasks: list[TaskRow],
                  weights: list[float] | None = None, rounds: int = 12,
                  sigma: float = 0.1, seed: int = 0, topk: int = 3,
                  budget: int = 10 ** 9) -> Refined:
    best = list(weights) if weights else [0.0] * len(EXTRA)
    best_fit = _fitness(policy, tasks, best, topk, budget)
    for step in range(rounds):
        rng = random.Random(seed + step)
        cand = [w + rng.gauss(0.0, sigma) for w in best]
        fit = _fitness(policy, tasks, cand, topk, budget)
        if fit > best_fit: best, best_fit = cand, fit
    return best, {"solved": best_fit[0], "tried": -best_fit[1]}
def _beam_scan(policy: SearchPolicy, ctx: dict[str, Any], rows: Rows,
    target: Vals, size: int, bound: int, budget: int, task_id: str,
    width: int, w: list[float], stash: Stash) -> bool:
    triples, sigs, order = _score_order(policy, ctx, rows, target,
                                        size, bound, budget, w)
    ctx["levels"][size] = []
    for idx in order:
        if ctx["tried"] >= budget: return False
        if len(ctx["levels"][size]) >= width:
            stash[size].append((triples[idx], sigs[idx]))
            continue
        ctx["tried"] += 1
        triple, sig = triples[idx], sigs[idx]
        if _commit(ctx, triple, sig, task_id, size, target): return True
    return False
def _backtrack(ctx: dict[str, Any], task_id: str, bound: int,
               budget: int, stash: Stash, target: Vals) -> bool:
    for size in range(1, bound + 1):
        while stash[size] and ctx["tried"] < budget:
            triple, sig = stash[size].pop(0)
            ctx["tried"] += 1
            if _commit(ctx, triple, sig, task_id, size, target): return True
    return False
def beam_search(policy: SearchPolicy, rows: Rows, target: Vals,
                bound: int, budget: int, task_id: str = "",
                width: int = 3, weights: list[float] | None = None,
                fallback: bool = True) -> dict[str, Any]:
    ctx, w = _new_ctx(), list(weights) if weights else [0.0] * len(EXTRA)
    stash: Stash = {s: [] for s in range(1, bound + 1)}
    for size in range(1, bound + 1):
        if _beam_scan(policy, ctx, rows, target, size, bound,
                      budget, task_id, width, w, stash):
            return _pack(ctx, True, ctx["graph"], "", "beam")
    if _backtrack(ctx, task_id, bound, budget, stash, target):
        return _pack(ctx, True, ctx["graph"], "", "beam")
    if fallback:
        return _use_fallback(policy, ctx, rows, target, bound, budget,
                             task_id)
    verdict = "budget-exhausted" if ctx["tried"] >= budget else \
        "no-authored-solution"
    return _pack(ctx, False, None, verdict, "beam")
