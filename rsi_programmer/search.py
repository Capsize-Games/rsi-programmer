#!/usr/bin/env python3
"""Policy-ranked search over typed DAGs (RSI search layer)."""

from __future__ import annotations

from typing import Any, Callable

from rsi_programmer.actions import ActionSpace
from rsi_programmer.policy import Rows, SearchPolicy, Triple, Vals
from rsi_programmer.policy import encode, layer_triples
from rsi_programmer.probe import Expr, const_node, const_terms
from rsi_programmer.probe import signature, terminals
OracleHook = Callable[[dict[str, Any]], bool]
def _new_ctx(consts: tuple[int, ...] = ()) -> dict[str, Any]:
    """Search context, plus const nodes when values given.

    Constant terminals join level zero with pre-created
    graph nodes, so triples over them intern exactly like
    committed expressions; empty consts keeps the bare
    input-only context byte-identical to before.
    """
    terms = list(terminals()) + list(const_terms(consts))
    graph: dict[str, Any] = {"input_types": {"x": "Int",
                                             "y": "Int"},
                             "nodes": [], "output": "x"}
    refs: dict[int, Any] = {}
    for term in terms:
        if term.op == "const":
            refs[id(term)] = len(graph["nodes"])
            graph["nodes"].append(const_node(term.left))
        else:
            refs[id(term)] = term.left
    levels: dict[int, list[Expr]] = {0: terms}
    return {"space": ActionSpace(), "graph": graph, "levels": levels,
            "refs": refs, "interned": {}, "tried": 0, "rejected": []}
def _intern(ctx: dict[str, Any], rows: Rows,
            triples: list[Triple], size: int) -> list[Vals]:
    graph, refs = ctx["graph"], ctx["refs"]
    sigs = []
    for op, left, right in triples:
        expr = Expr(op, left, right, size)
        sigs.append(signature(expr, rows))
        key = (op, refs[id(left)], refs[id(right)])
        if key not in ctx["interned"]:
            node = {"args": [key[1], key[2]], "op": op}
            node["type"] = "Int"
            graph["nodes"].append(node)
            ctx["interned"][key] = len(graph["nodes"]) - 1
    return sigs
def _members(ctx: dict[str, Any], triples: list[Triple],
             sigs: list[Vals], size: int) -> None:
    seen: set[Any] = set()
    ctx["levels"][size] = []
    for (op, left, right), sig in zip(triples, sigs):
        if (op, sig) not in seen:
            seen.add((op, sig))
            member = Expr(op, left, right, size)
            key = (op, ctx["refs"][id(left)], ctx["refs"][id(right)])
            ctx["refs"][id(member)] = ctx["interned"][key]
            ctx["levels"][size].append(member)
def _oracle_kept(ctx: dict[str, Any], found: Any, sig: Vals,
                 task_id: str, oracle: OracleHook | None) -> bool:
    """Keep a train match the post-hoc oracle accepts."""
    if oracle is None:
        return True
    got = _accept(ctx, found)
    if got["graph"] is not None and oracle(got["graph"]):
        return True
    ctx["rejected"].append({"task": task_id, "op": found["op"],
                            "verdict": got["verdict"] or
                            "oracle-rejected",
                            "signature": list(sig)})
    return False


def _scan(ctx: dict[str, Any], triples: list[Triple],
          sigs: list[Vals], order: list[int], target: Vals,
          task_id: str, oracle: OracleHook | None = None) -> Any:
    for idx in order:
        op, left, right = triples[idx]
        ctx["tried"] += 1
        if sigs[idx] != target:
            ctx["rejected"].append({"task": task_id, "op": op,
                                    "verdict":
                                    "signature-mismatch"})
            continue
        found = {"kind": "apply", "op": op,
                 "left": ctx["refs"][id(left)],
                 "right": ctx["refs"][id(right)]}
        if _oracle_kept(ctx, found, sigs[idx], task_id, oracle):
            return found
    return None
def _layer_gap(sigs: list[Vals], target: Vals) -> float:
    """Open fraction: one minus the best row-match rate."""
    total = len(target) or 1
    best = max((sum(1 for a, b in zip(sig, target) if a == b)
                for sig in sigs), default=0)
    return 1.0 - best / total


def _step(policy: SearchPolicy, ctx: dict[str, Any],
          rows: Rows, target: Vals, size: int, bound: int,
          budget: int, task_id: str, oracle: OracleHook | None
          ) -> Any:
    triples = layer_triples(ctx["levels"], size)
    sigs = _intern(ctx, rows, triples, size)
    _members(ctx, triples, sigs, size)
    feats = [encode(op, signature(left, rows),
                    signature(right, rows), target,
                    left.size, right.size, size, bound,
                    ctx["tried"] + i + 1, budget)
             for i, (op, left, right) in enumerate(triples)]
    gap = _layer_gap(sigs, target)
    return _scan(ctx, triples, sigs, policy.rank(feats, triples,
                                                gap),
                 target, task_id, oracle)
def _accept(ctx: dict[str, Any], found: Any) -> dict[str, Any]:
    try:
        graph = ctx["space"].apply(ctx["graph"], found)
    except ValueError as exc:
        return {"solved": False, "tried": ctx["tried"],
                "graph": None, "rejected": ctx["rejected"],
                "verdict": "validator: " + str(exc)}
    return {"solved": True, "tried": ctx["tried"],
            "graph": graph, "rejected": ctx["rejected"],
            "verdict": ""}
def guided_search(policy: SearchPolicy, rows: Rows,
                  expected: Vals, bound: int, budget: int,
                  task_id: str = "",
                  oracle: OracleHook | None = None,
                  consts: tuple[int, ...] = ()
                  ) -> dict[str, Any]:
    """Policy-ranked search; oracle vetoes retry past near-misses.

    Constant terminals join level zero when values are
    given; empty consts searches exactly as before.
    """
    ctx = _new_ctx(consts)
    for size in range(1, bound + 1):
        found = _step(policy, ctx, rows, expected, size, bound,
                      budget, task_id, oracle)
        if found is not None:
            return _accept(ctx, found)
    return {"solved": False, "tried": ctx["tried"],
            "graph": None, "rejected": ctx["rejected"],
            "verdict": "budget-exhausted"}
