#!/usr/bin/env python3
"""Policy-ranked search over typed DAGs (RSI search layer)."""

from __future__ import annotations

from typing import Any, Callable

from rsi_programmer.actions import ActionSpace
from rsi_programmer.policy import OP_ORDER, Rows, SearchPolicy, Triple, Vals
from rsi_programmer.policy import encode, layer_triples
from rsi_programmer.probe import Expr, const_node, const_terms
from rsi_programmer.probe import minimize_graph, signature, terminals
OracleHook = Callable[[dict[str, Any]], bool]
DEFAULT_ALTERNATIVE_SIGNATURE_BUDGET = 10_000


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
            "refs": refs, "interned": {}, "tried": 0, "rejected": [],
            "signature_evaluations": 0,
            "operand_signature_evaluations": 0,
            "equivalent_alternatives_dropped": 0,
            "equivalent_alternatives_retained": 0,
            "signature_budget_exhausted": False}


def _metrics(ctx: dict[str, Any], committed: int = 0) -> dict[str, Any]:
    """Separate eager signature work from ranked examinations."""
    return {"signature_evaluations": ctx["signature_evaluations"],
            "operand_signature_evaluations":
            ctx["operand_signature_evaluations"],
            "examined": ctx["tried"], "committed_nodes": committed,
            "tried_unit": "candidate_scans",
            "budget_unit": "candidate_scans",
            "deduplication": "operator_and_training_signature",
            "equivalent_alternatives_limit":
            ctx.get("equivalent_alternatives_limit", 1),
            "equivalent_alternatives_dropped":
            ctx["equivalent_alternatives_dropped"],
            "equivalent_alternatives_retained":
            ctx["equivalent_alternatives_retained"],
            "equivalent_alternatives_truncated":
            ctx["equivalent_alternatives_dropped"] > 0,
            "signature_budget": ctx.get("signature_budget"),
            "signature_budget_unit":
            ("candidate_signature_evaluations"
             if ctx.get("signature_budget") is not None else None),
            "signature_budget_exhausted":
            ctx["signature_budget_exhausted"]}
def _intern(ctx: dict[str, Any], rows: Rows,
            triples: list[Triple], size: int) -> list[Vals]:
    graph, refs = ctx["graph"], ctx["refs"]
    sigs = []
    for op, left, right in triples:
        expr = Expr(op, left, right, size)
        sigs.append(signature(expr, rows))
        ctx["signature_evaluations"] += 1
        key = (op, refs[id(left)], refs[id(right)])
        if key not in ctx["interned"]:
            node = {"args": [key[1], key[2]], "op": op}
            node["type"] = "Int"
            graph["nodes"].append(node)
            ctx["interned"][key] = len(graph["nodes"]) - 1
    return sigs
def _members(ctx: dict[str, Any], triples: list[Triple],
             sigs: list[Vals], size: int,
             alternatives_per_signature: int) -> None:
    seen: dict[Any, int] = {}
    ctx["levels"][size] = []
    for (op, left, right), sig in zip(triples, sigs):
        sig_key = (op, sig)
        count = seen.get(sig_key, 0)
        if count >= alternatives_per_signature:
            ctx["equivalent_alternatives_dropped"] += 1
            continue
        seen[sig_key] = count + 1
        if count:
            ctx["equivalent_alternatives_retained"] += 1
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
          task_id: str, oracle: OracleHook | None = None,
          budget: int | None = None) -> Any:
    for idx in order:
        if budget is not None and ctx["tried"] >= budget:
            return None
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
          budget: int, task_id: str, oracle: OracleHook | None,
          alternatives_per_signature: int, signature_budget: int | None
          ) -> Any:
    if ctx["tried"] >= budget:
        return None
    remaining = (None if signature_budget is None else
                 signature_budget - ctx["signature_evaluations"])
    if remaining is not None and remaining <= 0:
        if layer_triples(ctx["levels"], size, limit=1):
            ctx["signature_budget_exhausted"] = True
        return None
    width = len(OP_ORDER) * sum(
        len(ctx["levels"][left_size]) *
        len(ctx["levels"][size - left_size - 1])
        for left_size in range(size))
    triples = layer_triples(ctx["levels"], size,
                            limit=None if remaining is None
                            else remaining + 1)
    if remaining is not None and len(triples) > remaining:
        triples = triples[:remaining]
        ctx["signature_budget_exhausted"] = True
    if ctx["tried"] >= budget or not triples:
        return None
    # Gap is based on reachable lower layers, before this candidate batch.
    previous = [signature(e, rows) for level in ctx["levels"].values()
                for e in level]
    gap = _layer_gap(previous, target)
    ctx["operand_signature_evaluations"] += len(previous)
    sigs = _intern(ctx, rows, triples, size)
    _members(ctx, triples, sigs, size, alternatives_per_signature)
    feats = [encode(op, signature(left, rows),
                    signature(right, rows), target,
                    left.size, right.size, size, bound,
                    i + 1, width)
             for i, (op, left, right) in enumerate(triples)]
    ctx["operand_signature_evaluations"] += 2 * len(triples)
    return _scan(ctx, triples, sigs, policy.rank(feats, triples,
                                                gap),
                 target, task_id, oracle, budget)
def _accept(ctx: dict[str, Any], found: Any) -> dict[str, Any]:
    try:
        graph = minimize_graph(ctx["space"].apply(ctx["graph"], found))
    except ValueError as exc:
        return {"solved": False, "tried": ctx["tried"],
                "graph": None, "rejected": ctx["rejected"],
                "verdict": "validator: " + str(exc), **_metrics(ctx)}
    return {"solved": True, "tried": ctx["tried"],
            "graph": graph, "rejected": ctx["rejected"],
            "verdict": "", "output_nodes": len(graph["nodes"]),
            **_metrics(ctx, len(graph["nodes"]))}
def guided_search(policy: SearchPolicy, rows: Rows,
                  expected: Vals, bound: int, budget: int,
                  task_id: str = "",
                  oracle: OracleHook | None = None,
                  consts: tuple[int, ...] = (),
                  alternatives_per_signature: int = 1,
                  signature_budget: int | None = None
                  ) -> dict[str, Any]:
    """Policy-ranked search; oracle vetoes retry past near-misses.

    Budget limits ranked candidate scans; eager signature work is reported
    separately and may exceed this scan budget. Lower-layer train-signature
    deduplication is incomplete for held-out-correct alternatives. The
    alternative cap preserves a bounded number of representatives per
    operator/signature key; any dropped representatives are counted and mark
    the result as truncated, so this option does not imply completeness.
    """
    ctx = _new_ctx(consts)
    if budget < 0 or bound < 0:
        raise ValueError("bound and scan budget must be non-negative")
    if (type(alternatives_per_signature) is not int
            or alternatives_per_signature < 1):
        raise ValueError("alternative cap must be at least one")
    ctx["equivalent_alternatives_limit"] = alternatives_per_signature
    if signature_budget is None and alternatives_per_signature > 1:
        signature_budget = DEFAULT_ALTERNATIVE_SIGNATURE_BUDGET
    if (signature_budget is not None
            and (type(signature_budget) is not int
                 or signature_budget < 0)):
        raise ValueError("signature budget must be a non-negative integer")
    ctx["signature_budget"] = signature_budget
    for size in range(1, bound + 1):
        found = _step(policy, ctx, rows, expected, size, bound,
                      budget, task_id, oracle,
                      alternatives_per_signature, signature_budget)
        if found is not None:
            return _accept(ctx, found)
        if ctx["tried"] >= budget:
            break
        if ctx["signature_budget_exhausted"]:
            break
    return {"solved": False, "tried": ctx["tried"],
            "graph": None, "rejected": ctx["rejected"],
            "verdict": ("budget-exhausted" if ctx["tried"] >= budget
                        else "signature-budget-exhausted"
                        if ctx["signature_budget_exhausted"]
                        else "no-solution-within-bound"),
            **_metrics(ctx)}
