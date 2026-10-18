#!/usr/bin/env python3
"""Policy-ranked search that evaluates only selected candidates (#72)."""
from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Iterator, cast

from rsi_programmer.actions import ActionSpace
from rsi_programmer.policy import OP_ORDER, Rows, SearchPolicy, Triple, Vals
from rsi_programmer.policy import encode, layer_triples
from rsi_programmer.probe import CONSTS, OPS, Expr
from rsi_programmer.probe import const_terms, minimize_graph, terminals

OracleHook = Callable[[dict[str, Any]], bool]


def _validate(rows: Rows, expected: Vals, bound: int,
              budget: int, consts: tuple[int, ...],
              alternatives_per_signature: int) -> None:
    """Reject malformed search inputs before doing work."""
    if not isinstance(rows, tuple) or not rows:
        raise ValueError("rows must be a non-empty tuple")
    if any(not isinstance(row, tuple) or len(row) != 2
           or any(type(value) is not int for value in row)
           for row in rows):
        raise ValueError("rows must contain aligned integer pairs")
    if not isinstance(expected, tuple) or len(expected) != len(rows):
        raise ValueError("expected must align with rows")
    if any(type(value) is not int for value in expected):
        raise ValueError("expected values must be integers")
    if type(bound) is not int or bound < 0:
        raise ValueError("bound must be a non-negative integer")
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a non-negative integer")
    if not isinstance(consts, tuple) or any(
            type(value) is not int or value not in CONSTS
            for value in consts):
        raise ValueError("constants must be supported integer terminals")
    if (type(alternatives_per_signature) is not int
            or alternatives_per_signature < 1):
        raise ValueError("alternative cap must be a positive integer")


def _terminal_vector(term: Expr, rows: Rows) -> Vals:
    """Return a terminal's row vector without recursive probe evaluation."""
    if term.op == "input":
        col = 0 if term.left == "x" else 1
        return tuple(row[col] for row in rows)
    value = cast(int, term.left)
    return tuple(value for _row in rows)


def _candidate_vector(op: str, left: Vals, right: Vals) -> Vals:
    """Compute one candidate from already cached operand vectors."""
    fn = OPS[op]
    return tuple(fn(a, b) for a, b in zip(left, right))


def _iter_triples(levels: dict[int, list[Expr]],
                  size: int) -> Iterator[Triple]:
    """Yield one layer in the reference grammar's generation order."""
    for left_size in range(size):
        right_size = size - left_size - 1
        for left in levels[left_size]:
            for right in levels[right_size]:
                for op in OP_ORDER:
                    yield op, left, right


def _rank(policy: Any, feats: list[list[float]],
          triples: list[Triple], gap: float,
          metrics: dict[str, Any]) -> list[int]:
    """Run a policy and require a complete permutation of the layer."""
    started = perf_counter()
    order = policy.rank(feats, triples, gap)
    metrics["policy_time_seconds"] += perf_counter() - started
    metrics["policy_batches"] += 1
    if not isinstance(order, list) or any(
            type(index) is not int for index in order):
        raise ValueError("policy rank must return integer indices")
    if len(order) != len(triples) or set(order) != set(range(len(triples))):
        raise ValueError("policy rank must be a permutation of candidates")
    metrics["scored_descriptors"] += len(triples)
    steps = getattr(policy, "_steps", 0)
    if type(steps) is int and steps > 0:
        metrics["policy_batch_ticks"] += steps
        metrics["policy_row_steps"] += steps * len(feats)
        if isinstance(policy, SearchPolicy):
            metrics["snn_steps"] += steps
    return order


def _graph_for(root: Expr, const_values: tuple[int, ...],
               metrics: dict[str, Any]) -> dict[str, Any]:
    """Build and minimize only the accepted expression's typed DAG."""
    space = ActionSpace()
    graph: dict[str, Any] = {"input_types": {"x": "Int", "y": "Int"},
                             "nodes": [], "output": "x"}
    refs: dict[int, str | int] = {}

    def visit(term: Expr | str) -> str | int:
        nonlocal graph
        if isinstance(term, str):
            return term
        if id(term) in refs:
            return refs[id(term)]
        if term.op == "input":
            ref = cast(str, term.left)
            refs[id(term)] = ref
            return ref
        if term.op == "const":
            value = cast(int, term.left)
            if value not in const_values:
                raise ValueError("constant is absent from search terminals")
            action: dict[str, Any] = {"kind": "const", "value": value}
        else:
            action = {"kind": "apply", "op": term.op,
                      "left": visit(cast(Expr, term.left)),
                      "right": visit(cast(Expr, term.right))}
        started = perf_counter()
        graph = space.apply(graph, action)
        metrics["validation_time_seconds"] += perf_counter() - started
        metrics["graph_validation_calls"] += 1
        ref = len(graph["nodes"]) - 1
        refs[id(term)] = ref
        return ref

    graph["output"] = visit(root)
    started = perf_counter()
    graph = minimize_graph(graph)
    metrics["graph_validation_calls"] += 1
    metrics["validation_time_seconds"] += perf_counter() - started
    return graph


def _accept(root: Expr, const_values: tuple[int, ...],
            task_id: str, oracle: OracleHook | None,
            metrics: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Build a candidate graph and apply independent validation/oracle."""
    try:
        graph = _graph_for(root, const_values, metrics)
    except ValueError as exc:
        return None, "validator: " + str(exc)
    if oracle is None:
        return graph, ""
    started = perf_counter()
    metrics["oracle_calls"] += 1
    try:
        accepted = oracle(graph)
    finally:
        metrics["oracle_time_seconds"] += perf_counter() - started
    if accepted:
        return graph, ""
    return None, "oracle-rejected"


def pre_evaluation_search(
        policy: SearchPolicy | Any | None, rows: Rows, expected: Vals,
        bound: int, budget: int, task_id: str = "",
        oracle: OracleHook | None = None,
        consts: tuple[int, ...] = (),
        alternatives_per_signature: int = 1,
        signature_budget: int | None = None) -> dict[str, Any]:
    """Search by ranking descriptors before lazily evaluating candidates.

    A non-None policy ranks every descriptor in a complete layer, using only
    training targets and cached operand vectors. ``budget`` limits actual
    candidate signature evaluations. An interrupted layer is not retained,
    so later-layer reachability is incomplete by design. Completed layers
    retain alternatives in grammar generation order, bounded per
    operator/training-signature key.
    """
    _validate(rows, expected, bound, budget, consts,
              alternatives_per_signature)
    if signature_budget is not None and (
            type(signature_budget) is not int or signature_budget < 0):
        raise ValueError("signature budget must be non-negative")
    if oracle is not None and not callable(oracle):
        raise ValueError("oracle must be callable")
    cap = budget if signature_budget is None else min(
        budget, signature_budget)
    started = perf_counter()
    terms = list(terminals()) + list(const_terms(consts))
    levels: dict[int, list[Expr]] = {0: terms}
    vectors = {id(term): _terminal_vector(term, rows) for term in terms}
    metrics: dict[str, Any] = {
        "solved": False, "tried": 0,
        "signature_evaluations": 0,
        "operand_signature_evaluations": len(vectors),
        "retained_operand_vectors": 0,
        "operand_vector_lookups": 0,
        "generated_descriptors": 0, "scored_descriptors": 0,
        "feature_builds": 0, "policy_batches": 0,
        "policy_time_seconds": 0.0, "snn_steps": 0,
        "policy_batch_ticks": 0, "policy_row_steps": 0,
        "node_row_operations": 0, "graph_validation_calls": 0,
        "validation_time_seconds": 0.0, "oracle_calls": 0,
        "oracle_time_seconds": 0.0, "rejected": [],
        "equivalent_alternatives_limit": alternatives_per_signature,
        "equivalent_alternatives_dropped": 0,
        "equivalent_alternatives_retained": 0,
        "signature_budget": signature_budget,
        "signature_budget_exhausted": False,
        "budget_exhausted": False,
    }

    def finish(graph: dict[str, Any] | None, verdict: str
               ) -> dict[str, Any]:
        metrics["elapsed_seconds"] = perf_counter() - started
        metrics["graph"] = graph
        metrics["verdict"] = verdict
        metrics["committed_nodes"] = (
            len(graph["nodes"]) if graph is not None else 0)
        if graph is not None:
            metrics["output_nodes"] = len(graph["nodes"])
        metrics["equivalent_alternatives_truncated"] = (
            metrics["equivalent_alternatives_dropped"] > 0)
        metrics["tried_unit"] = "candidate_signature_evaluations"
        metrics["budget_unit"] = "candidate_signature_evaluations"
        metrics["signature_budget_unit"] = (
            "candidate_signature_evaluations"
            if signature_budget is not None else None)
        if graph is not None:
            metrics["solved"] = True
        return metrics

    # Terminals have no candidate execution cost and are safe to test directly.
    for term in terms:
        if vectors[id(term)] != expected:
            continue
        graph, reason = _accept(term, consts, task_id, oracle, metrics)
        if graph is not None:
            return finish(graph, "")
        metrics["rejected"].append({
            "task": task_id, "op": term.op,
            "verdict": reason or "oracle-rejected",
            "signature": list(vectors[id(term)])})

    if cap == 0:
        metrics["signature_budget_exhausted"] = (
            signature_budget == 0)
        metrics["budget_exhausted"] = signature_budget != 0
        return finish(None, "signature-budget-exhausted"
                      if signature_budget == 0 else "budget-exhausted")

    for size in range(1, bound + 1):
        if metrics["signature_evaluations"] >= cap:
            metrics["budget_exhausted"] = True
            break
        if policy is None:
            triples: list[Triple] = []
            width = len(OP_ORDER) * sum(
                len(levels[left_size]) *
                len(levels[size - left_size - 1])
                for left_size in range(size))
            order: list[int] | None = None
        else:
            triples = layer_triples(levels, size)
            width = len(triples)
            metrics["generated_descriptors"] += width
            feats: list[list[float]] = []
            previous = [vectors[id(term)] for layer in levels.values()
                        for term in layer]
            total = len(expected)
            best = max((sum(a == b for a, b in zip(vec, expected))
                        for vec in previous), default=0)
            gap = 1.0 - best / (total or 1)
            for index, (op, left, right) in enumerate(triples):
                left_sig = vectors[id(left)]
                right_sig = vectors[id(right)]
                metrics["operand_vector_lookups"] += 2
                feats.append(encode(op, left_sig, right_sig, expected,
                                    left.size, right.size, size, bound,
                                    index + 1, width))
            metrics["feature_builds"] += len(feats)
            order = _rank(policy, feats, triples, gap, metrics)

        signatures: dict[int, Vals] = {}
        layer_complete = True
        candidates = (enumerate(_iter_triples(levels, size))
                      if order is None else
                      ((index, triples[index]) for index in order))
        for index, triple in candidates:
            if order is None:
                triples.append(triple)
                metrics["generated_descriptors"] += 1
            if metrics["signature_evaluations"] >= cap:
                metrics["signature_budget_exhausted"] = (
                    signature_budget is not None
                    and metrics["signature_evaluations"] >= signature_budget)
                metrics["budget_exhausted"] = not (
                    metrics["signature_budget_exhausted"])
                layer_complete = False
                break
            op, left, right = triple
            left_sig = vectors[id(left)]
            right_sig = vectors[id(right)]
            metrics["operand_vector_lookups"] += 2
            sig = _candidate_vector(op, left_sig, right_sig)
            expr = Expr(op, left, right, size)
            signatures[len(triples) - 1 if order is None else index] = sig
            metrics["signature_evaluations"] += 1
            metrics["tried"] += 1
            metrics["node_row_operations"] += len(rows)
            if sig == expected:
                graph, reason = _accept(expr, consts, task_id, oracle,
                                        metrics)
                if graph is not None:
                    return finish(graph, "")
                metrics["rejected"].append({
                    "task": task_id, "op": op,
                    "verdict": reason or "oracle-rejected",
                    "signature": list(sig)})

        if not layer_complete:
            break

        # Retain completed layer members in generation order regardless of
        # the policy's execution order.
        seen: dict[tuple[str, Vals], int] = {}
        members: list[Expr] = []
        for index, (op, left, right) in enumerate(triples):
            sig = signatures[index]
            key = (op, sig)
            count = seen.get(key, 0)
            if count >= alternatives_per_signature:
                metrics["equivalent_alternatives_dropped"] += 1
                continue
            seen[key] = count + 1
            if count:
                metrics["equivalent_alternatives_retained"] += 1
            expr = Expr(op, left, right, size)
            members.append(expr)
            vectors[id(expr)] = sig
        levels[size] = members
        metrics["retained_operand_vectors"] += len(members)

    if metrics["signature_budget_exhausted"]:
        verdict = "signature-budget-exhausted"
    elif metrics["budget_exhausted"]:
        verdict = "budget-exhausted"
    else:
        verdict = "no-solution-within-bound"
    return finish(None, verdict)
