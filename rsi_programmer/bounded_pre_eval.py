"""Experimental bounded-window policy search (#81)."""
from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

from rsi_programmer import pre_eval
from rsi_programmer.policy import OP_ORDER, Rows, SearchPolicy, Triple, Vals
from rsi_programmer.pre_eval_features import FeatureStatistics
from rsi_programmer.probe import Expr, const_terms
from rsi_programmer.probe import terminals

OracleHook = Callable[[dict[str, Any]], bool]


def bounded_pre_evaluation_search(
        policy: SearchPolicy | Any | None, rows: Rows, expected: Vals,
        bound: int, budget: int, task_id: str = "",
        oracle: OracleHook | None = None,
        consts: tuple[int, ...] = (),
        alternatives_per_signature: int = 1,
        signature_budget: int | None = None,
        window_size: int = 256,
        candidates_per_window: int = 64) -> dict[str, Any]:
    """Search ranked blocks, preserving whole-layer feature coordinates.

    Policy order is local to each grammar-order block. Consequently this is
    not equivalent to ranking an entire layer except when a block covers
    it. Active-block peaks exclude retained and pending layer members.
    Streaming delegates to the historical implementation unchanged.
    """
    if type(window_size) is not int or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    if (type(candidates_per_window) is not int
            or candidates_per_window < 1
            or candidates_per_window > window_size):
        raise ValueError("candidates_per_window must be in [1, window_size]")
    if policy is None:
        return pre_eval.pre_evaluation_search(
            policy, rows, expected, bound, budget, task_id, oracle, consts,
            alternatives_per_signature, signature_budget)
    pre_eval._validate(rows, expected, bound, budget, consts,
                       alternatives_per_signature)
    if signature_budget is not None and (
            type(signature_budget) is not int or signature_budget < 0):
        raise ValueError("signature budget must be non-negative")
    if oracle is not None and not callable(oracle):
        raise ValueError("oracle must be callable")

    cap = budget if signature_budget is None else min(budget,
                                                       signature_budget)
    started = perf_counter()
    terms = list(terminals()) + list(const_terms(consts))
    levels: dict[int, list[Expr]] = {0: terms}
    vectors = {id(term): pre_eval._terminal_vector(term, rows)
               for term in terms}
    metrics: dict[str, Any] = {
        "solved": False, "tried": 0, "signature_evaluations": 0,
        "operand_signature_evaluations": len(vectors),
        "retained_operand_vectors": 0, "operand_vector_lookups": 0,
        "generated_descriptors": 0, "scored_descriptors": 0,
        "feature_builds": 0, "policy_batches": 0,
        "feature_stat_cache_hits": 0, "feature_stat_cache_misses": 0,
        "feature_stat_cache_lookups": 0,
        "feature_target_range_elements_scanned": 0,
        "feature_operand_elements_scanned": 0,
        "reference_feature_target_range_elements_scanned": 0,
        "reference_feature_operand_elements_scanned": 0,
        "reference_gap_elements_scanned": 0,
        "policy_time_seconds": 0.0, "snn_steps": 0,
        "policy_batch_ticks": 0, "policy_row_steps": 0,
        "node_row_operations": 0, "graph_validation_calls": 0,
        "validation_time_seconds": 0.0, "oracle_calls": 0,
        "oracle_time_seconds": 0.0, "rejected": [],
        "equivalent_alternatives_limit": alternatives_per_signature,
        "equivalent_alternatives_dropped": 0,
        "equivalent_alternatives_retained": 0,
        "signature_budget": signature_budget,
        "signature_budget_exhausted": False, "budget_exhausted": False,
        "window_size": window_size,
        "candidates_per_window": candidates_per_window,
        "skipped_descriptors": 0, "peak_live_descriptors": 0,
        "peak_live_features": 0, "layer_widths": [],
        "layer_generated_descriptors": [], "layer_executed_candidates": [],
    }
    stats = FeatureStatistics(expected)

    def finish(graph: dict[str, Any] | None, verdict: str
               ) -> dict[str, Any]:
        metrics["feature_stat_cache_hits"] = stats.cache_hits
        metrics["feature_stat_cache_misses"] = stats.cache_misses
        metrics["feature_stat_cache_lookups"] = stats.cache_lookups
        metrics["feature_target_range_elements_scanned"] = (
            stats.target_range_elements_scanned)
        metrics["feature_operand_elements_scanned"] = (
            stats.operand_feature_elements_scanned)
        builds = metrics["feature_builds"]
        metrics["reference_feature_target_range_elements_scanned"] = (
            4 * len(expected) * builds)
        metrics["reference_feature_operand_elements_scanned"] = (
            4 * len(expected) * builds)
        metrics["scored_unexecuted_descriptors"] = (
            metrics["scored_descriptors"] - metrics["signature_evaluations"])
        metrics["elapsed_seconds"] = perf_counter() - started
        metrics["graph"] = graph
        metrics["verdict"] = verdict
        metrics["committed_nodes"] = len(graph["nodes"]) if graph else 0
        if graph is not None:
            metrics["output_nodes"] = len(graph["nodes"])
            metrics["solved"] = True
        metrics["equivalent_alternatives_truncated"] = (
            metrics["equivalent_alternatives_dropped"] > 0)
        metrics["tried_unit"] = "candidate_signature_evaluations"
        metrics["budget_unit"] = "candidate_signature_evaluations"
        metrics["signature_budget_unit"] = (
            "candidate_signature_evaluations"
            if signature_budget is not None else None)
        return metrics

    for term in terms:
        if vectors[id(term)] != expected:
            continue
        graph, reason = pre_eval._accept(term, consts, task_id, oracle,
                                         metrics)
        if graph is not None:
            return finish(graph, "")
        metrics["rejected"].append({
            "task": task_id, "op": term.op,
            "verdict": reason or "oracle-rejected",
            "signature": list(vectors[id(term)])})

    if cap == 0:
        metrics["signature_budget_exhausted"] = signature_budget == 0
        metrics["budget_exhausted"] = signature_budget != 0
        verdict = ("signature-budget-exhausted" if signature_budget == 0
                   else "budget-exhausted")
        return finish(None, verdict)

    def mark_cap_exhaustion() -> None:
        """Record which execution cap stopped candidate work."""
        metrics["signature_budget_exhausted"] = (
            signature_budget is not None
            and metrics["signature_evaluations"] >= signature_budget)
        metrics["budget_exhausted"] = not (
            metrics["signature_budget_exhausted"])

    for size in range(1, bound + 1):
        if metrics["signature_evaluations"] >= cap:
            mark_cap_exhaustion()
            break
        width = len(OP_ORDER) * sum(
            len(levels[left_size]) * len(levels[size - left_size - 1])
            for left_size in range(size))
        metrics["layer_widths"].append(width)
        metrics["layer_generated_descriptors"].append(0)
        metrics["layer_executed_candidates"].append(0)
        previous = [(id(term), vectors[id(term)])
                    for layer in levels.values() for term in layer]
        metrics["reference_gap_elements_scanned"] += sum(
            min(len(signature), len(expected))
            for _key, signature in previous)
        gap = stats.gap(previous)
        generated = pre_eval._iter_triples(levels, size)
        ordinal = 0
        complete = True
        seen: dict[tuple[str, Vals], int] = {}
        retained: list[tuple[Expr, Vals]] = []
        while ordinal < width:
            remaining = cap - metrics["signature_evaluations"]
            block_limit = min(window_size, width - ordinal)
            block: list[Triple] = []
            for _ in range(block_limit):
                try:
                    block.append(next(generated))
                except StopIteration:
                    break
            if not block:
                complete = False
                break
            feats: list[list[float]] = []
            for local, (op, left, right) in enumerate(block):
                lsig, rsig = vectors[id(left)], vectors[id(right)]
                metrics["operand_vector_lookups"] += 2
                feats.append(stats.encode(
                    op, id(left), lsig, id(right), rsig, left.size,
                    right.size, size, bound, ordinal + local + 1, width))
            metrics["generated_descriptors"] += len(block)
            metrics["layer_generated_descriptors"][-1] += len(block)
            metrics["feature_builds"] += len(feats)
            metrics["peak_live_descriptors"] = max(
                metrics["peak_live_descriptors"], len(block))
            metrics["peak_live_features"] = max(
                metrics["peak_live_features"], len(feats))
            order = pre_eval._rank(policy, feats, block, gap, metrics)
            planned = min(candidates_per_window, len(order))
            selected_count = min(planned, remaining)
            selected = order[:selected_count]
            cap_interrupted_block = selected_count < planned
            metrics["skipped_descriptors"] += len(block) - planned
            block_results: list[tuple[Expr, Vals] | None] = [None] * len(block)
            for index in selected:
                op, left, right = block[index]
                lsig, rsig = vectors[id(left)], vectors[id(right)]
                metrics["operand_vector_lookups"] += 2
                sig = pre_eval._candidate_vector(op, lsig, rsig)
                expr = Expr(op, left, right, size)
                block_results[index] = (expr, sig)
                metrics["signature_evaluations"] += 1
                metrics["layer_executed_candidates"][-1] += 1
                metrics["tried"] += 1
                metrics["node_row_operations"] += len(rows)
                if sig == expected:
                    graph, reason = pre_eval._accept(
                        expr, consts, task_id, oracle, metrics)
                    if graph is not None:
                        return finish(graph, "")
                    metrics["rejected"].append({
                        "task": task_id, "op": op,
                        "verdict": reason or "oracle-rejected",
                        "signature": list(sig)})
            # Compact retention state in original grammar order.
            for result in block_results:
                if result is None:
                    continue
                expr, sig = result
                key = (expr.op, sig)
                count = seen.get(key, 0)
                if count >= alternatives_per_signature:
                    metrics["equivalent_alternatives_dropped"] += 1
                    continue
                seen[key] = count + 1
                if count:
                    metrics["equivalent_alternatives_retained"] += 1
                retained.append((expr, sig))
            ordinal += len(block)
            if cap_interrupted_block:
                complete = False
                mark_cap_exhaustion()
                break
            if not complete:
                break
            if metrics["signature_evaluations"] >= cap and ordinal < width:
                complete = False
                mark_cap_exhaustion()
                break
        if complete and ordinal == width:
            levels[size] = [expr for expr, _sig in retained]
            for expr, sig in retained:
                vectors[id(expr)] = sig
            metrics["retained_operand_vectors"] += len(retained)
        else:
            break

    if metrics["signature_budget_exhausted"]:
        verdict = "signature-budget-exhausted"
    elif metrics["budget_exhausted"]:
        verdict = "budget-exhausted"
    else:
        verdict = "no-solution-within-bound"
    return finish(None, verdict)
