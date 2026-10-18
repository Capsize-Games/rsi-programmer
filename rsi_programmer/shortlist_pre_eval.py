"""Experimental cheap-shortlist search with complete block fallback."""
from __future__ import annotations

import heapq
from itertools import islice
from time import perf_counter
from typing import Any, Callable

from rsi_programmer import pre_eval
from rsi_programmer.policy import OP_ORDER, Rows, SearchPolicy, Triple, Vals
from rsi_programmer.pre_eval_features import FeatureStatistics
from rsi_programmer.probe import OPS, Expr, const_terms, terminals

OracleHook = Callable[[dict[str, Any]], bool]


def shortlist_pre_evaluation_search(
        policy: SearchPolicy | Any | None, rows: Rows, expected: Vals,
        bound: int, budget: int, task_id: str = "",
        oracle: OracleHook | None = None,
        consts: tuple[int, ...] = (),
        alternatives_per_signature: int = 1,
        signature_budget: int | None = None, *,
        window_size: int = 64, shortlist_size: int = 4,
        cheap_row_index: int = 0, cheap_only: bool = False
        ) -> dict[str, Any]:
    """Rank a cheap shortlist, then evaluate the complete block fallback.

    Only shortlist ordering differs from streaming: every descriptor is
    executed at most once unless an accepted graph or cap stops the search.
    Completed layers are retained in original grammar order. With a policy,
    only the selected shortlist receives 12-feature construction; remaining
    descriptors are executed in generation order.
    """
    _validate_options(window_size, shortlist_size, cheap_row_index,
                      cheap_only)
    if policy is None and not cheap_only:
        return pre_eval.pre_evaluation_search(
            policy, rows, expected, bound, budget, task_id, oracle, consts,
            alternatives_per_signature, signature_budget)
    pre_eval._validate(rows, expected, bound, budget, consts,
                       alternatives_per_signature)
    if cheap_row_index >= len(rows):
        raise ValueError("cheap_row_index must select an observed row")
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
    vectors = {id(term): pre_eval._terminal_vector(term, rows)
               for term in terms}
    stats = None if cheap_only else FeatureStatistics(expected)
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
        "window_size": window_size, "shortlist_size": shortlist_size,
        "cheap_row_index": cheap_row_index, "cheap_only": cheap_only,
        "cheap_candidate_row_operations": 0,
        "cheap_operand_value_lookups": 0,
        "cheap_target_value_lookups": 0,
        "cheap_residual_arithmetic_operations": 0,
        "cheap_scored_descriptors": 0,
        "shortlist_descriptors": 0, "shortlist_executions": 0,
        "fallback_executions": 0, "descriptor_replays": 0,
        "duplicate_signature_evaluations": 0,
        "permanently_pruned_descriptors": 0,
        "skipped_descriptors": 0,
        "peak_live_descriptors": 0, "peak_live_features": 0,
        "peak_live_cheap_scores": 0,
        "layer_widths": [], "layer_generated_descriptors": [],
        "layer_executed_candidates": [],
        "layer_shortlist_descriptors": [],
        "layer_shortlist_executions": [],
        "layer_fallback_executions": [],
    }

    def finish(graph: dict[str, Any] | None, verdict: str
               ) -> dict[str, Any]:
        if stats is not None:
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
        metrics["scored_unexecuted_descriptors"] = max(
            0, metrics["scored_descriptors"]
            - metrics["shortlist_executions"])
        metrics["cheap_scored_unexecuted_descriptors"] = max(
            0, metrics["cheap_scored_descriptors"]
            - metrics["signature_evaluations"])
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

    target_value: int | None = None

    def mark_cap_exhaustion() -> None:
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
        for name in ("layer_generated_descriptors",
                     "layer_executed_candidates",
                     "layer_shortlist_descriptors",
                     "layer_shortlist_executions",
                     "layer_fallback_executions"):
            metrics[name].append(0)
        if stats is None:
            gap = 0.0
        else:
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
            if remaining <= 0:
                complete = False
                mark_cap_exhaustion()
                break
            block = list(islice(generated, min(window_size,
                                               width - ordinal)))
            if not block:
                complete = False
                break
            metrics["generated_descriptors"] += len(block)
            metrics["layer_generated_descriptors"][-1] += len(block)
            metrics["peak_live_descriptors"] = max(
                metrics["peak_live_descriptors"], len(block))
            if target_value is None:
                target_value = expected[cheap_row_index]
                metrics["cheap_target_value_lookups"] = 1
            assert target_value is not None
            ranked_cheap: list[tuple[int, int, int]] = []
            row_index = cheap_row_index
            for local, (op, left, right) in enumerate(block):
                left_value = vectors[id(left)][row_index]
                right_value = vectors[id(right)][row_index]
                metrics["cheap_operand_value_lookups"] += 2
                residual = abs(OPS[op](left_value, right_value)
                               - target_value)
                metrics["cheap_candidate_row_operations"] += 1
                metrics["cheap_residual_arithmetic_operations"] += 2
                ranked_cheap.append((residual, ordinal + local, local))
            metrics["cheap_scored_descriptors"] += len(block)
            metrics["peak_live_cheap_scores"] = max(
                metrics["peak_live_cheap_scores"], len(ranked_cheap))
            selected_cheap = heapq.nsmallest(
                min(shortlist_size, len(block)), ranked_cheap)
            selected_indices = [item[2] for item in selected_cheap]
            selected_set = set(selected_indices)
            metrics["shortlist_descriptors"] += len(selected_indices)
            metrics["layer_shortlist_descriptors"][-1] += len(
                selected_indices)
            if stats is None:
                execution_order = selected_indices + [
                    index for index in range(len(block))
                    if index not in selected_set]
            else:
                features: list[list[float]] = []
                selected_triples = [block[index]
                                    for index in selected_indices]
                for index, (op, left, right) in zip(
                        selected_indices, selected_triples):
                    left_sig, right_sig = vectors[id(left)], vectors[id(right)]
                    metrics["operand_vector_lookups"] += 2
                    features.append(stats.encode(
                        op, id(left), left_sig, id(right), right_sig,
                        left.size, right.size, size, bound,
                        ordinal + index + 1, width))
                metrics["feature_builds"] += len(features)
                metrics["peak_live_features"] = max(
                    metrics["peak_live_features"], len(features))
                policy_order = pre_eval._rank(
                    policy, features, selected_triples, gap, metrics)
                execution_order = [selected_indices[index]
                                   for index in policy_order]
                execution_order.extend(
                    index for index in range(len(block))
                    if index not in selected_set)
            block_results: list[tuple[Expr, Vals] | None] = [
                None] * len(block)
            for rank_position, index in enumerate(execution_order):
                if metrics["signature_evaluations"] >= cap:
                    complete = False
                    mark_cap_exhaustion()
                    break
                op, left, right = block[index]
                left_sig, right_sig = vectors[id(left)], vectors[id(right)]
                metrics["operand_vector_lookups"] += 2
                signature = pre_eval._candidate_vector(
                    op, left_sig, right_sig)
                expr = Expr(op, left, right, size)
                block_results[index] = (expr, signature)
                metrics["signature_evaluations"] += 1
                metrics["layer_executed_candidates"][-1] += 1
                metrics["tried"] += 1
                metrics["node_row_operations"] += len(rows)
                if rank_position < len(selected_indices):
                    metrics["shortlist_executions"] += 1
                    metrics["layer_shortlist_executions"][-1] += 1
                else:
                    metrics["fallback_executions"] += 1
                    metrics["layer_fallback_executions"][-1] += 1
                if signature == expected:
                    graph, reason = pre_eval._accept(
                        expr, consts, task_id, oracle, metrics)
                    if graph is not None:
                        return finish(graph, "")
                    metrics["rejected"].append({
                        "task": task_id, "op": op,
                        "verdict": reason or "oracle-rejected",
                        "signature": list(signature)})
            for result in block_results:
                if result is None:
                    continue
                expr, signature = result
                key = (expr.op, signature)
                count = seen.get(key, 0)
                if count >= alternatives_per_signature:
                    metrics["equivalent_alternatives_dropped"] += 1
                    continue
                seen[key] = count + 1
                if count:
                    metrics["equivalent_alternatives_retained"] += 1
                retained.append((expr, signature))
            ordinal += len(block)
            if not complete:
                break
        if complete and ordinal == width:
            levels[size] = [expr for expr, _signature in retained]
            for expr, signature in retained:
                vectors[id(expr)] = signature
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


def _validate_options(window_size: int, shortlist_size: int,
                      cheap_row_index: int, cheap_only: bool) -> None:
    """Validate bounded shortlist controls before search work."""
    if type(window_size) is not int or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    if (type(shortlist_size) is not int or shortlist_size < 1
            or shortlist_size > window_size):
        raise ValueError("shortlist_size must be in [1, window_size]")
    if type(cheap_row_index) is not int or cheap_row_index < 0:
        raise ValueError("cheap_row_index must be a non-negative integer")
    if type(cheap_only) is not bool:
        raise ValueError("cheap_only must be a boolean")
