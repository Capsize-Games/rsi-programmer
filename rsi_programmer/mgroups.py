#!/usr/bin/env python3
"""Layer teaching groups and per-head metrics for ModularPolicy.

Each layer of each task becomes one group: thirteen-dim drives with
the open-gap channel, the row's own op index, and positive/negative
row sets for the choice, op, operand, and emit terms. Reports score
each head against chance with the group counts that produced them.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any

from rsi_programmer.author import (
    open_gap,
    solution_sigs_by_size_report,
)
from rsi_programmer.modules import N_OP
from rsi_programmer.policy import OP_ORDER, Rows, Vals
from rsi_programmer.policy import encode, layer_triples
from rsi_programmer.probe import Expr, signature, terminals
from rsi_programmer.traces import TraceRecorder


def _layers(rows: Rows, target: Vals,
            bound: int) -> list[tuple[int, list, list[Vals]]]:
    """Deduplicated Int layers mirroring the enumerator order."""
    levels: dict[int, list[Expr]] = {0: list(terminals())}
    out: list[tuple[int, list, list[Vals]]] = []
    for size in range(1, bound + 1):
        triples = layer_triples(levels, size)
        lower_sigs = [signature(expr, rows)
                      for level in levels.values() for expr in level]
        exprs: dict[Any, Expr] = {}
        for op, left, right in triples:
            expr = Expr(op, left, right, size)
            exprs.setdefault((op, signature(expr, rows)), expr)
        levels[size] = list(exprs.values())
        out.append((size, triples, lower_sigs))
        if any(signature(e, rows) == target
               for e in levels[size]):
            break
    return out


def modular_groups(fixture: str,
                   budget: int = 10 ** 9,
                   supervision_report: list[dict[str, Any]] | None = None
                   ) -> list[dict[str, Any]]:
    """One teaching group per task layer with exact-size labels.

    A task whose retained solution graph hits its evidence cap is omitted
    rather than assigning false-negative labels to unexamined candidates.
    Pass ``supervision_report`` to inspect included and truncated tasks.
    """
    recorder = TraceRecorder(fixture)
    groups: list[dict[str, Any]] = []
    for task in recorder.load_family()["tasks"]:
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        want_by_size, evidence = solution_sigs_by_size_report(
            rows, target, bound)
        task_report = {
            "task": task["id"], "bound": bound,
            "mode": "reachability",
            "included": not evidence["truncated"], **evidence,
        }
        if supervision_report is not None:
            supervision_report.append(task_report)
        if evidence["truncated"]:
            continue
        for size, triples, lower_sigs in _layers(rows, target, bound):
            want = want_by_size.get(size, set())
            sigs = [signature(Expr(o, l, r, size), rows)
                    for o, l, r in triples]
            lefts = [signature(l, rows) for _, l, _ in triples]
            gap = open_gap(lower_sigs, target)
            feats = [encode(o, signature(l, rows),
                            signature(r, rows), target,
                            l.size, r.size, size, bound,
                            i + 1, len(triples))
                     for i, (o, l, r) in enumerate(triples)]
            pos = [i for i, sig in enumerate(sigs)
                   if sig in want]
            neg = [i for i in range(len(triples))
                   if i not in pos]
            ops = [OP_ORDER.index(o) for o, _, _ in triples]
            solops = {OP_ORDER.index(o) for i, (o, _, _)
                      in enumerate(triples) if i in pos}
            groups.append({"feats": feats, "gap": gap,
                           "task": task["id"],
                           "label_semantics": evidence["label_semantics"],
                           "label_size": size,
                           "pos": pos, "neg": neg, "ops": ops,
                           "solops": sorted(solops),
                           "sigs": [list(s) for s in sigs],
                           "lefts": [list(s) for s in lefts],
                           "want": sorted(map(list, want))})
    return groups


def _pairwise_preference(scores: list[float], pos: list[int],
                         neg: list[int]) -> float | None:
    """Fraction of objective pairs ordered positive-first, ties half."""
    if not pos or not neg:
        return None
    sorted_neg = sorted(scores[index] for index in neg)
    wins = ties = 0
    for p in pos:
        lower = bisect_left(sorted_neg, scores[p])
        upper = bisect_right(sorted_neg, scores[p])
        wins += lower
        ties += upper - lower
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _ranking_report(scores: dict[str, float],
                    layers: int) -> dict[str, Any]:
    """Build objective metrics independently of choice diagnostics."""
    names = {"combined": "combined", "op_own": "op_own",
             "left": "left_score", "right": "right_score",
             "emit": "emit"}
    result: dict[str, Any] = {
        "objective_ranking_layers": layers,
        "objective_ranking_chance": 0.5,
        "objective_ranking_role": "positive_negative_rows_as_in_group_loss"}
    for key, value in scores.items():
        result[names[key] + "_pairwise_preference_accuracy"] = (
            round(value / layers, 4) if layers else None)
    return result


def module_report(policy: Any,
                  groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Report objective-aligned row rankings and choice diagnostics.

    Opcode argmax is a not-directly-supervised diagnostic: the fit objective
    ranks each row's own opcode rate against other rows, without a within-row
    opcode classification loss. Older metric names remain compatibility
    aliases and are identified in ``legacy_metric_aliases``.
    """
    op_hits = op_chance = 0.0
    lo_hits = lo_chance = 0.0
    counted = 0
    objective_scores = {key: 0.0 for key in
                        ("combined", "op_own", "left", "right", "emit")}
    objective_layers = 0
    for group in groups:
        rates = policy.head_rates(group["feats"], group.get("gap"))
        n = len(group["feats"])
        if not n:
            continue
        if group.get("pos") and group.get("neg"):
            own = [rates["op"][i][group["ops"][i]]
                   for i in range(n)]
            combined = [own[i] + rates["lo"][i][0] +
                        rates["ro"][i][0] + rates["st"][i][0]
                        for i in range(n)]
            heads = {"combined": combined, "op_own": own,
                     "left": [row[0] for row in rates["lo"]],
                     "right": [row[0] for row in rates["ro"]],
                     "emit": [row[0] for row in rates["st"]]}
            values = {key: _pairwise_preference(
                score, group["pos"], group["neg"])
                      for key, score in heads.items()}
            for key, value in values.items():
                if value is not None:
                    objective_scores[key] += value
            objective_layers += 1
        if group["solops"]:
            counted += 1
            preds = [row.index(max(row)) for row in rates["op"]]
            op_hits += sum(1 for i, p in enumerate(preds)
                           if p == group["ops"][i]) / n
            op_chance += 1.0 / N_OP
            want = {tuple(s) for s in group["want"]}
            lo = rates["lo"]
            top = max(range(n), key=lambda i: lo[i][0])
            lo_hits += float(tuple(group["lefts"][top]) in want)
            lo_chance += sum(tuple(sig) in want
                             for sig in group["lefts"]) / n
    result = {"layers": counted,
              "op_argmax_role": "not_directly_supervised_diagnostic",
              "op_argmax_directly_supervised": False,
              "legacy_metric_names_deprecated": True,
              "left_operand_top1_role":
                  "candidate_row_signature_membership_diagnostic",
              "legacy_metric_aliases": {
                  "op_accuracy": "op_argmax_accuracy",
                  "op_chance": "op_argmax_chance",
                  "operand_top1": "left_operand_top1_match",
                  "operand_chance": "left_operand_top1_chance"}}
    result.update(_ranking_report(objective_scores, objective_layers))
    if not counted:
        return result
    op_accuracy = round(op_hits / counted, 4)
    op_chance_rate = round(op_chance / counted, 4)
    operand_match = round(lo_hits / counted, 4)
    operand_chance_rate = round(lo_chance / counted, 4)
    result.update({
        "op_argmax_accuracy": op_accuracy,
        "op_argmax_chance": op_chance_rate,
        "left_operand_top1_match": operand_match,
        "left_operand_top1_chance": operand_chance_rate,
        "op_accuracy": op_accuracy,
        "op_chance": op_chance_rate,
        "operand_top1": operand_match,
        "operand_chance": operand_chance_rate})
    return result
