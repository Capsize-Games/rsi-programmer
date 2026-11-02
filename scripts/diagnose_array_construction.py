"""Audit frozen array-policy rankings against supplied controls.

Control tokens are audit labels only. They are never passed to fitting or
runtime synthesis, and this script does not alter policy parameters.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from rsi_programmer.array_curriculum import teachers
from rsi_programmer.array_policy import ArrayPolicy, prefix_ids
from rsi_programmer.array_syntax import TOKENS, analyze, parse
from rsi_programmer.array_tasks import CONTROL, TASKS, oracle
from rsi_programmer.benchmark_policy import state_fingerprint

VISIBLE = [[-3, 0, 1, 4, -1], [2, -2, 5, 1], [0, 3, -4, 2]]
Examples = list[tuple[list[int], list[int]]]
CAMPAIGNS = {
    "artifacts/array-statements-v1-20261101/report.json":
        ("trained_hash", {"seed": 0}),
    "artifacts/array-statements-context-v1-20261101/report.json":
        ("trained_hash", {"seed": 0, "execution_context": True}),
    "artifacts/array-statements-broad-search-v1-20261101/report.json":
        ("trained_hash", {"seed": 0, "execution_context": True}),
}


def _examples(name: str) -> Examples:
    return [(values, oracle(name, values)) for values in VISIBLE]


def _legal_probabilities(policy: ArrayPolicy, examples: Examples,
                         prefix: tuple[str, ...]) -> tuple[dict[str, float],
                                                           list[str]]:
    inputs = [row[0] for row in examples]
    targets = [row[1] for row in examples]
    matrix = policy.observed_features(inputs, targets, prefix).unsqueeze(0)
    logits = policy(matrix, torch.tensor([prefix_ids(prefix)]))[0]
    _, allowed = analyze(prefix)
    probs = torch.softmax(logits.masked_fill(
        torch.tensor([token not in allowed for token in TOKENS]), -torch.inf),
        dim=0)
    return {token: float(probs[TOKENS.index(token)].detach())
            for token in allowed}, allowed


def _teacher_forced(policy: ArrayPolicy, name: str) -> dict[str, Any]:
    tokens = CONTROL[name]
    examples = _examples(name)
    ranks: list[int] = []
    chosen_probabilities: list[float] = []
    for index, token in enumerate(tokens):
        probs, allowed = _legal_probabilities(policy, examples,
                                              tokens[:index])
        ordered = sorted(allowed, key=lambda item: (-probs[item],
                                                     TOKENS.index(item)))
        ranks.append(ordered.index(token) + 1)
        chosen_probabilities.append(probs[token])
    return {"action_count": len(tokens),
            "rank_1": sum(rank == 1 for rank in ranks),
            "mean_rank": sum(ranks) / len(ranks), "ranks": ranks,
            "correct_prefix_probability": _product(chosen_probabilities)}


def _product(values: list[float]) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result


def _tree_features(tokens: tuple[str, ...]) -> dict[str, Any]:
    body = parse(tokens)
    statement_kinds: Counter[str] = Counter()
    expression_kinds: Counter[str] = Counter()
    comparisons: Counter[str] = Counter()
    max_depth = 0

    def expression(node: tuple[Any, ...]) -> None:
        if node[0] in ("add", "sub", "mul", "rem"):
            expression_kinds[node[0]] += 1
            expression(node[1])
            expression(node[2])
        else:
            expression_kinds[node[0]] += 1

    def walk(block: tuple[tuple[Any, ...], ...], depth: int) -> None:
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        for statement in block:
            statement_kinds[statement[0]] += 1
            if statement[0] == "append":
                expression(statement[1])
            else:
                comparisons[statement[1][0]] += 1
                expression(statement[1][1])
                expression(statement[1][2])
                walk(statement[2], depth + 1)

    walk(body, 0)
    return {"if_count": statement_kinds["if"],
            "append_count": statement_kinds["append"],
            "max_condition_depth": max_depth,
            "comparisons": dict(comparisons),
            "expressions": dict(expression_kinds)}


def _beam_trace(policy: ArrayPolicy, name: str, width: int,
                choices: int = 4) -> dict[str, Any]:
    witness = CONTROL[name]
    examples = _examples(name)
    beam: list[tuple[float, tuple[str, ...]]] = [(0.0, ())]
    survival: list[dict[str, Any]] = []
    completed: set[tuple[str, ...]] = set()
    first_pruning: dict[str, Any] | None = None
    for depth in range(40):
        candidates: list[tuple[float, tuple[str, ...]]] = []
        correct_parent_live = any(prefix == witness[:len(prefix)]
                                  for _, prefix in beam)
        correct_token_rank: int | None = None
        correct_token_in_expansion = False
        for score, prefix in beam:
            probs, allowed = _legal_probabilities(policy, examples, prefix)
            ranked = sorted(allowed, key=lambda item: (-probs[item],
                                                        TOKENS.index(item)))
            if correct_parent_live and prefix == witness[:depth]:
                correct_token_rank = ranked.index(witness[depth]) + 1
                correct_token_in_expansion = witness[depth] in ranked[:choices]
            for token in ranked[:choices]:
                child = (*prefix, token)
                next_score = score + math.log(probs[token])
                if token == "stop":
                    completed.add(child)
                else:
                    candidates.append((next_score, child))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        correct_candidate_live = any(prefix == witness[:depth + 1]
                                     for _, prefix in candidates)
        if (first_pruning is None and correct_parent_live
                and not correct_token_in_expansion):
            first_pruning = {"depth": depth + 1,
                             "reason": "token_choices_cutoff",
                             "correct_action_rank": correct_token_rank}
        elif (first_pruning is None and correct_candidate_live
              and len(candidates) > width
              and not any(prefix == witness[:depth + 1]
                          for _, prefix in candidates[:width])):
            first_pruning = {"depth": depth + 1, "reason": "beam_width",
                             "correct_action_rank": correct_token_rank}
        beam = candidates[:width]
        survives = (witness in completed or any(
            prefix == witness[:len(prefix)] for _, prefix in beam))
        survival.append({"depth": depth + 1, "beam_size": len(beam),
                         "correct_prefix_survives": survives,
                         "distinct_prefixes": len({p for _, p in beam}),
                         "completed_candidates_seen": len(completed)})
        if not survives:
            break
    correct_probability = _product([
        _legal_probabilities(policy, examples, witness[:index])[0][token]
        for index, token in enumerate(witness)])
    return {"width": width, "token_choices": choices,
            "scope_note": "Only this supplied witness path is traced; its "
                          "pruning does not prove every equivalent program "
                          "was pruned.",
            "first_pruning_point": first_pruning,
            "correct_prefix_probability": correct_probability,
            "survival": survival,
            "completed_candidate_count": len(completed),
            "completed_candidate_hashes": sorted(
                hashlib.sha256(" ".join(tokens).encode()).hexdigest()
                for tokens in completed)}


def diagnose(policy: ArrayPolicy) -> dict[str, Any]:
    teacher_set = teachers()
    teacher_features = [_tree_features(item) for item in teacher_set]
    coverage = {
        "teacher_functions": len(teacher_set),
        "contains_conditions": sum("if" in item for item in teacher_set),
        "contains_arithmetic": sum(any(token in item for token in
                                        ("add", "sub", "mul"))
                                    for item in teacher_set),
        "contains_multiple_appends": sum(item.count("append") > 1
                                          for item in teacher_set),
        "contains_nested_conditions": sum(item.count("if") > 1
                                           for item in teacher_set),
        "condition_comparisons": dict(Counter(
            operator for feature in teacher_features
            for operator, count in feature["comparisons"].items()
            for _ in range(count))),
        "expression_nodes": dict(Counter(
            operator for feature in teacher_features
            for operator, count in feature["expressions"].items()
            for _ in range(count))),
        "body_structures": dict(Counter(
            "if=%d,append=%d,depth=%d" % (
                feature["if_count"], feature["append_count"],
                feature["max_condition_depth"])
            for feature in teacher_features)),
        "teacher_token_counts": {token: sum(item.count(token)
                                             for item in teacher_set)
                                 for token in TOKENS},
    }
    tasks: dict[str, Any] = {}
    for name in TASKS:
        witness = CONTROL[name]
        parse(witness)
        tasks[name] = {
            "grammarable": True,
            "witness_token_count": len(witness),
            "witness_structure": _tree_features(witness),
            "teacher_forced": _teacher_forced(policy, name),
            "beam_32": _beam_trace(policy, name, 32),
            "beam_256": _beam_trace(policy, name, 256, choices=8),
        }
    return {"scope": "Frozen-model control-witness audit only; controls are "
                     "not training data, runtime guidance, or donors.",
            "execution_context": policy.execution_context,
            "teacher_coverage": coverage, "tasks": tasks}


def _last_survival(trace: dict[str, Any]) -> bool:
    rows = trace["survival"]
    return bool(rows and rows[-1]["correct_prefix_survives"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--execution-context", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = ArrayPolicy(seed=0, execution_context=args.execution_context)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    metadata = {"seed": 0}
    metadata["execution_context"] = args.execution_context
    fingerprint = state_fingerprint(policy.state_dict(), metadata)
    legacy_fingerprint = state_fingerprint(policy.state_dict(), {"seed": 0})
    matches = []
    for relative, (key, expected_metadata) in CAMPAIGNS.items():
        report_path = Path(relative)
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        has_metadata = "metadata" in report
        report_fingerprint = (fingerprint if has_metadata else
                              legacy_fingerprint)
        if report.get(key) == report_fingerprint:
            matches.append({"report": relative,
                            "expected_metadata": expected_metadata,
                            "fingerprint_basis": (
                                "exact report metadata" if has_metadata else
                                "legacy seed-only metadata; checkpoint exact "
                                "metadata is recorded separately"),
                            "report_fingerprint": report_fingerprint,
                            "outcomes": {
                                row["id"]: row["chosen"].get("tokens")
                                is not None
                                for row in report.get("rows", [])
                                if row.get("method") == "neural"}})
    if not matches:
        raise ValueError("checkpoint fingerprint matches no retained array "
                         "campaign for this architecture")
    result = diagnose(policy)
    result["checkpoint"] = str(args.checkpoint)
    result["checkpoint_sha256"] = hashlib.sha256(
        args.checkpoint.read_bytes()).hexdigest()
    result["fingerprint"] = fingerprint
    result["fingerprint_metadata"] = metadata
    result["legacy_seed_only_fingerprint"] = legacy_fingerprint
    result["matching_campaigns"] = matches
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = {
        name: {"grammarable": row["grammarable"],
               "mean_teacher_rank": row["teacher_forced"]["mean_rank"],
               "beam32_survives": _last_survival(row["beam_32"]),
               "beam256_survives": _last_survival(row["beam_256"])}
        for name, row in result["tasks"].items()}
    print(json.dumps({"output": str(args.output), "fingerprint": fingerprint,
                      "matching_campaigns": matches,
                      "tasks": summary},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
