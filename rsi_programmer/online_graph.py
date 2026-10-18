"""Bounded graph execution with post-attempt online policy feedback.

This additive experiment chooses one root-operator lane at a time. It does
not rank a full candidate layer or change the historical search engines.
"""
from __future__ import annotations

import itertools
import math
import time
from typing import Any, Iterator

from rsi_programmer.probe import interpret, validate_graph

OPERATORS = ("add", "sub", "mul")
CONSTANTS = (2, 3, 4, 5)
LANE_SIZE = 144
GRAMMAR_SIZE = 432
Polynomial = tuple[tuple[int, int, int], ...]


def iter_candidates(root: str | None = None
                    ) -> Iterator[dict[str, Any]]:
    """Yield typed two-operation trees without executing candidates."""
    if root is not None and root not in OPERATORS:
        raise ValueError("unknown root operator")
    roots = OPERATORS if root is None else (root,)
    for constant in CONSTANTS:
        for shape in ("left", "right"):
            for leaves in itertools.permutations(("x", "y", "c")):
                refs = [0 if leaf == "c" else leaf for leaf in leaves]
                for inner in OPERATORS:
                    args = refs[:2] if shape == "left" else refs[1:]
                    outer = ([1, refs[2]] if shape == "left"
                             else [refs[0], 1])
                    for op in roots:
                        yield {
                            "input_types": {"x": "Int", "y": "Int"},
                            "nodes": [
                                {"op": "const", "type": "Int",
                                 "args": [], "value": constant},
                                {"op": inner, "type": "Int",
                                 "args": list(args)},
                                {"op": op, "type": "Int",
                                 "args": list(outer)},
                            ],
                            "output": 2,
                        }


def graph_polynomial(graph: dict[str, Any]) -> Polynomial:
    """Independently expand a returned graph for external acceptance."""
    validate_graph(graph)
    values: dict[str | int, dict[tuple[int, int], int]] = {
        "x": {(1, 0): 1}, "y": {(0, 1): 1},
    }
    for index, node in enumerate(graph["nodes"]):
        if node["op"] == "const":
            values[index] = {(0, 0): node["value"]}
            continue
        if node["op"] not in OPERATORS:
            raise ValueError("polynomial audit supports arithmetic only")
        first, second = (values[arg] for arg in node["args"])
        result: dict[tuple[int, int], int] = {}
        if node["op"] == "mul":
            for (xp, yp), coefficient in first.items():
                for (xq, yq), other in second.items():
                    key = xp + xq, yp + yq
                    result[key] = result.get(key, 0) + coefficient * other
        else:
            result.update(first)
            sign = 1 if node["op"] == "add" else -1
            for key, coefficient in second.items():
                result[key] = result.get(key, 0) + sign * coefficient
        values[index] = {key: value for key, value in result.items()
                         if value}
    return tuple((xp, yp, value) for (xp, yp), value
                 in sorted(values[graph["output"]].items()) if value)


def _observations(rows: tuple[tuple[int, int, int], ...]) -> None:
    if not isinstance(rows, tuple) or not rows or any(
            not isinstance(row, tuple) or len(row) != 3
            or any(type(value) is not int for value in row)
            for row in rows):
        raise ValueError("observed rows must be nonempty integer triples")


def run_episode(policy: Any | None,
                observed: tuple[tuple[int, int, int], ...], *,
                budget: int = 144, episode_id: str = "episode"
                ) -> dict[str, Any]:
    """Choose, execute, then update; never read held or symbolic answers.

    None selects lanes in the grammar's fixed round-robin order. A policy
    receives only six public features and a mask of unexhausted lanes.
    Every emitted graph is executed on every observed row. The first exact
    observed match stops the episode; external acceptance cannot retry it.
    """
    _observations(observed)
    if type(budget) is not int or not 0 <= budget <= GRAMMAR_SIZE:
        raise ValueError("budget must be an integer from 0 through 432")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode identity must be a nonempty string")
    started, wall = time.process_time(), time.perf_counter()
    lanes = {op: iter_candidates(op) for op in OPERATORS}
    cursors = {op: 0 for op in OPERATORS}
    features = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    attempts: list[dict[str, Any]] = []
    graph: dict[str, Any] | None = None
    policy_cpu = 0.0
    update_cpu = 0.0
    if policy is not None:
        policy.start_episode(episode_id)
    for step in range(budget):
        available = tuple(op for op in OPERATORS
                          if cursors[op] < LANE_SIZE)
        if not available:
            break
        tick = time.process_time()
        action = (policy.choose(features, available) if policy is not None
                  else next(op for offset in range(3)
                            if (op := OPERATORS[(step + offset) % 3])
                            in available))
        policy_cpu += time.process_time() - tick
        if action not in available:
            raise ValueError("policy selected an unavailable root lane")
        candidate = next(lanes[action])
        cursors[action] += 1
        predictions = tuple(interpret(candidate, x, y)
                            for x, y, _want in observed)
        error = sum(min(1.0, abs(got - want) / (1 + abs(want)))
                    for got, (_x, _y, want)
                    in zip(predictions, observed)) / len(observed)
        reward = 1.0 - error
        if not math.isfinite(reward):
            raise ValueError("execution produced a nonfinite reward")
        if policy is not None:
            tick = time.process_time()
            policy.observe(reward)
            update_cpu += time.process_time() - tick
        exact = all(got == row[2]
                    for got, row in zip(predictions, observed))
        attempts.append({"action": action, "features": list(features),
                         "graph": candidate,
                         "predictions": list(predictions),
                         "error": error, "reward": reward,
                         "observed_exact": exact})
        if exact:
            graph = candidate
            break
        features = (1.0, *(float(action == op) for op in OPERATORS),
                    error, (step + 1) / max(1, budget))
    if policy is not None:
        policy.end_episode()
    count = len(attempts)
    return {
        "graph": graph, "observed_solved": graph is not None,
        "attempts": attempts, "candidate_evaluations": count,
        "generated_candidates": count,
        "interpreter_rows": count * len(observed),
        "node_row_operations": count * len(observed) * 3,
        "graph_validation_calls": count * len(observed),
        "decision_calls": count if policy is not None else 0,
        "feedback_calls": count if policy is not None else 0,
        "policy_cpu_seconds": policy_cpu,
        "update_cpu_seconds": update_cpu,
        "episode_cpu_seconds": time.process_time() - started,
        "episode_wall_seconds": time.perf_counter() - wall,
        "budget_exhausted": graph is None and count == budget,
        "grammar_exhausted": graph is None and count == GRAMMAR_SIZE,
    }


def check_result(graph: dict[str, Any] | None,
                 held: tuple[tuple[int, int, int], ...],
                 polynomial: Polynomial) -> dict[str, Any]:
    """Check held rows and exact identity after the learner has ended."""
    _observations(held)
    started = time.process_time()
    if graph is None:
        return {"accepted": False, "held_correct": False,
                "polynomial_correct": False, "held_rows_executed": 0,
                "checker_cpu_seconds": time.process_time() - started}
    outputs = tuple(interpret(graph, x, y) for x, y, _want in held)
    held_correct = outputs == tuple(want for _x, _y, want in held)
    polynomial_correct = graph_polynomial(graph) == polynomial
    return {
        "accepted": held_correct and polynomial_correct,
        "held_correct": held_correct,
        "polynomial_correct": polynomial_correct,
        "held_rows_executed": len(held),
        "checker_cpu_seconds": time.process_time() - started,
    }
