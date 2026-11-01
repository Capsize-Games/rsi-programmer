"""Learner-independent acceptance for the construction pilot."""
from __future__ import annotations

from typing import Any

from rsi_programmer.construction_state import Rows
from rsi_programmer.online_graph import Polynomial, graph_polynomial
from rsi_programmer.probe import OPS, interpret, validate_graph


def check_graph(graph: dict[str, Any] | None, observed: Rows, held: Rows,
                polynomial: Polynomial, bound: int = 4) -> dict[str, bool]:
    """Check the declared grammar/budget, both row sets and exact identity.

    Call only after construction ends. This function supplies no feedback
    to the controller and does not accept learner-reported correctness.
    """
    result = dict.fromkeys(("valid", "observed_correct", "held_correct",
                            "polynomial_correct", "accepted"), False)
    if type(bound) is not int or not 1 <= bound <= 8:
        return result
    for rows in (observed, held):
        if (not isinstance(rows, tuple) or not rows or any(
                not isinstance(row, tuple) or len(row) != 3
                or any(type(value) is not int for value in row)
                for row in rows)):
            return result
    if graph is None:
        return result
    try:
        validate_graph(graph)
        nodes = graph["nodes"]
        if len(nodes) > bound:
            return result
        costs: dict[str | int, int] = {"x": 0, "y": 0}
        for index, node in enumerate(nodes):
            if node["op"] not in OPS:
                return result
            costs[index] = 1 + sum(costs[arg] for arg in node["args"])
            if costs[index] > bound:
                return result
        result["valid"] = True
        result["observed_correct"] = all(
            interpret(graph, x, y) == want for x, y, want in observed
        )
        result["held_correct"] = all(
            interpret(graph, x, y) == want for x, y, want in held
        )
        result["polynomial_correct"] = graph_polynomial(graph) == polynomial
        result["accepted"] = all(result[key] for key in (
            "valid", "observed_correct", "held_correct", "polynomial_correct",
        ))
    except (KeyError, TypeError, ValueError, IndexError):
        return dict.fromkeys(result, False)
    return result
