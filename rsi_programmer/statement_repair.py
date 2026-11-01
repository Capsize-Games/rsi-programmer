"""Bounded counterexample-guided resynthesis with no held-test interface."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rsi_programmer.statement_execution import execute

Examples = list[tuple[list[int], int]]
Solver = Callable[[Examples], dict[str, Any]]
Oracle = Callable[[list[int]], int]


def refine(solve: Solver, examples: Examples, query_pool: list[list[int]],
           oracle: Oracle, rounds: int = 3) -> dict[str, Any]:
    """Add a distinguishing query example, then let the learner choose again.

    With no candidate, use the first unused query. With a candidate, use
    the first query on which it disagrees with the task engine. The caller
    owns this explicitly training-visible pool; held data is never passed.
    """
    if rounds < 0 or not examples or not query_pool:
        raise ValueError("examples, queries and nonnegative rounds required")
    observed = [(values[:], expected) for values, expected in examples]
    history = []
    oracle_calls = executions = 0
    for step in range(rounds + 1):
        attempt = solve(observed)
        tokens = attempt["tokens"]
        entry = {"round": step, "attempt": attempt,
                 "observed_count": len(observed), "counterexample": None}
        history.append(entry)
        if step == rounds:
            break
        used = {tuple(values) for values, _ in observed}
        selected = None
        for values in query_pool:
            if tuple(values) in used:
                continue
            expected = oracle(values)
            oracle_calls += 1
            actual = None
            if tokens is not None:
                executions += 1
                try:
                    actual = execute(tuple(tokens), values)
                except ValueError:
                    pass
                if actual == expected:
                    continue
            selected = (values[:], expected)
            entry["counterexample"] = {"input": values[:],
                                        "expected": expected, "actual": actual}
            break
        if selected is None:
            break
        observed.append(selected)
    return {"chosen": history[-1]["attempt"], "history": history,
            "observed": observed,
            "added_examples": len(observed) - len(examples),
            "oracle_calls": oracle_calls, "candidate_executions": executions}
