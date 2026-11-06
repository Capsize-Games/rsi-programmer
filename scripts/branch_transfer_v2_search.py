"""Fixed v2 solver routes: classical primary, neural fallback."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rsi_programmer.statement_branch_search import synthesize as compose
from rsi_programmer.statement_fragment_classical import synthesize as classical

Pairs = list[tuple[list[int], int]]


def solve(method: str, models: dict[str, Any], old: Any, new: Any,
          branches: list[Any], examples: Pairs,
          remaining: Callable[[], float]) -> dict[str, Any]:
    budget = remaining()
    if budget <= 0:
        return {"tokens": None, "budget_exhausted": True}
    if method == "classical":
        return solve_classical(new, branches, examples, remaining)
    if method == "neural":
        return solve_neural(models, old, new, branches, examples,
                            remaining, budget)
    return solve_second(models, new, branches, examples, [],
                        remaining)


def solve_classical(new: Any, branches: list[Any], examples: Pairs,
                    remaining: Callable[[], float]) -> dict[str, Any]:
    first = compose(None, examples, branches,
                    budget_seconds=remaining())
    if first["tokens"] is not None or remaining() <= 0:
        return first
    budget = remaining()
    return (classical(examples, new["fragments"], budget)
            if budget > 0 else {"tokens": None})


def solve_neural(models: dict[str, Any], old: Any, new: Any,
                 branches: list[Any], examples: Pairs,
                 remaining: Callable[[], float], budget: float
                 ) -> dict[str, Any]:
    from rsi_programmer.statement_fragment_search import synthesize

    route: list[Any] = []
    first = synthesize(models["previous"], examples, old["fragments"],
                       budget_seconds=min(1.0, budget))
    route.append(first)
    if first["tokens"] is not None or remaining() <= 0:
        return {**first, "route": route}
    return solve_second(models, new, branches, examples, route,
                        remaining)


def solve_second(models: dict[str, Any], new: Any,
                 branches: list[Any], examples: Pairs,
                 route: list[Any], remaining: Callable[[], float]
                 ) -> dict[str, Any]:
    from rsi_programmer.statement_fragment_search import synthesize

    budget = remaining()
    if budget <= 0:
        return {"tokens": None, "route": route}
    second = synthesize(models["current"], examples, new["fragments"],
                        budget_seconds=min(1.0, budget))
    route.append(second)
    if second["tokens"] is not None or remaining() <= 0:
        return {**second, "route": route}
    budget = remaining()
    if budget <= 0:
        return {"tokens": None, "route": route}
    third = compose(models["current"], examples, branches,
                    budget_seconds=budget)
    return {**third, "route": [*route, third]}
