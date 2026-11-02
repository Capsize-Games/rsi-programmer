"""Generate engine-supervised array statements without target witnesses."""
from __future__ import annotations

from functools import lru_cache
import random

from rsi_programmer.array_execution import execute
from rsi_programmer.array_syntax import ARITHMETIC, COMPARISONS, LEAVES
from rsi_programmer.array_tasks import TASKS, oracle


def probes() -> list[list[int]]:
    rng = random.Random(723011)
    return [[], [0], [-1, 0, 1], [-3, 2, 2, 0, 4, -1]] + [
        [rng.randrange(-5, 6) for _ in range(rng.randrange(1, 10))]
        for _ in range(20)]


@lru_cache(maxsize=2)
def teachers(limit: int = 512) -> tuple[tuple[str, ...], ...]:
    """Exclude complete development functions on 24 declared probes."""
    rows = probes()
    seen = {tuple(tuple(oracle(name, values)) for values in rows)
            for name in TASKS}
    seen.add(tuple(() for _ in rows))
    rng = random.Random(723017)
    result = []

    def expr() -> tuple[str, ...]:
        if rng.random() < 0.3:
            return rng.choice(ARITHMETIC), rng.choice(LEAVES), (
                rng.choice(LEAVES))
        return (rng.choice(LEAVES),)

    def branch(nested: bool = False) -> tuple[str, ...]:
        condition = (rng.choice(COMPARISONS), *expr(), *expr())
        body = (branch() if nested else ("append", *expr()))
        return "if", *condition, *body, "end"

    for _ in range(limit * 30):
        kind = rng.random()
        body = (("append", *expr()) if kind < 0.1 else
                branch() if kind < 0.5 else
                (*branch(), *branch()) if kind < 0.95 else branch(True))
        tokens = ("for", *body, "end", "return", "output", "stop")
        try:
            signature = tuple(tuple(execute(tokens, values))
                              for values in rows)
        except ValueError:
            continue
        if signature not in seen:
            seen.add(signature)
            result.append(tokens)
            if len(result) == limit:
                return tuple(result)
    raise ValueError("array curriculum did not reach its declared size")
