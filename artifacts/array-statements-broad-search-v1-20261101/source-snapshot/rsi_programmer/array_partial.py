"""Complete array prefixes without a target and execute visible inputs."""
from __future__ import annotations

from functools import lru_cache

from rsi_programmer.array_execution import execute
from rsi_programmer.array_syntax import MAX_TOKENS, analyze


@lru_cache(maxsize=16384)
def complete(prefix: tuple[str, ...]) -> tuple[str, ...] | None:
    tokens = prefix
    while len(tokens) <= MAX_TOKENS:
        body, allowed = analyze(tokens)
        if body is not None:
            return tokens
        if not allowed:
            return None
        for token in ("stop", "output", "return", "end", "append",
                      "value", "gt"):
            if token in allowed:
                tokens = (*tokens, token)
                break
        else:
            tokens = (*tokens, allowed[0])
    return None


def outputs(prefix: tuple[str, ...], inputs: list[list[int]]
            ) -> tuple[list[list[int]], list[float]]:
    """Use fixed syntax defaults; no expected outputs enter completion."""
    tokens = complete(prefix)
    rows, valid = [], []
    for values in inputs:
        try:
            if tokens is None:
                raise ValueError("array prefix cannot fit syntax budget")
            rows.append(execute(tokens, values))
            valid.append(1.0)
        except ValueError:
            rows.append([])
            valid.append(0.0)
    return rows, valid
