"""Task-visible execution features for incomplete statement programs."""
from __future__ import annotations

from functools import lru_cache

from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_syntax import MAX_TOKENS, analyze


@lru_cache(maxsize=16384)
def complete(prefix: tuple[str, ...]) -> tuple[str, ...] | None:
    """Close pending syntax with fixed defaults; no target is read."""
    tokens = prefix
    while len(tokens) <= MAX_TOKENS:
        block, allowed = analyze(tokens)
        if block is not None:
            return tokens
        if not allowed:
            return None
        for token in ("stop", "return", "end", "acc", "set", "eq"):
            if token in allowed:
                tokens = (*tokens, token)
                break
        else:
            tokens = (*tokens, allowed[0])
    return None


def execution_features(prefix: tuple[str, ...], inputs: list[list[int]],
                       targets: list[int]) -> list[list[float]]:
    """Report completed-prefix outputs on observed rows only."""
    tokens = complete(prefix)
    result = []
    for values, target in zip(inputs, targets):
        scale = max(1, abs(target), *(abs(value) for value in values))
        try:
            if tokens is None:
                raise ValueError("completion exceeds syntax budget")
            actual = execute(tokens, values)
            result.append([actual / scale, (target - actual) / scale,
                           float(actual == target), 1.0])
        except ValueError:
            result.append([0.0, 0.0, 0.0, 0.0])
    return result
