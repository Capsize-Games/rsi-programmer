"""Classical breadth-first filling of the same reusable Bool contexts."""
from __future__ import annotations

from collections.abc import Iterator
import time
from typing import Any

from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_syntax import ARITHMETIC, COMPARISONS, LEAVES
from rsi_programmer.statement_syntax import MAX_TOKENS, parse


def _expressions(size: int, depth: int) -> Iterator[tuple[str, ...]]:
    if size == 1:
        yield from ((leaf,) for leaf in (*LEAVES, "value"))
    elif depth:
        for left_size in range(1, size - 1, 2):
            right_size = size - left_size - 1
            for op in ARITHMETIC:
                for left in _expressions(left_size, depth - 1):
                    for right in _expressions(right_size, depth - 1):
                        yield op, *left, *right


def synthesize(examples: list[tuple[list[int], int]],
               fragments: list[dict[str, Any]], budget_seconds: float = 0.5
               ) -> dict[str, Any]:
    """Interleave contexts at each expression size, without neural scores."""
    if budget_seconds <= 0:
        raise ValueError("positive search time budget required")
    started = time.monotonic()
    visited = 0
    if not fragments:
        return {"tokens": None, "visited": 0,
                "wall_seconds": time.monotonic() - started,
                "budget_exhausted": False}
    for size in range(3, 16, 2):
        for left_size in range(1, size - 1, 2):
            right_size = size - left_size - 1
            for op in COMPARISONS:
                for left in _expressions(left_size, 2):
                    for right in _expressions(right_size, 2):
                        for fragment in fragments:
                            if time.monotonic() - started >= budget_seconds:
                                return {"tokens": None, "visited": visited,
                                        "wall_seconds": time.monotonic()
                                        - started, "budget_exhausted": True}
                            full = (*fragment["prefix"], op, *left, *right,
                                    *fragment["suffix"])
                            if len(full) > MAX_TOKENS:
                                continue
                            visited += 1
                            try:
                                parse(full)
                                correct = all(execute(full, values) == expected
                                              for values, expected in examples)
                            except ValueError:
                                correct = False
                            if correct:
                                return {"tokens": full,
                                        "fragment": fragment["id"],
                                        "donors": fragment["donors"],
                                        "visited": visited,
                                        "wall_seconds": time.monotonic()
                                        - started, "budget_exhausted": False}
    return {"tokens": None, "visited": visited,
            "wall_seconds": time.monotonic() - started,
            "budget_exhausted": False}
