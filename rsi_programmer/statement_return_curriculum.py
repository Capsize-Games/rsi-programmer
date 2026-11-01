"""Expose leaf-level loop components inside distinct return functions."""
from __future__ import annotations

import itertools

from rsi_programmer.statement_syntax import COMPARISONS, LEAVES


def candidates() -> list[tuple[str, ...]]:
    """Enumerate a generic grammar grid, without task identifiers or code."""
    leaves = (*LEAVES, "value")
    returns = ("add acc one", "sub acc one", "mul acc two", "sub zero acc",
               "add acc length", "sub acc length")
    return [tuple(f"for if {op} {left} {right} set {update} end end "
                  f"return {result} stop".split())
            for op, left, right, update, result in itertools.product(
                COMPARISONS, leaves, leaves, leaves, returns)]
