"""Small typed library for useful integer-list program synthesis."""
from __future__ import annotations

from itertools import accumulate

OPS = ("positive", "nonnegative", "even", "odd", "absolute", "negate",
       "double", "increment", "unique", "ascending", "descending",
       "reverse", "prefix_sum")


def apply_op(name: str, values: list[int]) -> list[int]:
    """Execute one library operation; no complete task is selected here."""
    if name == "positive":
        return [value for value in values if value > 0]
    if name == "nonnegative":
        return [value for value in values if value >= 0]
    if name == "even":
        return [value for value in values if value % 2 == 0]
    if name == "odd":
        return [value for value in values if value % 2 != 0]
    if name == "absolute":
        return [abs(value) for value in values]
    if name == "negate":
        return [-value for value in values]
    if name == "double":
        return [2 * value for value in values]
    if name == "increment":
        return [value + 1 for value in values]
    if name == "unique":
        return list(dict.fromkeys(values))
    if name == "ascending":
        return sorted(values)
    if name == "descending":
        return sorted(values, reverse=True)
    if name == "reverse":
        return list(reversed(values))
    if name == "prefix_sum":
        return list(accumulate(values))
    raise ValueError("unknown utility operation")


def execute(program: tuple[str, ...], values: list[int]) -> list[int]:
    result = values[:]
    for op in program:
        result = apply_op(op, result)
    return result
