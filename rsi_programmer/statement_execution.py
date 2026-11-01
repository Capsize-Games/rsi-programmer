"""Bounded task-engine execution of typed imperative statements."""
from __future__ import annotations

from rsi_programmer.statement_syntax import Block, Expr, parse


def expression(expr: Expr, values: dict[str, int]) -> int:
    op = expr[0]
    if len(expr) == 1:
        return values[op]
    left, right = expression(expr[1], values), expression(expr[2], values)
    if op == "add":
        result = left + right
    elif op == "sub":
        result = left - right
    elif op == "mul":
        result = left * right
    else:
        if not right:
            raise ValueError("remainder by zero")
        # Match truncation-toward-zero signed remainder, including CURLEE.
        result = abs(left) % abs(right)
        result = -result if left < 0 else result
    if not -(2 ** 63) <= result < 2 ** 63:
        raise ValueError("integer overflow")
    return result


def predicate(expr: Expr, values: dict[str, int]) -> bool:
    op = expr[0]
    a, b = expression(expr[1], values), expression(expr[2], values)
    return {"gt": a > b, "ge": a >= b, "lt": a < b, "le": a <= b,
            "eq": a == b, "ne": a != b}[op]


def execute(tokens: tuple[str, ...], items: list[int]) -> int:
    """Interpret complete syntax without a target-language emitter."""
    if len(items) > 256 or any(type(item) is not int for item in items):
        raise ValueError("at most 256 integer readings required")
    if any(not -(2 ** 63) <= item < 2 ** 63 for item in items):
        raise ValueError("signed 64-bit integer readings required")
    values = {"acc": 0, "zero": 0, "one": 1, "two": 2,
              "minus_one": -1, "length": len(items)}
    fuel = 4096

    def run(block: Block) -> int | None:
        nonlocal fuel
        for statement in block:
            fuel -= 1
            if fuel < 0:
                raise ValueError("statement fuel exhausted")
            op = statement[0]
            if op == "set":
                values["acc"] = expression(statement[1], values)
            elif op == "if":
                if predicate(statement[1], values):
                    run(statement[2])
            elif op == "for":
                for value in items:
                    values["value"] = value
                    run(statement[1])
            else:
                return expression(statement[1], values)
        return None

    result = run(parse(tokens))
    if result is None:
        raise ValueError("integer return required")
    return result
