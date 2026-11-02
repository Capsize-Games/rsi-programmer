"""Independent bounded interpretation of list-producing statements."""
from __future__ import annotations

from rsi_programmer.array_syntax import Block, parse
from rsi_programmer.statement_execution import expression, predicate


def execute(tokens: tuple[str, ...], items: list[int]) -> list[int]:
    if len(items) > 256 or any(type(value) is not int for value in items):
        raise ValueError("at most 256 integer readings required")
    if any(not -(2 ** 63) <= value < 2 ** 63 for value in items):
        raise ValueError("signed 64-bit integer readings required")
    values = {"zero": 0, "one": 1, "two": 2, "minus_one": -1,
              "length": len(items), "outlen": 0}
    output: list[int] = []
    fuel = 4096

    def run(block: Block) -> None:
        nonlocal fuel
        for statement in block:
            fuel -= 1
            if fuel < 0:
                raise ValueError("array statement fuel exhausted")
            if statement[0] == "append":
                output.append(expression(statement[1], values))
                values["outlen"] = len(output)
                if len(output) > 512:
                    raise ValueError("array output capacity exceeded")
            elif predicate(statement[1], values):
                run(statement[2])

    body = parse(tokens)
    for value in items:
        values["value"] = value
        run(body)
    return output
