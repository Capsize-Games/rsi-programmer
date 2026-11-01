"""Typed prefix grammar for constructing bounded imperative programs."""
from __future__ import annotations

from collections.abc import Generator
from typing import Any

Expr = tuple[Any, ...]
Block = tuple[tuple[Any, ...], ...]
Reader = Generator[tuple[str, ...], str, Any]
LEAVES = ("acc", "zero", "one", "two", "minus_one", "length")
ARITHMETIC = ("add", "sub", "mul", "rem")
COMPARISONS = ("gt", "ge", "lt", "le", "eq", "ne")
TOKENS = ("for", "if", "set", "end", "return", "stop", "value",
          *LEAVES, *ARITHMETIC, *COMPARISONS)
MAX_TOKENS = 32


def _expr(in_loop: bool, depth: int = 2) -> Reader:
    choices = (*LEAVES, *(("value",) if in_loop else ()),
               *(ARITHMETIC if depth else ()))
    token = yield choices
    if token in ARITHMETIC:
        left = yield from _expr(in_loop, depth - 1)
        right = yield from _expr(in_loop, depth - 1)
        return token, left, right
    return (token,)


def _block(in_loop: bool = False, depth: int = 0) -> Reader:
    statements: list[tuple[Any, ...]] = []
    loop_used = False
    while True:
        choices = ["set"]
        if depth < 3:
            choices.append("if")
        if depth == 0 and not loop_used:
            choices.append("for")
        if depth == 0:
            choices.append("return")
        elif statements:
            choices.append("end")
        token = yield tuple(choices)
        if token == "end":
            return tuple(statements)
        if token == "return":
            value = yield from _expr(in_loop)
            yield ("stop",)
            return (*statements, ("return", value))
        if token == "set":
            value = yield from _expr(in_loop)
            statements.append(("set", value))
        elif token == "for":
            body = yield from _block(True, depth + 1)
            statements.append(("for", body))
            loop_used = True
        else:
            comparison = yield COMPARISONS
            left = yield from _expr(in_loop)
            right = yield from _expr(in_loop)
            body = yield from _block(in_loop, depth + 1)
            statements.append(("if", (comparison, left, right), body))


def analyze(tokens: tuple[str, ...]) -> tuple[Block | None, tuple[str, ...]]:
    """Return the complete AST or the next well-typed token choices."""
    if len(tokens) > MAX_TOKENS:
        raise ValueError("statement token bound exceeded")
    reader = _block()
    choices = next(reader)
    for index, token in enumerate(tokens):
        if token not in choices:
            raise ValueError(f"invalid statement token {token!r} at {index}")
        try:
            choices = reader.send(token)
        except StopIteration as finished:
            if index != len(tokens) - 1:
                raise ValueError("tokens after completed program") from None
            return finished.value, ()
    return None, choices if len(tokens) < MAX_TOKENS else ()


def parse(tokens: tuple[str, ...]) -> Block:
    block, _ = analyze(tokens)
    if block is None:
        raise ValueError("incomplete statement program")
    return block
