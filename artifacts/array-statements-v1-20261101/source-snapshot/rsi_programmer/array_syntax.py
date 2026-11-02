"""Typed construction of one-loop, list-producing statement programs."""
from __future__ import annotations

from collections.abc import Generator
from typing import Any

from rsi_programmer.statement_syntax import ARITHMETIC, COMPARISONS

Expr = tuple[Any, ...]
Block = tuple[tuple[Any, ...], ...]
Reader = Generator[tuple[str, ...], str, Any]
LEAVES = ("value", "zero", "one", "two", "minus_one", "length", "outlen")
TOKENS = ("for", "if", "append", "end", "return", "output", "stop",
          *LEAVES, *ARITHMETIC, *COMPARISONS)
MAX_TOKENS = 40


def _expr(depth: int = 1) -> Reader:
    choices = (*LEAVES, *(ARITHMETIC if depth else ()))
    token = yield choices
    if token in ARITHMETIC:
        left = yield from _expr(depth - 1)
        right = yield from _expr(depth - 1)
        return token, left, right
    return (token,)


def _body(writes: list[int], depth: int = 0) -> Reader:
    statements: list[tuple[Any, ...]] = []
    while True:
        choices = []
        if writes[0] < 2:
            choices.append("append")
            if depth < 2:
                choices.append("if")
        if statements:
            choices.append("end")
        token = yield tuple(choices)
        if token == "end":
            return tuple(statements)
        if token == "append":
            value = yield from _expr()
            writes[0] += 1
            statements.append(("append", value))
        else:
            comparison = yield COMPARISONS
            left = yield from _expr()
            right = yield from _expr()
            body = yield from _body(writes, depth + 1)
            statements.append(("if", (comparison, left, right), body))


def _program() -> Reader:
    yield ("for",)
    body = yield from _body([0])
    yield ("return",)
    yield ("output",)
    yield ("stop",)
    return body


def analyze(tokens: tuple[str, ...]) -> tuple[Block | None, tuple[str, ...]]:
    """Constrain control flow, expression types and at most two writes."""
    if len(tokens) > MAX_TOKENS:
        raise ValueError("array statement token bound exceeded")
    reader = _program()
    choices = next(reader)
    for index, token in enumerate(tokens):
        if token not in choices:
            raise ValueError(f"invalid array token {token!r} at {index}")
        try:
            choices = reader.send(token)
        except StopIteration as finished:
            if index != len(tokens) - 1:
                raise ValueError("tokens after completed array program")
            return finished.value, ()
    return None, choices if len(tokens) < MAX_TOKENS else ()


def parse(tokens: tuple[str, ...]) -> Block:
    block, _ = analyze(tokens)
    if block is None:
        raise ValueError("incomplete array program")
    return block
