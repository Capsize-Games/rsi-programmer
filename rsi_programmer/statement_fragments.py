"""Mine reusable condition contexts from independently checked programs."""
from __future__ import annotations

import hashlib
from typing import Any

from rsi_programmer.statement_syntax import Block, Expr, parse


def expression_tokens(expr: Expr) -> tuple[str, ...]:
    if len(expr) == 1:
        return (expr[0],)
    return (expr[0], *expression_tokens(expr[1]), *expression_tokens(expr[2]))


def _conditions(block: Block, offset: int
                ) -> tuple[list[tuple[int, int]], int]:
    spans = []
    for node in block:
        if node[0] == "if":
            start = offset + 1
            end = start + len(expression_tokens(node[1]))
            spans.append((start, end))
            nested, offset = _conditions(node[2], end)
            spans.extend(nested)
            offset += 1
        elif node[0] == "for":
            nested, offset = _conditions(node[1], offset + 1)
            spans.extend(nested)
            offset += 1
        else:
            offset += 1 + len(expression_tokens(node[1]))
            if node[0] == "return":
                offset += 1
    return spans, offset


def mine(donors: list[tuple[str, tuple[str, ...]]]) -> list[dict[str, Any]]:
    """Abstract each condition into a Bool hole; merge identical contexts.

    The caller must qualify donors independently before promotion. This
    function sees code and provenance only, never donor test rows or labels.
    """
    contexts: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = {}
    for name, tokens in sorted(donors):
        spans, end = _conditions(parse(tokens), 0)
        if end != len(tokens):
            raise ValueError("condition offsets do not match syntax")
        for start, stop in spans:
            key = (tokens[:start], tokens[stop:])
            contexts.setdefault(key, []).append(name)
    return [{"id": hashlib.sha256(repr(key).encode()).hexdigest()[:16],
             "kind": "Bool condition", "prefix": key[0], "suffix": key[1],
             "donors": sorted(set(names))}
            for key, names in sorted(contexts.items())]
