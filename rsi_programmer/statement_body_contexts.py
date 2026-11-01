"""Mine loop-body holes from independently qualified donor programs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rsi_programmer.statement_fragment_library import qualify
from rsi_programmer.statement_fragments import expression_tokens
from rsi_programmer.statement_syntax import Block, parse


def _loops(block: Block, offset: int
           ) -> tuple[list[tuple[int, int]], int]:
    spans = []
    for node in block:
        if node[0] == "for":
            start = offset + 1
            nested, end = _loops(node[1], start)
            spans.extend([(start, end), *nested])
            offset = end + 1
        elif node[0] == "if":
            start = offset + 1 + len(expression_tokens(node[1]))
            nested, end = _loops(node[2], start)
            spans.extend(nested)
            offset = end + 1
        else:
            offset += 1 + len(expression_tokens(node[1]))
            if node[0] == "return":
                offset += 1
    return spans, offset


def mine(donors: list[tuple[str, tuple[str, ...]]]) -> list[dict[str, Any]]:
    """Abstract entire loop bodies, keeping outer scope and provenance."""
    contexts: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = {}
    for name, tokens in sorted(donors):
        spans, end = _loops(parse(tokens), 0)
        if end != len(tokens):
            raise ValueError("loop-body offsets do not match syntax")
        for start, stop in spans:
            key = (tokens[:start], tokens[stop:])
            contexts.setdefault(key, []).append(name)
    return [{"id": hashlib.sha256(repr(key).encode()).hexdigest()[:16],
             "kind": "nonempty loop body", "prefix": key[0],
             "suffix": key[1], "donors": sorted(set(names))}
            for key, names in sorted(contexts.items())]


def qualify_bodies(root: Path) -> dict[str, Any]:
    library = qualify(root)
    raw = (root / "report.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != library["report_sha256"]:
        raise ValueError("donor report changed after qualification")
    donors = [(row["id"], tuple(row["chosen"]["tokens"]))
              for row in json.loads(raw)["rows"]
              if row["id"] in library["donors"]]
    return {**library, "condition_fragments": library["fragments"],
            "fragments": mine(donors)}
