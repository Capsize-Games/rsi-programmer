"""Mine shared-input branch templates from qualified neural programs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rsi_programmer.statement_fragment_library import qualify
from rsi_programmer.statement_fragments import expression_tokens
from rsi_programmer.statement_syntax import ARITHMETIC, Expr, LEAVES, parse


def _abstract(expr: Expr) -> Expr:
    if len(expr) == 1:
        return ("$arg",) if expr[0] == "value" else tuple(expr)
    return expr[0], _abstract(expr[1]), _abstract(expr[2])


def qualify_branches(root: Path) -> dict[str, Any]:
    evidence = qualify(root)
    raw = (root / "report.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != evidence["report_sha256"]:
        raise ValueError("donor report changed after qualification")
    branches = []
    for row in json.loads(raw)["rows"]:
        if row["id"] not in evidence["donors"]:
            continue
        tokens = tuple(row["chosen"]["tokens"])
        tree = parse(tokens)
        if (len(tree) != 2 or tree[0][0] != "for"
                or tree[1] != ("return", ("acc",)) or len(tree[0][1]) != 1):
            continue
        node = tree[0][1][0]
        if (node[0] != "if" or len(node[2]) != 1
                or node[2][0][0] != "set"):
            continue
        condition, update = node[1], node[2][0][1]
        body = ("if", *expression_tokens(condition), "set",
                *expression_tokens(update), "end")
        branches.append({"donor": row["id"], "program": tokens,
                         "body": body, "condition": _abstract(condition),
                         "update": _abstract(update)})
    return {"qualification": evidence, "branches": branches,
            "scope": "Classically mined shared Int-input branch templates. "
                     "Accumulator add/sub variants are supplied grammar "
                     "choices, not learned abstraction discovery."}


def pattern(first: dict[str, Any], second: dict[str, Any],
            combine: str | None = None) -> tuple[str, ...]:
    update = second["update"]
    if combine is not None:
        if update[0] not in ("add", "sub") or tuple(update[1]) != ("acc",):
            raise ValueError("combine variant requires an accumulator update")
        update = (combine, *update[1:])
    return ("for", *first["body"], "if",
            *expression_tokens(second["condition"]), "set",
            *expression_tokens(update), "end", "end", "return", "acc", "stop")


def instantiate(template: tuple[str, ...], arg: tuple[str, ...]
                ) -> tuple[str, ...]:
    pending = 1
    for token in arg:
        if pending == 0:
            raise ValueError("argument contains more than one expression")
        if token in ARITHMETIC:
            pending += 1
        elif token in (*LEAVES, "value"):
            pending -= 1
        else:
            raise ValueError("shared argument must be an Int expression")
    if pending:
        raise ValueError("incomplete shared Int argument")
    tokens = tuple(part for token in template
                   for part in (arg if token == "$arg" else (token,)))
    parse(tokens)
    return tokens
