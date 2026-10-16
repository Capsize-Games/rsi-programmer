#!/usr/bin/env python3
"""Deterministic typed-task generator with frozen splits.

Grows random Int/Bool/select expression trees inside a size bound,
requires the requested IR features by construction, then samples
disjoint train/held-out rows and freezes them with split hashes.
Same seed always yields byte-identical fixture files; the
curriculum bridge turns PromotionGate reports into family
parameters for one closed failure-to-data loop.
"""
from __future__ import annotations

import random
from typing import Any

from rsi_programmer.probe import CONSTS, OPS, SELECT, Expr
from rsi_programmer.probe import const_terms, enumerate_programs
from rsi_programmer.probe import interpret, lower_to_dag
from rsi_programmer.probe import signature, terminals

ORACLE = "frozen expected outputs; not search-derived"
ROW_LO, ROW_HI = -4, 5
TRAIN_ROWS, HELD_ROWS = 8, 8


def _formula(expr: Expr | str) -> str:
    """Compact human formula for one expression tree."""
    if isinstance(expr, str):
        return expr
    if expr.op == "input":
        return str(expr.left)
    if expr.op == "const":
        return str(expr.left)
    if expr.op == "select":
        assert expr.third is not None
        return "(%s if %s else %s)" % (_formula(expr.right),
                                       _formula(expr.left),
                                       _formula(expr.third))
    sym = {"add": "+", "sub": "-", "mul": "*",
           "lt": "<", "eq": "=="}[expr.op]
    assert isinstance(expr.right, Expr)
    return "(%s %s %s)" % (_formula(expr.left), sym,
                           _formula(expr.right))


def _leaves(expr: Expr) -> list[Expr]:
    """Collect input leaves eligible for constant forcing."""
    if expr.op == "input":
        return [expr]
    found = _leaves(expr.left) if isinstance(expr.left, Expr) \
        else []
    if isinstance(expr.right, Expr):
        found += _leaves(expr.right)
    if isinstance(expr.third, Expr):
        found += _leaves(expr.third)
    return found


def _gen_int(rng: random.Random, budget: int,
             consts: bool) -> tuple[Expr, int, bool]:
    """Random Int tree within budget; reports const use."""
    if budget <= 0 or (budget <= 1 and rng.random() < 0.5):
        if consts and rng.random() < 0.4:
            value = rng.choice(CONSTS)
            return Expr("const", value, None, 0), 0, True
        pick = rng.choice(terminals())
        return Expr(pick.op, pick.left, None, 0), 0, False
    op = rng.choice(sorted(OPS))
    left_size = rng.randrange(budget)
    left, _, cleft = _gen_int(rng, left_size, consts)
    right, _, cright = _gen_int(rng, budget - 1 - left_size,
                                consts)
    size = left.size + right.size + 1
    return Expr(op, left, right, size), size, cleft or cright


def _gen_bool(rng: random.Random, budget: int,
              consts: bool) -> tuple[Expr, int, bool]:
    """Random comparison within budget; reports const use."""
    op = rng.choice(["lt", "eq"])
    left_size = rng.randrange(budget)
    left, _, cleft = _gen_int(rng, left_size, consts)
    right, _, cright = _gen_int(rng, budget - 1 - left_size,
                                consts)
    size = left.size + right.size + 1
    return Expr(op, left, right, size), size, cleft or cright


def _force_const(rng: random.Random, expr: Expr) -> Expr:
    """Swap one random input leaf for a constant in range."""
    leaves = [e for e in _leaves(expr)]
    if not leaves:
        return expr
    pick = rng.choice(leaves)
    value = rng.choice(CONSTS)
    return _swap(expr, pick, Expr("const", value, None, 0))


def _swap(expr: Expr | str, old: Expr, new: Expr) -> Expr | str:
    """Replace one leaf occurrence, rebuilding sizes upward."""
    if expr is old:
        return new
    if not isinstance(expr, Expr) or expr.op in ("input",
                                                 "const"):
        return expr
    left = _swap(expr.left, old, new)  # type: ignore[arg-type]
    right = _swap(expr.right, old, new) \
        if isinstance(expr.right, Expr) else expr.right
    third = _swap(expr.third, old, new) \
        if isinstance(expr.third, Expr) else expr.third
    return Expr(expr.op, left, right, _size_of(expr.op, left,
                                               right, third),
                third if isinstance(third, Expr) else None)


def _size_of(op: str, left: Any, right: Any,
             third: Any) -> int:
    """Node-count size for a rebuilt expression."""
    total = 0
    for part in (left, right, third):
        if isinstance(part, Expr):
            total += part.size + (0 if part.op in ("input",
                                                  "const")
                                  else 1)
    return total + (0 if op in ("input", "const") else 1)


def _has_branch(expr: Expr | str) -> bool:
    """True when a select node occurs anywhere."""
    if not isinstance(expr, Expr):
        return False
    if expr.op == SELECT:
        return True
    parts: list[Any] = [expr.left, expr.right, expr.third]
    return any(_has_branch(p) for p in parts
               if isinstance(p, Expr))


def _rows(rng: random.Random, count: int,
          taken: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Distinct input pairs outside the taken set."""
    out: list[tuple[int, int]] = []
    while len(out) < count:
        pair = (rng.randrange(ROW_LO, ROW_HI),
                rng.randrange(ROW_LO, ROW_HI))
        if pair not in taken:
            taken.add(pair)
            out.append(pair)
    return out


def generate_task(rng: random.Random, task_id: str, bound: int,
                  branch: bool, consts: bool) -> dict[str, Any]:
    """One frozen task requiring the requested IR features."""
    if branch and bound < 4:
        raise ValueError("select needs bound 4 or more")
    for _ in range(200):
        if branch:
            cond, used, cflag = _gen_bool(rng, 1, consts)
            rest = bound - 1 - used
            first_size = rng.randrange(rest)
            first, _, cfirst = _gen_int(rng, first_size, consts)
            second, _, csecond = _gen_int(
                rng, rest - 1 - first_size, consts)
            size = used + first.size + second.size + 1
            expr: Expr = Expr(SELECT, cond, first, size,
                              third=second)
            used_const = cflag or cfirst or csecond
        else:
            expr, size, used_const = _gen_int(rng, bound, consts)
            if consts and not used_const:
                expr = _force_const(rng, expr)
                used_const = True
        graph = lower_to_dag(expr)
        taken: set[tuple[int, int]] = set()
        train = _rows(rng, TRAIN_ROWS, taken)
        held = _rows(rng, HELD_ROWS, taken)
        expect = tuple(interpret(graph, x, y) for x, y in train)
        if len(set(expect)) < 2:
            continue
        if expect == tuple(x for x, _ in train) or \
                expect == tuple(y for _, y in train):
            continue
        held_expect = tuple(interpret(graph, x, y)
                            for x, y in held)
        branched = _has_branch(expr)
        check, _, _ = enumerate_programs(
            train, expect, bound,
            consts=CONSTS if used_const else (), branch=branched)
        if check is None:
            continue
        confirm = lower_to_dag(check)
        if any(interpret(confirm, x, y) != e
               for (x, y), e in zip(held, held_expect)):
            continue
        feats = sorted((["branch"] if branched else []) +
                        (["const"] if used_const else []))
        return {"held_out": [[x, y, e] for (x, y), e in
                             zip(held, held_expect)],
                "id": task_id, "max_tree_ops": bound,
                "oracle": ORACLE, "split_sha256": "",
                "task": "f(x, y) = %s" % _formula(expr),
                "train": [[x, y, e] for (x, y), e in
                           zip(train, expect)],
                "features": feats}
    raise ValueError("could not shape a non-trivial task")
