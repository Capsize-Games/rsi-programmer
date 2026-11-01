"""Generate imperative teachers from syntax and training execution."""
from __future__ import annotations

import itertools
import random

from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_syntax import MAX_TOKENS, parse


def oracle(name: str, values: list[int]) -> int:
    if name == "positive_total":
        return sum(value for value in values if value > 0)
    if name == "negative_count":
        return sum(1 for value in values if value < 0)
    if name == "even_total":
        return sum(value for value in values if value % 2 == 0)
    if name == "positive_square_total":
        return sum(value * value for value in values if value > 0)
    if name == "odd_count":
        return sum(1 for value in values if value % 2 != 0)
    if name == "above_two_count":
        return sum(1 for value in values if value > 2)
    from rsi_programmer.stateful_tasks import oracle as stateful_oracle
    return stateful_oracle(name, values)


def candidates() -> list[tuple[str, ...]]:
    """Generate witnesses; no task identifier chooses a whole program."""
    result = []
    for value in ("zero", "one", "two", "minus_one", "length",
                  "add length one", "sub length one", "mul length two"):
        result.append(tuple(f"return {value} stop".split()))
    updates = ("value", "one", "two", "zero", "length", "add acc value",
               "sub acc value", "add acc one", "sub acc one", "mul acc value",
               "add acc mul value value", "sub acc mul value value",
               "add acc sub zero value", "add acc length")
    conditions = [f"{comparison} {left} {right}" for comparison in
                  ("gt", "ge", "lt", "le", "eq", "ne")
                  for left in ("value", "rem value two")
                  for right in ("zero", "one", "two", "minus_one")]
    for update in updates:
        result.append(tuple(f"for set {update} end return acc stop".split()))
        for condition in conditions:
            code = (f"for if {condition} set {update} end end "
                    "return acc stop")
            result.append(tuple(code.split()))
    for first, second in itertools.product(updates, repeat=2):
        code = (f"for if gt value zero set {first} end "
                f"if lt value zero set {second} end end return acc stop")
        tokens = tuple(code.split())
        if len(tokens) <= MAX_TOKENS:
            result.append(tokens)
    return result


def teachers(excluded: list[str], seed: int = 0, stateful: bool = False,
             return_variants: bool = False
             ) -> tuple[list[tuple[str, ...]], list[list[int]]]:
    """Remove complete requested functions on a fixed finite probe suite."""
    rng = random.Random(seed)
    probes = [[], [0], [-1, 0, 1], [2, -2, 3, -3, 0, 2]] + [
        [rng.randrange(-5, 6) for _ in range(rng.randrange(3, 11))]
        for _ in range(20)]
    seen = {tuple(oracle(name, values) for values in probes)
            for name in excluded}
    source = candidates()
    if stateful:
        from rsi_programmer.statement_stateful_curriculum import (
            candidates as stateful_candidates)
        source += stateful_candidates()
    if return_variants:
        # Retain a component in other functions when its complete function
        # is excluded. The independent engine labels and filters all rows.
        source += [(*program[:-2], "add", "acc", "one", "stop")
                   for program in source if program[-3:] == (
                       "return", "acc", "stop")]
        from rsi_programmer.statement_return_curriculum import (
            candidates as return_candidates)
        source += return_candidates()
    programs = []
    for program in source:
        try:
            parse(program)
            signature = tuple(execute(program, values) for values in probes)
        except ValueError:
            continue
        if signature not in seen:
            seen.add(signature)
            programs.append(program)
    return programs, probes
