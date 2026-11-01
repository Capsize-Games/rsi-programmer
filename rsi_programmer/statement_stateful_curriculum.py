"""Bounded grammar-generated teachers with state-dependent predicates."""
from __future__ import annotations

import itertools
import random


def candidates(seed: int = 1237) -> list[tuple[str, ...]]:
    """Sample syntax combinations without reading target programs or names."""
    rng = random.Random(seed)
    conditions = [f"{op} {left} {right}" for op in
                  ("gt", "ge", "lt", "le", "eq", "ne")
                  for left in ("value", "acc", "rem value two",
                               "sub zero value")
                  for right in ("zero", "one", "two", "minus_one", "acc",
                                "length")]
    updates = ["value", "one", "two", "zero", "length", "sub zero value",
               "add value one", "sub value one", "add value two",
               "sub value two", "mul value value", "mul acc value"]
    terms = ["value", "one", "two", "minus_one", "length",
             "sub zero value", "mul value value"] + [
        f"{op} value {constant}" for op in ("add", "sub")
        for constant in ("one", "two")]
    updates += [f"{op} acc {term}" for op in ("add", "sub") for term in terms]
    combinations = list(itertools.product(conditions, updates))
    rng.shuffle(combinations)
    programs = [tuple(f"for if {condition} set {update} end end "
                      "return acc stop".split())
                for condition, update in combinations[:1536]]
    for _ in range(512):
        first, second = rng.choice(conditions), rng.choice(conditions)
        left, right = rng.choice(updates), rng.choice(updates)
        if rng.randrange(2):
            code = (f"for if {first} set {left} end "
                    f"if {second} set {right} end end return acc stop")
        else:
            code = (f"for if {first} if {second} set {left} "
                    "end end end return acc stop")
        programs.append(tuple(code.split()))
    return programs
