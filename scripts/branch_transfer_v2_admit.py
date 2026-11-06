"""Candidate admission for frozen transfer bank v2."""
from __future__ import annotations

import itertools
import random
from typing import Any

from scripts.branch_transfer_v2_probes import Sig
from scripts.branch_transfer_v2_probes import program_signature
from rsi_programmer.statement_branch_library import instantiate, pattern
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_syntax import ARITHMETIC, LEAVES

PER_GROUP_QUOTA = 6
SHUFFLE_SEED = 235079
Prog = tuple[str, ...]
Probes = list[list[int]]
Bank = list[dict[str, Any]]
Rej = dict[str, int]
Candidate = tuple[Prog, Prog]


def arguments() -> list[Prog]:
    leaves = (*LEAVES, "value")
    args = [(leaf,) for leaf in leaves]
    args += [(op, left, right) for op, left, right in itertools.product(
        ARITHMETIC, leaves, leaves)]
    return args


def _candidates(first: Any, branches: list[Any], args: list[Prog],
                salt: int) -> list[Candidate]:
    found: list[Candidate] = []
    for second in branches:
        update = second["update"]
        variants = ("add", "sub") if (update[0] in ("add", "sub")
                    and tuple(update[1]) == ("acc",)) else (None,)
        for variant, arg in itertools.product(variants, args):
            found.append((pattern(first, second, variant), arg))
    random.Random(SHUFFLE_SEED + salt).shuffle(found)
    return found


def _reject(rejected: Rej, key: str) -> bool:
    rejected[key] += 1
    return False


def _clean_expected(program: Prog, observed: Probes, queries: Probes,
                    held: Probes) -> list[int] | None:
    try:
        expected = [execute(program, v) for v in observed]
        for values in queries + held:
            execute(program, values)
    except ValueError:
        return None
    return expected


def _bank_row(index: int, donor: str, program: Prog, observed: Probes,
              expected: list[int]) -> dict[str, Any]:
    return {"id": f"transfer2_{index:03}", "generation_group": donor,
            "reference": program,
            "observed": list(zip(observed, expected))}


def _admit(program: Prog, excluded: set[Sig], probes: Probes,
           observed: Probes, queries: Probes, held: Probes,
           exact: set[Prog], bank: Bank, rejected: Rej,
           donor: str) -> bool:
    if program in exact:
        return _reject(rejected, "exact_repeat")
    signature = program_signature(program, probes)
    if signature in excluded:
        return _reject(rejected, "known_function")
    expected = _clean_expected(program, observed, queries, held)
    if expected is None:
        return _reject(rejected, "syntax_or_runtime")
    excluded.add(signature)
    exact.add(program)
    bank.append(_bank_row(len(bank), donor, program, observed,
                          expected))
    return True


def _try_candidate(template: Prog, arg: Prog, excluded: set[Sig],
                   probes: Probes, observed: Probes, queries: Probes,
                   held: Probes, exact: set[Prog], bank: Bank,
                   rejected: Rej, donor: str) -> bool:
    try:
        program = instantiate(template, arg)
    except ValueError:
        return _reject(rejected, "syntax_or_runtime")
    return _admit(program, excluded, probes, observed, queries, held,
                  exact, bank, rejected, donor)


def fill_group(first: Any, branches: list[Any], args: list[Prog],
               excluded: set[Sig], probes: Probes, observed: Probes,
               queries: Probes, held: Probes, exact: set[Prog],
               bank: Bank, rejected: Rej) -> None:
    accepted = 0
    options = _candidates(first, branches, args, len(bank))
    for template, arg in options:
        if _try_candidate(template, arg, excluded, probes, observed,
                          queries, held, exact, bank, rejected,
                          first["donor"]):
            accepted += 1
            if accepted == PER_GROUP_QUOTA:
                return
    raise ValueError("cannot fill a donor group without overlap")
