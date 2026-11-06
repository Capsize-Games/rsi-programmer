"""Extended separation probes for frozen transfer bank v2."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
import random

from rsi_programmer.statement_curriculum import oracle, teachers
from rsi_programmer.statement_execution import execute

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "fixtures/frozen_branch_transfer_v1"
EXTRA_PROBE_SEED = 613151
EXTRA_PROBE_COUNT = 48
BOUNDARY = [[], [0], [1], [-1], [100] * 256, [-100] * 256]
Sig = tuple[int | str, ...]


def arrays(seed: int, count: int, low: int, high: int, size: int
           ) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(-size, size + 1)
             for _ in range(rng.randrange(low, high + 1))]
            for _ in range(count)]


def training_names() -> list[str]:
    tasks = json.loads((ROOT / "fixtures/statement_tasks_v1.json"
                        ).read_text())
    tasks["tasks"] += json.loads(
        (ROOT / "fixtures/stateful_tasks_v1.json").read_text())["tasks"]
    return [task["id"] for task in tasks["tasks"]]


def v1_references() -> list[tuple[str, ...]]:
    bank = json.loads((V1 / "tasks.json").read_text())
    return [tuple(task["reference"]) for task in bank["tasks"]]


def v1_inputs() -> set[tuple[int, ...]]:
    bank = json.loads((V1 / "tasks.json").read_text())
    seen = {tuple(v) for v in bank["query_inputs"]}
    for task in bank["tasks"]:
        seen.update(tuple(v) for v, _ in task["observed"])
    held = json.loads((V1 / "held.json").read_text())
    return seen | {tuple(v) for v in held}


def canonical_probes(names: list[str]) -> list[list[int]]:
    _, probes = teachers(names[:6])
    assert len(probes) == 24
    return probes


def extended_probes(canonical: list[list[int]]) -> list[list[int]]:
    v1_held = json.loads((V1 / "held.json").read_text())
    extra = arrays(EXTRA_PROBE_SEED, EXTRA_PROBE_COUNT, 0, 256, 100)
    seen: set[tuple[int, ...]] = set()
    unique: list[list[int]] = []
    for values in canonical + v1_held + extra:
        if tuple(values) not in seen:
            seen.add(tuple(values))
            unique.append(values)
    return unique


def safe_run(apply: Callable[[list[int]], int],
             values: list[int]) -> int | str:
    try:
        return apply(values)
    except ValueError:
        return "ERR"


def program_signature(program: tuple[str, ...],
                      probes: list[list[int]]) -> Sig:
    return tuple(safe_run(lambda v: execute(program, v), values)
                 for values in probes)


def oracle_signature(name: str, probes: list[list[int]]
                     ) -> Sig:
    return tuple(safe_run(lambda v: oracle(name, v), values)
                 for values in probes)


def build_exclusion(names: list[str], probes: list[list[int]]
                    ) -> tuple[set[Sig], int]:
    training, _ = teachers(names, stateful=True, return_variants=True)
    excluded = {program_signature(p, probes) for p in training}
    excluded.update(oracle_signature(n, probes) for n in names)
    excluded.update(program_signature(p, probes)
                    for p in v1_references())
    return excluded, len(training)
