"""Freeze unseen branch functions without consulting any learner scores."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random

from rsi_programmer.statement_branch_library import instantiate, pattern
from rsi_programmer.statement_branch_library import qualify_branches
from rsi_programmer.statement_curriculum import oracle, teachers
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_syntax import ARITHMETIC, LEAVES

ROOT = Path(__file__).resolve().parents[1]


def _arrays(seed: int, count: int, low: int, high: int, size: int
            ) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(-size, size + 1)
             for _ in range(rng.randrange(low, high + 1))]
            for _ in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    tasks = json.loads((ROOT / "fixtures/statement_tasks_v1.json").read_text())
    tasks["tasks"] += json.loads(
        (ROOT / "fixtures/stateful_tasks_v1.json").read_text())["tasks"]
    names = [task["id"] for task in tasks["tasks"]]
    training, probes = teachers(names, stateful=True, return_variants=True)
    excluded = {tuple(execute(program, values) for values in probes)
                for program in training}
    excluded.update(tuple(oracle(name, values) for values in probes)
                    for name in names)
    base = ROOT / "artifacts/statement-context-repair-v1-20261101"
    extra = ROOT / "artifacts/positive-counter-v1-20261101"
    library = qualify_branches(base)
    library["branches"] += qualify_branches(extra)["branches"]
    observed = _arrays(613031, 3, 3, 12, 8)
    queries = _arrays(613037, 24, 3, 100, 100)
    held = [[], [0], [1], [-1], [100] * 256, [-100] * 256]
    held += _arrays(613043, 14, 13, 100, 100)
    if {tuple(v) for v in held} & {tuple(v) for v in observed + queries}:
        raise ValueError("visible and held inputs overlap")
    leaves = (*LEAVES, "value")
    arguments = [(leaf,) for leaf in leaves] + [
        (op, left, right) for op, left, right in itertools.product(
            ARITHMETIC, leaves, leaves)]
    rng = random.Random(135079)
    bank = []
    rejected = {"syntax_or_runtime": 0, "known_function": 0}
    for first in library["branches"]:
        candidates = []
        for second in library["branches"]:
            update = second["update"]
            variants = ("add", "sub") if (update[0] in ("add", "sub")
                        and tuple(update[1]) == ("acc",)) else (None,)
            for variant, arg in itertools.product(variants, arguments):
                candidates.append((pattern(first, second, variant), arg))
        rng.shuffle(candidates)
        accepted = 0
        for template, arg in candidates:
            try:
                program = instantiate(template, arg)
                signature = tuple(execute(program, v) for v in probes)
                if signature in excluded:
                    rejected["known_function"] += 1
                    continue
                expected = [execute(program, v) for v in observed]
                # Dataset admission only: no solver is called here.
                for values in queries + held:
                    execute(program, values)
            except ValueError:
                rejected["syntax_or_runtime"] += 1
                continue
            excluded.add(signature)
            bank.append({"id": f"transfer_{len(bank):03}",
                         "generation_group": first["donor"],
                         "reference": program,
                         "observed": list(zip(observed, expected))})
            accepted += 1
            if accepted == 2:
                break
        if accepted != 2:
            raise ValueError("cannot fill a donor group without overlap")
    data = {"scope": "Frozen unseen-function transfer within the supplied "
                     "two-branch grammar; not unrestricted programming.",
            "tasks": bank, "query_inputs": queries,
            "exclusion_scope": "Different from 2647 teacher functions, "
                               "eleven development functions and every "
                               "earlier selection on 24 fixed probes.",
            "training_functions": len(training), "rejected": rejected}
    protocol = {
        "source_checkpoint": "582830f7390b0149a69acf20da0a092799676c55",
        "frozen_before_solver_use": True, "tasks": len(bank),
        "methods": ["neural", "classical", "untrained"],
        "request_cpu_budget_seconds": 1.0,
        "budget_overrun_tolerance_seconds": 0.05,
        "max_added_examples": 3, "branch_width": 8, "first_limit": 3,
        "held_per_task": len(held),
        "cost_policy": "Measure setup, fresh fingerprint-reproducing "
                       "neural fits, search, visible feedback and actual "
                       "backend checks. Shared qualified library is an "
                       "existing input; historical donor creation costs "
                       "remain outside this controller comparison.",
        "actual_fit_usage_volume": len(bank),
        "hypothetical_fit_usage_volume": 1000,
        "hypothetical_scope": "Amortization scenario, not additional "
                              "quality evidence or the end-to-end gate.",
        "held_access": "Workers load tasks and visible queries only; "
                       "held input file is loaded after all choices end.",
        "generator_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()}
    for name, value in (("tasks.json", data), ("held.json", held),
                        ("protocol.json", protocol)):
        (root / name).write_text(json.dumps(value, indent=2) + "\n")
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("tasks.json", "held.json", "protocol.json")}
    (root / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({"tasks": len(bank), "teacher_functions": len(training),
                      "hashes": hashes}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
