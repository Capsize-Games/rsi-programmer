"""Test visible-query repair after verified donor growth, with no refit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import torch

from rsi_programmer.benchmark_policy import state_fingerprint
from rsi_programmer.statement_branch_library import qualify_branches
from rsi_programmer.statement_branch_search import synthesize as compose
from rsi_programmer.statement_curriculum import oracle
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_fragment_classical import synthesize as classical
from rsi_programmer.statement_fragment_library import qualify
from rsi_programmer.statement_fragment_search import synthesize
from rsi_programmer.statement_policy import StatementPolicy
from rsi_programmer.statement_repair import refine
from scripts.measure_context_growth import _model

ROOT = Path(__file__).resolve().parents[1]
Pairs = list[tuple[list[int], int]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch-donor", type=Path, action="append",
                        default=[])
    parser.add_argument("--held-seed", type=int, default=461047)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    old_root = ROOT / "artifacts/statement-programs-v1-execution-20261101"
    new_root = ROOT / "artifacts/statement-return-v1-20261101"
    old_tasks = json.loads(
        (ROOT / "fixtures/statement_tasks_v1.json").read_text())["tasks"]
    new_tasks = json.loads(
        (ROOT / "fixtures/stateful_tasks_v1.json").read_text())["tasks"]
    tasks = [*old_tasks, *new_tasks]
    pool = json.loads((ROOT / "artifacts/stateful-queries-v1-20261101" /
                       "protocol.json").read_text())["query_inputs"]
    protocol = {
        "scope": "Development visible-query repair after qualified donor "
                 "growth. Fixed weights; no held-row feedback.",
        "methods": ["neural", "untrained", "classical", "old_library"],
        "query_inputs": pool, "max_added_examples": 3,
        "search_budget_seconds": 1.0, "held_seed": args.held_seed,
        "held_per_task": 40,
        "extra_branch_donors": [str(path) for path in args.branch_donor],
        "composition_budget_seconds": 5.0 if args.branch_donor else None,
        "primary_backend_method": "neural",
        "cost_scope": "Neural route can use two searches per repair call; "
                      "all methods get the same query limit. Component "
                      "comparison, not matched total cost.",
        "source_hashes": {name: hashlib.sha256(
            (ROOT / name).read_bytes()).hexdigest() for name in (
                "scripts/measure_context_repair.py",
                "scripts/measure_context_growth.py",
                "rsi_programmer/statement_repair.py",
                "rsi_programmer/statement_fragment_search.py",
                "rsi_programmer/statement_fragment_classical.py",
                "rsi_programmer/statement_fragment_library.py",
                "rsi_programmer/statement_branch_library.py",
                "rsi_programmer/statement_branch_search.py")}}
    if args.branch_donor:
        protocol["cost_scope"] = (
            "Neural route can use two condition searches and one composition "
            "search per repair call; controls use one condition search and "
            "one composition search. Same query limit, differing work. "
            "Not matched total cost.")
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    old, new = qualify(old_root), qualify(new_root)
    branches = []
    branch_proofs = []
    if args.branch_donor:
        base = ROOT / "artifacts/statement-context-repair-v1-20261101"
        for donor_root in (base, *args.branch_donor):
            library = qualify_branches(donor_root.resolve())
            branches.extend(library["branches"])
            branch_proofs.append(library["qualification"])
    qualified_names = set(new["donors"])
    for proof in branch_proofs:
        qualified_names.update(proof["donors"])
    torch.set_num_threads(1)
    previous = _model("statement-repair-v1.pt", old_root)
    current = _model("statement-return-v1.pt", new_root)
    untrained = StatementPolicy(seed=0, execution_context=True)
    untrained.eval()
    initial = state_fingerprint(untrained.state_dict(), {"seed": 0})

    def solve(method: str, examples: Pairs) -> dict[str, Any]:
        if method == "classical":
            first = classical(examples, new["fragments"], 1.0)
            if first["tokens"] is not None or not branches:
                return first
            second = compose(None, examples, branches)
            return {**second, "route": [first, second]}
        if method == "untrained":
            first = synthesize(untrained, examples, new["fragments"],
                               budget_seconds=1.0)
            if first["tokens"] is not None or not branches:
                return first
            second = compose(untrained, examples, branches)
            return {**second, "route": [first, second]}
        if method == "old_library":
            return synthesize(current[0], examples, old["fragments"],
                              budget_seconds=1.0)
        first = synthesize(previous[0], examples, old["fragments"],
                           budget_seconds=1.0)
        if first["tokens"] is not None:
            return {**first, "selected_model": "previous", "route": [first]}
        second = synthesize(current[0], examples, new["fragments"],
                            budget_seconds=1.0)
        if second["tokens"] is None and branches:
            third = compose(current[0], examples, branches)
            return {**third, "selected_model": "current",
                    "route": [first, second, third]}
        return {**second, "selected_model": "current",
                "route": [first, second]}

    results = []
    for method in protocol["methods"]:
        rows = []
        for task in tasks:
            name = task["id"]
            wall, cpu = time.monotonic(), time.process_time()
            repair = refine(
                lambda examples: solve(method, examples), task["observed"],
                pool, lambda values: oracle(name, values),
                protocol["max_added_examples"])
            role = ("old_retention" if task in old_tasks else
                    "new_donor_retention" if name in qualified_names else
                    "transfer")
            rows.append({"id": name, "chosen": repair["chosen"],
                         "observed": repair["observed"], "role": role,
                         "repair": repair, "wall_seconds":
                         time.monotonic() - wall,
                         "cpu_seconds": time.process_time() - cpu})
            print(json.dumps({"method": method, "id": name,
                              "tokens": repair["chosen"]["tokens"]}),
                  flush=True)
        results.append((method, rows))
    for model, metadata, expected in (previous, current):
        if state_fingerprint(model.state_dict(), metadata) != expected:
            raise ValueError("query repair changed frozen parameters")
    if state_fingerprint(untrained.state_dict(), {"seed": 0}) != initial:
        raise ValueError("query repair changed untrained control")
    rng = random.Random(protocol["held_seed"])
    held = [[rng.randrange(-100, 101)
             for _ in range(rng.randrange(13, 101))] for _ in range(40)]
    visible = {tuple(values) for values in pool}
    visible |= {tuple(values) for _, rows in results for row in rows
                for values, _ in row["observed"]}
    if any(tuple(values) in visible for values in held):
        raise ValueError("query and held sets overlap")
    controls = []
    for method, rows in results:
        for row in rows:
            tokens = row["chosen"]["tokens"]
            try:
                row["held_ir_accepted"] = tokens is not None and all(
                    execute(tuple(tokens), values) == oracle(row["id"], values)
                    for values in held)
            except ValueError:
                row["held_ir_accepted"] = False
        controls.append({"method": method, "rows": rows, "accepted": sum(
            row["held_ir_accepted"] for row in rows)})
        print(json.dumps({"method": method, "held_ir_accepted":
                          controls[-1]["accepted"]}), flush=True)
    report = {"scope": protocol["scope"], "tasks": len(tasks),
              "rows": results[0][1], "held_per_task": len(held),
              "models": {"previous": previous[2], "current": current[2]},
              "execution_pending": True}
    for name, data in (("report.json", report), ("held-inputs.json", held),
                       ("fragment-qualification.json", {"old": old,
                                                         "new": new,
                                                         "branches":
                                                         branch_proofs}),
                       ("controls.json", {"scope": "Independent held IR "
                                         "checks only; target languages "
                                         "pending for the neural route.",
                                          "rows": controls})):
        (root / name).write_text(json.dumps(data, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
