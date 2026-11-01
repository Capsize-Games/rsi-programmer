"""Freeze transfer choices after independently qualifying a new donor."""
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
from rsi_programmer.statement_curriculum import oracle
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_fragment_classical import synthesize as classical
from rsi_programmer.statement_fragment_library import qualify
from rsi_programmer.statement_fragment_search import synthesize
from rsi_programmer.statement_policy import StatementPolicy

ROOT = Path(__file__).resolve().parents[1]
Pairs = list[tuple[list[int], int]]


def _model(filename: str, prior: Path
           ) -> tuple[StatementPolicy, dict[str, Any], str]:
    data = torch.load(ROOT / "weights" / filename, weights_only=True)
    model = StatementPolicy(seed=0, execution_context=True)
    model.load_state_dict(data["state"])
    model.eval()
    expected = json.loads((prior / "report.json").read_text())["trained_hash"]
    if state_fingerprint(model.state_dict(), data["metadata"]) != expected:
        raise ValueError("context-growth checkpoint differs from prior")
    return model, data["metadata"], expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    old_root = ROOT / "artifacts/statement-programs-v1-execution-20261101"
    new_root = ROOT / "artifacts/statement-return-v1-20261101"
    old_tasks = json.loads(
        (ROOT / "fixtures/statement_tasks_v1.json").read_text())
    new_tasks = json.loads(
        (ROOT / "fixtures/stateful_tasks_v1.json").read_text())
    tasks = [*old_tasks["tasks"], *new_tasks["tasks"]]
    queries = json.loads((ROOT / "artifacts/stateful-queries-v1-20261101" /
                          "protocol.json").read_text())["query_inputs"][:3]
    protocol = {
        "scope": "Development verified-donor growth. Classical context "
                 "mining; frozen neural condition filling. No refit.",
        "methods": ["retained_first", "current_old_library",
                    "current_new_library", "previous_new_library",
                    "untrained_new_library", "classical_new_library"],
        "budget_per_search": 1.0, "choices_per_prefix": 4, "beam_width": 32,
        "additional_observed_inputs": queries,
        "held_seed": 461043, "held_per_task": 40,
        "primary_backend_method": "retained_first",
        "cost_scope": "Search component only. The primary route can use "
                      "two searches; not a matched-total-cost gate.",
        "source_hashes": {name: hashlib.sha256(
            (ROOT / name).read_bytes()).hexdigest() for name in (
                "scripts/measure_context_growth.py",
                "rsi_programmer/statement_fragment_search.py",
                "rsi_programmer/statement_fragment_classical.py",
                "rsi_programmer/statement_fragment_library.py",
                "fixtures/statement_tasks_v1.json",
                "fixtures/stateful_tasks_v1.json")}}
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    started = time.monotonic()
    old, new = qualify(old_root), qualify(new_root)
    if not {f["id"] for f in old["fragments"]} <= {
            f["id"] for f in new["fragments"]}:
        raise ValueError("expanded library dropped an old context")
    qualification = {"old": old, "new": new,
                     "wall_seconds": time.monotonic() - started}
    torch.set_num_threads(1)
    models = {
        "previous": _model("statement-repair-v1.pt", old_root),
        "current": _model("statement-return-v1.pt", new_root)}
    untrained = StatementPolicy(seed=0, execution_context=True)
    untrained.eval()
    metadata = {"seed": 0, "execution_context": True}
    models["untrained"] = (
        untrained, metadata,
        state_fingerprint(untrained.state_dict(), metadata))
    budget = protocol["budget_per_search"]

    def solve(method: str, observed: Pairs) -> dict[str, Any]:
        if method == "classical_new_library":
            return classical(observed, new["fragments"], budget)
        if method == "retained_first":
            first = synthesize(models["previous"][0], observed,
                               old["fragments"], budget_seconds=budget)
            if first["tokens"] is not None:
                return {**first, "selected_model": "previous",
                        "route": [first]}
            second = synthesize(models["current"][0], observed,
                                new["fragments"], budget_seconds=budget)
            return {**second, "selected_model": "current",
                    "route": [first, second]}
        model_name = method.split("_", 1)[0]
        fragments = old if method == "current_old_library" else new
        return synthesize(models[model_name][0], observed,
                          fragments["fragments"], budget_seconds=budget)

    decisions = []
    for method in protocol["methods"]:
        rows = []
        for task in tasks:
            name = task["id"]
            observed = [*task["observed"], *[
                (values, oracle(name, values)) for values in queries]]
            wall, cpu = time.monotonic(), time.process_time()
            chosen = solve(method, observed)
            role = ("old_retention" if task in old_tasks["tasks"] else
                    "new_donor_retention" if name in new["donors"] else
                    "transfer")
            rows.append({"id": name, "observed": observed, "chosen": chosen,
                         "role": role, "wall_seconds": time.monotonic() - wall,
                         "cpu_seconds": time.process_time() - cpu})
            print(json.dumps({"method": method, "id": name,
                              "tokens": chosen["tokens"]}), flush=True)
        decisions.append((method, rows))
    for model, metadata, expected in models.values():
        if state_fingerprint(model.state_dict(), metadata) != expected:
            raise ValueError("context transfer changed model parameters")
    # Solvers and controller comparisons end before creating new held rows.
    rng = random.Random(protocol["held_seed"])
    held = [[rng.randrange(-100, 101)
             for _ in range(rng.randrange(13, 101))] for _ in range(40)]
    controls = []
    for method, rows in decisions:
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
    primary = decisions[0][1]
    report = {"scope": protocol["scope"], "tasks": len(primary),
              "held_per_task": len(held), "rows": primary,
              "models": {name: fingerprint for name, (_, _, fingerprint)
                         in models.items()}, "execution_pending": True}
    for name, data in (("report.json", report), ("held-inputs.json", held),
                       ("fragment-qualification.json", qualification),
                       ("controls.json", {"scope": "Held IR only; not "
                                         "target-language execution or "
                                         "matched total cost.",
                                          "rows": controls})):
        (root / name).write_text(json.dumps(data, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
