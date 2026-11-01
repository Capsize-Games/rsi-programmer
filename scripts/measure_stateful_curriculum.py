"""One frozen curriculum extension, with transfer and retention checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import torch

from rsi_programmer.benchmark_policy import state_fingerprint
from rsi_programmer.statement_curriculum import oracle
from rsi_programmer.statement_fragment_library import qualify
from rsi_programmer.statement_fragment_search import synthesize as transfer
from rsi_programmer.statement_policy import StatementPolicy
from rsi_programmer.statement_repair import refine
from rsi_programmer.statement_syntax import TOKENS
from scripts.evaluate_statement_decisions import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    sources = ["rsi_programmer/statement_stateful_curriculum.py",
               "rsi_programmer/statement_curriculum.py",
               "rsi_programmer/statement_policy.py", __file__,
               "fixtures/statement_tasks_v1.json",
               "fixtures/stateful_tasks_v1.json",
               "fixtures/statement_repair_v1.json"]
    protocol = {
        "scope": "Development curriculum trial, not final evaluation. "
                 "One seed and recipe; no held-guided retries.",
        "seed": 0, "epochs": 30, "curriculum_seed": 1237,
        "single_condition_candidates": 1536,
        "composed_condition_candidates": 512,
        "held_seed": 461027, "held_per_task": 40,
        "beam_width": 32, "transfer_seconds": 0.5,
        "source_hashes": {
            str(Path(path).resolve().relative_to(ROOT)): hashlib.sha256(
                (ROOT / path).read_bytes()).hexdigest() for path in sources}}
    # Write the recipe before fitting or observing outcomes.
    (root / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n")
    old = json.loads((ROOT / "fixtures/statement_tasks_v1.json").read_text())
    new = json.loads((ROOT / "fixtures/stateful_tasks_v1.json").read_text())
    repair = json.loads(
        (ROOT / "fixtures/statement_repair_v1.json").read_text())
    tasks = [*old["tasks"], *new["tasks"]]
    excluded = [task["id"] for task in tasks]
    library = qualify(
        ROOT / "artifacts/statement-programs-v1-execution-20261101")
    torch.set_num_threads(1)
    model = StatementPolicy(seed=0, execution_context=True)
    fit = model.fit(excluded, epochs=protocol["epochs"], stateful=True)
    metadata = {"seed": 0, "tokens": TOKENS, "max_tokens": 32,
                "execution_context": True, "curriculum": "stateful-v1"}
    trained_hash = state_fingerprint(model.state_dict(), metadata)
    checkpoint = ROOT / "weights/statement-stateful-v1.pt"
    torch.save({"state": model.state_dict(), "metadata": metadata,
                "trained_hash": trained_hash, "fit": fit}, checkpoint)
    print(json.dumps({"fit": fit, "trained_hash": trained_hash}), flush=True)
    decisions = []
    for task in tasks:
        name = task["id"]
        retention = task in old["tasks"]
        if retention:
            result = refine(
                model.synthesize, task["observed"], repair["query_inputs"],
                lambda values: oracle(name, values),
                repair["max_added_examples"])
            chosen, observed = result["chosen"], result["observed"]
        else:
            result = model.synthesize(task["observed"])
            chosen, observed = result, task["observed"]
        if chosen["tokens"] is None:
            chosen = transfer(model, observed, library["fragments"],
                              budget_seconds=protocol["transfer_seconds"])
        decisions.append({"id": name, "observed": observed,
                          "role": "retention" if retention else "new task",
                          "initial": result, "chosen": chosen})
        print(json.dumps({"id": name, "tokens": chosen["tokens"]}),
              flush=True)
    if state_fingerprint(model.state_dict(), metadata) != trained_hash:
        raise ValueError("program construction changed frozen parameters")
    # Generate held rows only after all programming and query decisions end.
    rng = random.Random(protocol["held_seed"])
    held = [[], [0], [-4, -2, 0], [1, -2, 0], [9, 1, 3, 2],
            [100] * 256, [-100] * 256, list(range(-100, 101))] + [
        [rng.randrange(-100, 101) for _ in range(rng.randrange(13, 101))]
        for _ in range(protocol["held_per_task"] - 8)]
    seen = {tuple(values) for decision in decisions
            for values, _ in decision["observed"]}
    # Empty/singleton query rows are excluded as held rows where necessary.
    held = [values for values in held if tuple(values) not in seen]
    while len(held) < protocol["held_per_task"]:
        values = [rng.randrange(-100, 101)
                  for _ in range(rng.randrange(13, 101))]
        if tuple(values) not in seen:
            held.append(values)
    report: dict[str, Any] = {
        "scope": protocol["scope"], "fit": fit, "metadata": metadata,
        "trained_hash": trained_hash, "tasks": len(decisions),
        "held_per_task": len(held), "rows": decisions,
        "execution_pending": True}
    for name, data in (("held-inputs.json", held), ("report.json", report),
                       ("fragment-qualification.json", library)):
        (root / name).write_text(json.dumps(data, indent=2) + "\n")
    rows = evaluate(root, decisions, held)
    report.update(rows=rows, execution_pending=False,
                  accepted=sum(row["accepted"] for row in rows))
    report["accepted_by_role"] = {
        role: sum(row["accepted"] for row in rows if row["role"] == role)
        for role in ("retention", "new task")}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["accepted_by_role"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
