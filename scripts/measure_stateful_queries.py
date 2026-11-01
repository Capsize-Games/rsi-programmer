"""Test visible query feedback without refitting or reusing held rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch

from rsi_programmer.benchmark_policy import state_fingerprint
from rsi_programmer.statement_fragment_library import qualify
from rsi_programmer.statement_fragment_search import synthesize as transfer
from rsi_programmer.statement_policy import StatementPolicy
from rsi_programmer.statement_repair import refine
from rsi_programmer.stateful_tasks import oracle
from scripts.evaluate_statement_decisions import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    fixture = ROOT / "fixtures/stateful_tasks_v1.json"
    tasks = json.loads(fixture.read_text())["tasks"]
    query_file = ROOT / "fixtures/statement_repair_v1.json"
    pool = json.loads(query_file.read_text())["query_inputs"]
    priors = [ROOT / "artifacts" / name for name in (
        "stateful-programs-v1-execution-20261101",
        "stateful-curriculum-v1-20261101")]
    sealed = {tuple(values) for folder in priors for values in
              json.loads((folder / "held-inputs.json").read_text())}
    pool = [values for values in pool if tuple(values) not in sealed]
    protocol = {"scope": "Development query trial, no refit or held retry.",
                "max_added_examples": 3, "query_inputs": pool,
                "excluded_prior_held_rows": len(sealed),
                "held_seed": 461033, "held_per_task": 40,
                "transfer_seconds": 0.5,
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()).hexdigest(),
                "fixture_sha256": hashlib.sha256(
                    fixture.read_bytes()).hexdigest()}
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    library = qualify(
        ROOT / "artifacts/statement-programs-v1-execution-20261101")
    torch.set_num_threads(1)
    results = []
    for label, name, prior in zip(
            ("qualified", "broader"),
            ("statement-repair-v1.pt", "statement-stateful-v1.pt"), priors):
        checkpoint = torch.load(ROOT / "weights" / name, weights_only=True)
        expected = json.loads((prior / "report.json").read_text())[
            "trained_hash"]
        model = StatementPolicy(seed=0, execution_context=True)
        model.load_state_dict(checkpoint["state"])
        model.eval()
        if state_fingerprint(model.state_dict(), checkpoint["metadata"]) != (
                expected):
            raise ValueError("query trial checkpoint differs from prior")
        decisions = []
        for task in tasks:
            name = task["id"]
            repair = refine(
                model.synthesize, task["observed"], pool,
                lambda values: oracle(name, values),
                protocol["max_added_examples"])
            chosen = repair["chosen"]
            if chosen["tokens"] is None:
                chosen = transfer(model, repair["observed"],
                                  library["fragments"],
                                  budget_seconds=protocol["transfer_seconds"])
            decisions.append({"id": name, "chosen": chosen,
                              "observed": repair["observed"],
                              "query_history": repair})
            print(json.dumps({"model": label, "id": name,
                              "tokens": chosen["tokens"]}), flush=True)
        if state_fingerprint(model.state_dict(), checkpoint["metadata"]) != (
                expected):
            raise ValueError("query feedback changed model weights")
        results.append((label, expected, decisions))
    # All models finish decisions before any new held input is generated.
    rng = random.Random(protocol["held_seed"])
    forbidden = sealed | {tuple(values) for values in pool}
    forbidden |= {tuple(values) for task in tasks
                  for values, _ in task["observed"]}
    held = []
    while len(held) < protocol["held_per_task"]:
        values = [rng.randrange(-100, 101)
                  for _ in range(rng.randrange(13, 101))]
        if tuple(values) not in forbidden:
            held.append(values)
    for label, fingerprint, decisions in results:
        folder = root / label
        folder.mkdir()
        rows = evaluate(folder, decisions, held, task_oracle=oracle)
        report = {"scope": protocol["scope"], "model": label,
                  "trained_hash": fingerprint, "tasks": len(rows),
                  "held_per_task": len(held), "rows": rows,
                  "accepted": sum(row["accepted"] for row in rows)}
        (folder / "report.json").write_text(
            json.dumps(report, indent=2) + "\n")
        (folder / "held-inputs.json").write_text(json.dumps(held) + "\n")
        print(json.dumps({"model": label, "accepted": report["accepted"]}),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
