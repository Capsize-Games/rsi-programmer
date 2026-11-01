"""Probe frozen programming capability on unseen stateful task functions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch

from rsi_programmer.benchmark_policy import state_fingerprint
from rsi_programmer.statement_checkpoint import load_or_fit
from rsi_programmer.statement_curriculum import teachers
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_fragment_library import qualify
from rsi_programmer.statement_fragment_search import synthesize as transfer
from rsi_programmer.stateful_tasks import oracle
from scripts.evaluate_statement_decisions import evaluate

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_HASH = (
    "52dccad6629d6bbb5de87e9e1c7888baaf6e7fd7ed4e08abdaff377443c241ae")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    raw = (ROOT / "fixtures/stateful_tasks_v1.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_HASH:
        raise ValueError("frozen stateful task probe changed")
    spec = json.loads(raw)
    old_spec = json.loads(
        (ROOT / "fixtures/statement_tasks_v1.json").read_text())
    old_ids = [task["id"] for task in old_spec["tasks"]]
    training, probes = teachers(old_ids)
    signatures = {tuple(execute(program, values) for values in probes)
                  for program in training}
    for task in spec["tasks"]:
        signature = tuple(oracle(task["id"], values) for values in probes)
        if signature in signatures:
            raise ValueError("probe task matches a teacher on training probes")
    donor_root = ROOT / "artifacts/statement-programs-v1-execution-20261101"
    library = qualify(donor_root)
    prior = json.loads((donor_root / "report.json").read_text())
    torch.set_num_threads(1)
    model, checkpoint = load_or_fit(ROOT / "weights/statement-repair-v1.pt",
                                    prior["trained_hash"], old_ids)
    decisions = []
    for task in spec["tasks"]:
        initial = model.synthesize(task["observed"])
        chosen = initial
        if initial["tokens"] is None:
            chosen = transfer(
                model, task["observed"], library["fragments"],
                budget_seconds=spec["transfer_wall_budget_seconds"])
        decisions.append({"id": task["id"], "observed": task["observed"],
                          "initial": initial, "chosen": chosen})
        print(json.dumps({"id": task["id"], "tokens": chosen["tokens"]}),
              flush=True)
    if state_fingerprint(model.state_dict(), checkpoint["metadata"]) != (
            checkpoint["trained_hash"]):
        raise ValueError("frontier probe changed model parameters")
    # Neither these arrays nor witness programs enter the programming API.
    rng = random.Random(spec["held_seed"])
    held = [[], [0], [-4, -2, 0], [1, -2, 0], [9, 1, 3, 2],
            [100] * 256, [-100] * 256, list(range(-100, 101))] + [
        [rng.randrange(-100, 101) for _ in range(rng.randrange(1, 101))]
        for _ in range(spec["held_per_task"] - 8)]
    rows = evaluate(root, decisions, held, task_oracle=oracle)
    witness_root = root / "grammar-controls"
    witness_root.mkdir()
    witnesses = [{"id": task["id"], "chosen": {
        "tokens": task["grammar_witness"]}, "role": "supplied grammar control"}
                 for task in spec["tasks"]]
    witness_rows = evaluate(witness_root, witnesses, held, task_oracle=oracle)
    report = {"scope": "Development frontier probe with a frozen learner. "
                       "No refit, new teacher functions or held retry. "
                       "Supplied witnesses are controls, not neural outputs.",
              "fixture_sha256": FIXTURE_HASH, **checkpoint,
              "training_probe_function_overlap": False,
              "tasks": len(rows), "held_per_task": len(held), "rows": rows,
              "accepted": sum(row["accepted"] for row in rows)}
    control = {"scope": "Supplied programs proving grammar/backend capacity; "
                        "not neural programming successes.",
               "tasks": len(witness_rows), "rows": witness_rows,
               "accepted": sum(row["accepted"] for row in witness_rows)}
    for folder, data in ((root, report), (witness_root, control)):
        (folder / "report.json").write_text(json.dumps(data, indent=2) + "\n")
        (folder / "held-inputs.json").write_text(json.dumps(held) + "\n")
    print(json.dumps({"neural": report["accepted"], "tasks": len(rows),
                      "grammar_controls": control["accepted"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
