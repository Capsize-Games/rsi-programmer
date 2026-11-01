"""Reproduce query-guided resynthesis and verified-fragment transfer."""
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
from rsi_programmer.statement_checkpoint import load_or_fit
from rsi_programmer.statement_curriculum import oracle
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_fragment_classical import synthesize as classical
from rsi_programmer.statement_fragment_library import qualify
from rsi_programmer.statement_fragment_search import synthesize as transfer
from rsi_programmer.statement_policy import StatementPolicy
from rsi_programmer.statement_repair import refine
from scripts.evaluate_statement_decisions import evaluate

ROOT = Path(__file__).resolve().parents[1]
REPAIR_HASH = (
    "028fc79e4807052f680d52735622c27f3f78985bc345ca67cebf04e71e536168")
TRANSFER_HASH = (
    "4e728eb0f8d8424525d89961fc4035f33f35d9de157cb9c8304c5353c32c105e")


def fixture(name: str, expected: str) -> dict[str, Any]:
    raw = (ROOT / "fixtures" / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("frozen repair/transfer protocol changed")
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "weights/statement-repair-v1.pt")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    protocol = fixture("statement_repair_v1.json", REPAIR_HASH)
    control_spec = fixture("statement_transfer_v1.json", TRANSFER_HASH)
    tasks = fixture("statement_tasks_v1.json",
                    protocol["base_fixture_sha256"])["tasks"]
    donor_root = ROOT / "artifacts/statement-programs-v1-execution-20261101"
    qualification = qualify(donor_root)
    donor_report = json.loads((donor_root / "report.json").read_text())
    fragments = qualification["fragments"]
    torch.set_num_threads(1)
    model, checkpoint = load_or_fit(args.checkpoint,
                                    donor_report["trained_hash"],
                                    [task["id"] for task in tasks])
    repair_rows, decisions, control_rows = [], [], []
    untrained = StatementPolicy(seed=0, execution_context=True)
    untrained.eval()
    for task in tasks:
        name = task["id"]
        repair = refine(model.synthesize, task["observed"],
                        protocol["query_inputs"],
                        lambda values: oracle(name, values),
                        protocol["max_added_examples"])
        repair["id"] = name
        repair_rows.append(repair)
        chosen = repair["chosen"]
        if chosen["tokens"] is None:
            chosen = transfer(
                model, repair["observed"], fragments,
                budget_seconds=control_spec["wall_budget_seconds"])
            for method in control_spec["methods"]:
                started = time.process_time()
                if method == "classical_breadth_first":
                    out = classical(repair["observed"], fragments,
                                    control_spec["wall_budget_seconds"])
                else:
                    active = (untrained if method == "untrained_neural"
                              else model)
                    out = transfer(active, repair["observed"], fragments,
                                   rank_contexts=method.endswith(
                                       "context_rank"),
                                   budget_seconds=control_spec[
                                       "wall_budget_seconds"])
                control_rows.append({"id": name, "method": method,
                                     "chosen": out, "cpu_seconds":
                                     time.process_time() - started})
        decisions.append({"id": name, "chosen": chosen,
                          "observed": repair["observed"], "used_fragment":
                          chosen.get("fragment") is not None})
        print(json.dumps({"id": name, "tokens": chosen["tokens"]}), flush=True)
    if state_fingerprint(model.state_dict(), checkpoint["metadata"]) != (
            checkpoint["trained_hash"]):
        raise ValueError("repair or transfer changed neural parameters")
    # Held values are unavailable until every programming/control choice ends.
    rng = random.Random(protocol["held_seed"])
    held = [list(range(-100, 101)), [100] * 256, [-100] * 256,
            [0, 100, -100, 100, 0, -100] * 20] + [
        [rng.randrange(-100, 101) for _ in range(rng.randrange(13, 101))]
        for _ in range(protocol["held_per_task"] - 4)]
    if {tuple(values) for values in held} & {
            tuple(values) for values in protocol["query_inputs"]}:
        raise ValueError("query and held pools overlap")
    for row in control_rows:
        tokens = row["chosen"]["tokens"]
        row["accepted"] = tokens is not None and all(
            execute(tuple(tokens), values) == oracle(row["id"], values)
            for values in held)
    rows = evaluate(root, decisions, held)
    records = {
        "fragment-qualification.json": qualification,
        "repair-decisions.json": {**checkpoint, "rows": repair_rows},
        "transfer-controls.json": {"scope": "Transfer-stage wall budget only; "
                                   "not a complete matched-total-cost gate.",
                                   "rows": control_rows},
        "held-inputs.json": held,
        "report.json": {"scope": "Development resynthesis and transfer; "
                                  "four donor retentions and two transfers.",
                        **checkpoint, "tasks": len(rows), "rows": rows,
                        "held_per_task": len(held),
                        "accepted": sum(row["accepted"] for row in rows)},
    }
    for filename, record in records.items():
        (root / filename).write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
