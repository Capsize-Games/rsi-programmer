"""Retrospective composition-first classical control on fresh held rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

from rsi_programmer.statement_branch_library import qualify_branches
from rsi_programmer.statement_branch_search import synthesize as compose
from rsi_programmer.statement_curriculum import oracle
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_fragment_classical import synthesize
from rsi_programmer.statement_fragment_library import qualify
from rsi_programmer.statement_repair import refine

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    prior = ROOT / "artifacts/statement-composed-route-v1-20261101"
    protocol = json.loads((prior / "protocol.json").read_text())
    spec = {"scope": "Retrospective stronger classical control. Compose "
                     "before condition-only search; same visible query "
                     "pool and three-added-example cap. Fresh held IR "
                     "only; no target-language acceptance claim.",
            "held_seed": 461071, "held_per_task": 40,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()
                                          ).hexdigest()}
    (root / "protocol.json").write_text(json.dumps(spec, indent=2) + "\n")
    donors = ROOT / "artifacts/statement-context-repair-v1-20261101"
    branches = qualify_branches(donors)["branches"]
    branches += qualify_branches(
        ROOT / "artifacts/positive-counter-v1-20261101")["branches"]
    contexts = qualify(
        ROOT / "artifacts/statement-return-v1-20261101")["fragments"]

    def solve(examples: list[tuple[list[int], int]]) -> dict[str, Any]:
        first = compose(None, examples, branches)
        if first["tokens"] is not None:
            return first
        second = synthesize(examples, contexts, 1.0)
        return {**second, "route": [first, second]}

    decisions = []
    fixture = json.loads(
        (ROOT / "fixtures/statement_tasks_v1.json").read_text())
    fixture["tasks"] += json.loads(
        (ROOT / "fixtures/stateful_tasks_v1.json").read_text())["tasks"]
    for task in fixture["tasks"]:
        name = task["id"]
        wall, cpu = time.monotonic(), time.process_time()
        repair = refine(solve, task["observed"], protocol["query_inputs"],
                        lambda values: oracle(name, values), rounds=3)
        decisions.append({"id": name, "chosen": repair["chosen"],
                          "repair": repair, "wall_seconds":
                          time.monotonic() - wall,
                          "cpu_seconds": time.process_time() - cpu})
    rng = random.Random(spec["held_seed"])
    held = [[rng.randrange(-100, 101)
             for _ in range(rng.randrange(13, 101))] for _ in range(40)]
    for row in decisions:
        tokens = row["chosen"]["tokens"]
        try:
            row["held_ir_accepted"] = tokens is not None and all(
                execute(tuple(tokens), values) == oracle(row["id"], values)
                for values in held)
        except ValueError:
            row["held_ir_accepted"] = False
    # Re-evaluate frozen prior choices; no new decisions or feedback.
    paired = {}
    for method in ("neural", "untrained", "classical"):
        rows = next(data["rows"] for data in json.loads(
            (prior / "controls.json").read_text())["rows"]
                    if data["method"] == method)
        paired[method] = sum(row["chosen"]["tokens"] is not None and all(
            execute(tuple(row["chosen"]["tokens"]), values) == oracle(
                row["id"], values) for values in held) for row in rows)
    result = {"scope": spec["scope"], "rows": decisions,
              "accepted": sum(row["held_ir_accepted"] for row in decisions),
              "frozen_prior_choices_on_same_fresh_rows": paired,
              "prior_report_sha256": hashlib.sha256(
                  (prior / "report.json").read_bytes()).hexdigest()}
    for name, data in (("report.json", result), ("held-inputs.json", held)):
        (root / name).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"accepted": result["accepted"], "prior": paired}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
