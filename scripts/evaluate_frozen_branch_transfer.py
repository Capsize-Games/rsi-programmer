"""Evaluate sealed inputs only after every frozen controller finishes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource

from rsi_programmer.program_cost import cpu_seconds
from rsi_programmer.statement_execution import execute
from scripts.evaluate_statement_decisions import evaluate

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "fixtures/frozen_branch_transfer_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True,
                        choices=("neural", "classical", "untrained"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    # Refuse to expose held data while another controller is undecided.
    for method in ("neural", "classical", "untrained"):
        path = root / method / "report.json"
        if not path.is_file():
            raise ValueError("all three controllers must finish first")
    hashes = json.loads((BANK / "sha256.json").read_text())
    if hashlib.sha256((BANK / "held.json").read_bytes()).hexdigest() != (
            hashes["held.json"]):
        raise ValueError("sealed held inputs changed")
    bank = json.loads((BANK / "tasks.json").read_text())
    references = {task["id"]: tuple(task["reference"])
                  for task in bank["tasks"]}
    held = json.loads((BANK / "held.json").read_text())
    folder = root / args.method
    report = json.loads((folder / "report.json").read_text())
    if report["execution_pending"] is not True:
        raise ValueError("frozen choices already evaluated")
    rows = evaluate(folder, report["rows"], held,
                    task_oracle=lambda name, values:
                    execute(references[name], values))
    report.update(rows=rows, execution_pending=False,
                  accepted=sum(row["accepted"] for row in rows))
    (folder / "held-inputs.json").write_text(json.dumps(held) + "\n")
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    cost = {
        "process_cpu_seconds": cpu_seconds(),
        "peak_parent_rss_kib": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss}
    (folder / "python-cost.json").write_text(json.dumps(cost, indent=2) + "\n")
    print(json.dumps({"method": args.method, "accepted": report["accepted"],
                      "tasks": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
