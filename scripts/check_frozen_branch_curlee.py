"""Qualify frozen-transfer CURLEE code with engine-owned reference outputs."""
from __future__ import annotations

import json
from pathlib import Path
import resource

from rsi_programmer.program_cost import cpu_seconds
from rsi_programmer.statement_execution import execute
from scripts.check_statement_curlee import main

BANK = (Path(__file__).resolve().parents[1]
        / "fixtures/frozen_branch_transfer_v1")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    tasks = json.loads((BANK / "tasks.json").read_text())["tasks"]
    reference = {task["id"]: tuple(task["reference"]) for task in tasks}
    try:
        code = main(task_oracle=lambda name, values:
                    execute(reference[name], values))
    finally:
        cost = {
            "process_cpu_seconds": cpu_seconds(),
            "peak_parent_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss}
        (args.output / "curlee-cost.json").write_text(
            json.dumps(cost, indent=2) + "\n")
    raise SystemExit(code)
