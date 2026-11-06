"""Qualify frozen-transfer v2 CURLEE code with engine-owned outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource

from rsi_programmer.program_cost import cpu_seconds
from rsi_programmer.statement_execution import execute
from scripts.check_statement_curlee import main

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "fixtures/frozen_branch_transfer_v2"


def qualifies(bank: Path) -> dict[str, tuple[str, ...]]:
    tasks = json.loads((bank / "tasks.json").read_text())["tasks"]
    return {task["id"]: tuple(task["reference"]) for task in tasks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    args, _ = parser.parse_known_args()
    reference = qualifies(args.bank.resolve())
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
