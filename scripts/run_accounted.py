"""Record complete task-command CPU and wall cost, including child jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess

from rsi_programmer.program_cost import CostClock, cpu_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("task command required")
    clock = CostClock()
    result = subprocess.run(command, check=False)
    data = {"command": command, "exit_code": result.returncode,
            **clock.elapsed(), "whole_process_cpu_seconds": cpu_seconds(),
            "largest_child_rss_kib": resource.getrusage(
                resource.RUSAGE_CHILDREN).ru_maxrss,
            "accounting_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest()}
    args.cost_file.parent.mkdir(parents=True, exist_ok=True)
    args.cost_file.write_text(json.dumps(data, indent=2) + "\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
