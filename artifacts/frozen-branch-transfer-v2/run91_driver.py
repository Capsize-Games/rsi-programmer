"""Parent driver for issue #91 serialized arms; ledger-timed.

Runs the frozen protocol exactly: all seven search arms first
(held inputs stay sealed), then held evaluation per arm, then
CURLEE checks per arm. Every command's CPU cost is recorded in a
CostLedger (issue #92) written next to this file. Donor-library
creation remains a declared exclusion (see cost-accounting.json).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts/frozen-branch-transfer-v2"
PY = str(ROOT / ".venv/bin/python")
COMPILER = "/home/joe/Projects/curlee/build/linux-debug/curlee"

from rsi_programmer.cost_ledger import CostLedger  # noqa: E402

ARMS: list[tuple[str, int | None]] = [
    ("classical", None),
    ("neural", 0), ("neural", 1), ("neural", 2),
    ("untrained", 0), ("untrained", 1), ("untrained", 2),
]


def arm_path(method: str, seed: int | None) -> Path:
    if seed is None:
        return OUT / method
    return OUT / method / f"seed-{seed}"


def arm_args(method: str, seed: int | None) -> list[str]:
    args = ["--method", method, "--output", str(OUT)]
    if seed is not None:
        args[2:2] = ["--seed", str(seed)]
    return args


def run(ledger: CostLedger, phase: str, argv: list[str]) -> None:
    print(f"== {phase}: {' '.join(argv[1:])}", flush=True)
    code = ledger.run(phase, argv)
    print(f"   exit={code}", flush=True)
    if code != 0:
        raise SystemExit(f"command failed ({code}): {argv}")


def main() -> int:
    os.chdir(ROOT)
    os.environ.update({"PYTHONPATH": str(ROOT),
                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    ledger = CostLedger()
    ledger.declare_excluded(
        "donor-library creation from empty (mixed-unit ~50.1s "
        "lower bound; needs instrumented rerun, issue #92)")
    ledger.declare_excluded(
        "bank creation/freeze (complete before this run)")
    for method, seed in ARMS:
        run(ledger, "search",
            [PY, "scripts/run_frozen_branch_transfer_v2.py",
             *arm_args(method, seed)])
    for method, seed in ARMS:
        run(ledger, "candidate_evaluation",
            [PY, "scripts/evaluate_frozen_branch_transfer_v2.py",
             *arm_args(method, seed)])
    for method, seed in ARMS:
        run(ledger, "backend_execution",
            [PY, "scripts/check_frozen_branch_curlee_v2.py",
             "--compiler", COMPILER,
             "--output", str(arm_path(method, seed))])
    ledger.write_json(OUT / "cost-ledger.json")
    print("LEDGER TOTALS:", ledger.totals(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
