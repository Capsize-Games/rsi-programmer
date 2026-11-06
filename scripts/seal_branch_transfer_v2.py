"""Pin v2 controller and checker sources before opening outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "fixtures/frozen_branch_transfer_v2"
FILES = ["scripts/run_frozen_branch_transfer_v2.py",
         "scripts/evaluate_frozen_branch_transfer_v2.py",
         "scripts/check_frozen_branch_curlee_v2.py",
         "scripts/branch_transfer_v2_search.py",
         "scripts/evaluate_statement_decisions.py",
         "scripts/check_statement_curlee.py",
         "scripts/check_useful_curlee.py",
         "scripts/measure_useful_program.py",
         "rsi_programmer/program_cost.py",
         "rsi_programmer/statement_policy.py",
         "rsi_programmer/statement_curriculum.py",
         "rsi_programmer/statement_stateful_curriculum.py",
         "rsi_programmer/statement_return_curriculum.py",
         "rsi_programmer/stateful_tasks.py",
         "rsi_programmer/statement_syntax.py",
         "rsi_programmer/statement_execution.py",
         "rsi_programmer/statement_partial.py",
         "rsi_programmer/statement_source.py",
         "rsi_programmer/statement_branch_library.py",
         "rsi_programmer/statement_branch_search.py",
         "rsi_programmer/statement_fragment_search.py",
         "rsi_programmer/statement_fragment_classical.py",
         "rsi_programmer/statement_fragment_library.py",
         "rsi_programmer/statement_fragments.py",
         "rsi_programmer/statement_repair.py",
         "rsi_programmer/utility_policy.py",
         "rsi_programmer/utility_source.py",
         "rsi_programmer/utility_ops.py",
         "rsi_programmer/utility_harness.py",
         "rsi_programmer/benchmark_policy.py"]


def seal_bank(bank: Path) -> dict[str, str]:
    path = bank / "seal.json"
    if path.exists():
        raise ValueError("bank already sealed; do not overwrite its seal")
    seal = {"scope": "Sources pinned before any transfer controller runs.",
            "bank_hashes": json.loads((bank / "sha256.json").read_text()),
            "solver_sources": {name: hashlib.sha256(
                (ROOT / name).read_bytes()).hexdigest() for name in FILES},
            "backend_policy": "Execute fixed choices only. Timeout, "
                              "process failure or malformed output rejects "
                              "a program and stops further cases for it; "
                              "acceptance requires all held cases in both "
                              "backends."}
    path.write_text(json.dumps(seal, indent=2) + "\n")
    return {"sealed_sources": len(FILES), "bank": str(bank)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    args = parser.parse_args()
    print(json.dumps(seal_bank(args.bank.resolve())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
