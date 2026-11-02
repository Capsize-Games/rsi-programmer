"""Pin controller and checker sources before opening transfer outcomes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "fixtures/frozen_branch_transfer_v1"


def main() -> int:
    path = BANK / "seal.json"
    if path.exists():
        raise ValueError("bank already sealed; do not overwrite its seal")
    files = ["scripts/run_frozen_branch_transfer.py",
             "scripts/evaluate_frozen_branch_transfer.py",
             "scripts/check_frozen_branch_curlee.py",
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
    seal = {"scope": "Sources pinned before any transfer controller runs.",
            "bank_hashes": json.loads((BANK / "sha256.json").read_text()),
            "solver_sources": {name: hashlib.sha256(
                (ROOT / name).read_bytes()).hexdigest() for name in files},
            "backend_policy": "Execute fixed choices only. Timeout, process "
                              "failure or malformed output rejects a program "
                              "and stops further cases for it; acceptance "
                              "requires all held cases in both backends."}
    path.write_text(json.dumps(seal, indent=2) + "\n")
    print(json.dumps({"sealed_sources": len(files), "bank": str(BANK)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
