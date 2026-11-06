"""Assemble summary.json from per-arm reports and the cost ledger."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ARMS: list[tuple[str, int | None]] = [
    ("classical", None),
    ("neural", 0), ("neural", 1), ("neural", 2),
    ("untrained", 0), ("untrained", 1), ("untrained", 2),
]


def arm_dir(method: str, seed: int | None) -> Path:
    if seed is None:
        return ROOT / method
    return ROOT / method / f"seed-{seed}"


def command_cpu(ledger: dict[str, Any]) -> dict[str, list[float]]:
    """Per-phase command CPU lists in driver (arm) order."""
    phases: dict[str, list[float]] = {}
    for key, entry in ledger["command_costs"].items():
        phase = key.rsplit("_", 1)[0] if key[-1].isdigit() else key
        phases.setdefault(phase, []).append(entry["cpu_seconds"])
    return phases


def arm_record(method: str, seed: int | None,
               costs: dict[str, float]) -> dict[str, Any]:
    folder = arm_dir(method, seed)
    report = json.loads((folder / "report.json").read_text())
    curlee = json.loads((folder / "curlee-execution.json").read_text())
    rows = report["rows"]
    py_ok = {row["id"] for row in rows if row["accepted"]}
    cu_ok = {row["id"] for row in curlee["rows"] if row["accepted"]}
    emitted = sum(1 for row in rows
                  if (row.get("chosen") or {}).get("tokens"))
    fits = report.get("fresh_fits") or []
    return {
        "method": method, "seed": seed, "tasks": report["tasks"],
        "emitted": emitted, "python_accepted": len(py_ok),
        "curlee_accepted": len(cu_ok), "both_accepted": len(py_ok & cu_ok),
        "python_actual_cases": sum(len(row["python_records"])
                                   for row in rows),
        "curlee_actual_cases": sum(len(row["records"])
                                   for row in curlee["rows"]),
        "wrong_expected_control_rejected": bool(
            curlee["wrong_expected_control"]),
        "compiler_stable": curlee["compiler_stable"],
        "fresh_fit_wall_seconds": sum(
            item["fit"]["wall_seconds"] for item in fits),
        "command_cpu_seconds": costs,
        "total_command_cpu_seconds": sum(costs.values()),
        "report_sha256": hashlib.sha256(
            (folder / "report.json").read_bytes()).hexdigest(),
    }


def main() -> int:
    ledger = json.loads((ROOT / "cost-ledger.json").read_text())
    by_phase = command_cpu(ledger)
    methods = []
    for index, (method, seed) in enumerate(ARMS):
        costs = {
            "search_incl_fit": by_phase["search"][index],
            "python_evaluation":
                by_phase["candidate_evaluation"][index],
            "curlee_execution": by_phase["backend_execution"][index],
        }
        methods.append(arm_record(method, seed, costs))
    summary = {
        "scope": ("Frozen branch-transfer v2: 54 sealed functions, "
                  "actual Python and CURLEE execution of every arm "
                  "on 20 held cases per task. Donor-library "
                  "creation declared excluded (see cost ledger)."),
        "bank": "fixtures/frozen_branch_transfer_v2",
        "methods": methods,
        "excluded_costs": ledger["excluded_costs"],
        "cost_ledger": "cost-ledger.json",
        "cost_definition": ledger["cost_definition"],
    }
    (ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(
        [(m["method"], m["seed"], m["both_accepted"],
          round(m["total_command_cpu_seconds"], 1)) for m in methods]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
