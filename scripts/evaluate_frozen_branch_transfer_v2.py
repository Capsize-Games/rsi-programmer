"""Evaluate sealed v2 inputs only after every frozen arm finishes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
from typing import Any

from rsi_programmer.program_cost import cpu_seconds
from rsi_programmer.statement_execution import execute
from scripts.evaluate_statement_decisions import evaluate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "fixtures/frozen_branch_transfer_v2"


def expected_reports(seeds: list[int]) -> list[Path]:
    paths = [Path("classical")]
    for method in ("neural", "untrained"):
        paths += [Path(method) / f"seed-{seed}" for seed in seeds]
    return paths


def _parse_seeds(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def _gate_reports(root: Path, seeds: list[int]) -> None:
    # Refuse to expose held data while another arm is undecided.
    for relative in expected_reports(seeds):
        if not (root / relative / "report.json").is_file():
            raise ValueError("all frozen arms must finish first")


def _load_held(bank: Path) -> list[list[int]]:
    hashes = json.loads((bank / "sha256.json").read_text())
    if hashlib.sha256((bank / "held.json").read_bytes()).hexdigest() != (
            hashes["held.json"]):
        raise ValueError("sealed held inputs changed")
    return json.loads((bank / "held.json").read_text())


def _references(bank: Path) -> dict[str, tuple[str, ...]]:
    tasks = json.loads((bank / "tasks.json").read_text())
    return {task["id"]: tuple(task["reference"])
            for task in tasks["tasks"]}


def _evaluate_report(folder: Path, held: list[list[int]],
                     references: dict[str, tuple[str, ...]]) -> None:
    report = json.loads((folder / "report.json").read_text())
    if report["execution_pending"] is not True:
        raise ValueError("frozen choices already evaluated")
    rows = evaluate(folder, report["rows"], held,
                    task_oracle=lambda name, values:
                    execute(references[name], values))
    report.update(rows=rows, execution_pending=False,
                  accepted=sum(row["accepted"] for row in rows))
    _write_outputs(folder, report, held)


def _write_outputs(folder: Path, report: dict[str, Any],
                   held: list[list[int]]) -> None:
    (folder / "held-inputs.json").write_text(json.dumps(held) + "\n")
    (folder / "report.json").write_text(json.dumps(report, indent=2)
                                        + "\n")
    cost = {"process_cpu_seconds": cpu_seconds(),
            "peak_parent_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss}
    (folder / "python-cost.json").write_text(json.dumps(cost, indent=2)
                                             + "\n")
    print(json.dumps({"accepted": report["accepted"],
                      "tasks": len(report["rows"])}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True,
                        choices=("neural", "classical", "untrained"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    args = parser.parse_args()
    bank = args.bank.resolve()
    root = args.output.resolve()
    _gate_reports(root, _parse_seeds(args.seeds))
    held = _load_held(bank)
    folder = root / args.method if args.method == "classical" else (
        root / args.method / f"seed-{args.seed}")
    _evaluate_report(folder, held, _references(bank))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
