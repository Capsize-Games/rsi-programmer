"""Measure every frozen useful-composition task and execute emitted Python."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import torch

from rsi_programmer.utility_ops import execute
from rsi_programmer.utility_policy import UtilityPolicy
from rsi_programmer.utility_source import emit_curlee, emit_python
from rsi_programmer.utility_suite import oracle
from scripts.measure_useful_program import run_cli

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_HASH = (
    "e23d0477d5c4a11837734b906c66d6860699ec031874c3ac44efd627e15a63b3")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    raw = (ROOT / "fixtures/useful_compositions_v1.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_HASH:
        raise ValueError("frozen task suite changed")
    spec = json.loads(raw)
    torch.set_num_threads(1)
    model = UtilityPolicy(seed=spec["seed"])
    initial = {task["id"]: model.synthesize(task["observed"])
               for task in spec["tasks"]}
    started = time.monotonic()
    fit = model.fit(epochs=spec["epochs"], excluded=[
        tuple(task["exclusion_program"]) for task in spec["tasks"]])
    fit["wall_seconds"] = time.monotonic() - started
    # Finish every programming choice before generating held inputs.
    started = time.monotonic()
    chosen = {task["id"]: model.synthesize(task["observed"])
              for task in spec["tasks"]}
    synthesis_seconds = time.monotonic() - started
    print(json.dumps({"fit": fit, "programs": chosen}), flush=True)
    rng = random.Random(spec["held_seed"])
    held = [[], [0], [-3, -1, 0], [2, 2, 0, -2, -2, 3],
            list(range(100, -101, -1))] + [
        [rng.randrange(-100, 101) for _ in range(rng.randrange(1, 101))]
        for _ in range(spec["held_per_task"] - 5)]
    (root / "held-inputs.json").write_text(json.dumps(held) + "\n")
    rows: list[dict[str, Any]] = []
    for task in spec["tasks"]:
        name = task["id"]
        program = chosen[name]
        expected = [oracle(name, values) for values in held]
        entry: dict[str, Any] = {
            "id": name, "previously_tested": task["previously_tested"],
            "program": program, "untrained_program": initial[name],
            "observed": task["observed"], "expected": expected,
            "python_records": [], "python_accepted": False,
            "ir_passed": 0,
        }
        if program is not None:
            folder = root / name
            for directory in ("python", "curlee", "inputs"):
                (folder / directory).mkdir(parents=True)
            python = emit_python(program)
            curlee = emit_curlee(program)
            (folder / "python/clean_readings.py").write_text(python)
            (folder / "curlee/process.curlee").write_text(curlee)
            entry["python_sha256"] = hashlib.sha256(
                python.encode()).hexdigest()
            entry["curlee_sha256"] = hashlib.sha256(
                curlee.encode()).hexdigest()
            entry["ir_passed"] = sum(
                execute(program, values) == answers
                for values, answers in zip(held, expected))
            with tempfile.TemporaryDirectory(prefix="rsi-suite-") as temp:
                output = Path(temp)
                for index, (values, answers) in enumerate(zip(held, expected)):
                    filename = f"{index:03}.json"
                    (folder / "inputs" / filename).write_text(
                        json.dumps(values) + "\n")
                    result = run_cli(folder / "python", folder / "inputs",
                                     output, filename, filename)
                    path = output / filename
                    actual = (json.loads(path.read_text())
                              if path.exists() else None)
                    entry["python_records"].append({
                        "case": index, "exit_code": result.returncode,
                        "actual": actual, "expected": answers,
                        "correct": result.returncode == 0
                        and actual == answers,
                        "stderr": result.stderr})
            entry["python_accepted"] = all(
                record["correct"] for record in entry["python_records"])
        rows.append(entry)
        print(json.dumps({key: entry[key] for key in
                          ("id", "program", "python_accepted", "ir_passed")}),
              flush=True)
    new = [row for row in rows if not row["previously_tested"]]
    report = {"scope": "Development only. Eight new whole task functions "
                       "plus one previous-task control; all requested "
                       "behaviors excluded on teacher probes before fit. "
                       "The emitter supplies primitive implementations.",
              "fixture_sha256": FIXTURE_HASH, "fit": fit,
              "synthesis_wall_seconds": synthesis_seconds,
              "held_per_task": len(held), "tasks": len(rows),
              "python_accepted": sum(row["python_accepted"] for row in rows),
              "new_tasks": len(new), "new_python_accepted": sum(
                  row["python_accepted"] for row in new), "rows": rows}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("tasks", "python_accepted", "new_tasks",
                       "new_python_accepted")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
