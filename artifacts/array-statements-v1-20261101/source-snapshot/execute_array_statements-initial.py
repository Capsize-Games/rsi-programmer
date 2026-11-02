"""Execute fixed array choices; run through CURLEE's existing guard."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from rsi_programmer.array_execution import execute as interpret
from rsi_programmer.array_source import (
    curlee_harness, curlee_source, python_source)
from rsi_programmer.array_tasks import oracle
from rsi_programmer.program_cost import CostClock
from scripts.check_statement_curlee import passed
from scripts.check_useful_curlee import digest, execute
from scripts.measure_useful_program import run_cli


def python_cases(folder: Path, tokens: tuple[str, ...], name: str,
                 inputs: list[list[int]]) -> list[dict[str, Any]]:
    records = []
    with tempfile.TemporaryDirectory(prefix="rsi-array-held-") as temp:
        work = Path(temp)
        source, output = work / "inputs", work / "outputs"
        source.mkdir()
        output.mkdir()
        for index, values in enumerate(inputs):
            filename = f"{index:03}.json"
            (source / filename).write_text(json.dumps(values) + "\n")
            try:
                result = run_cli(folder, source, output, filename, filename)
            except subprocess.TimeoutExpired:
                result = subprocess.CompletedProcess(
                    [], 124, "", "Generated array program timed out.")
            try:
                actual = json.loads((output / filename).read_text())
                output_error = None
            except (OSError, ValueError) as error:
                actual, output_error = None, str(error)
            try:
                interpreted = interpret(tokens, values)
            except ValueError:
                interpreted = None
            expected = oracle(name, values)
            correct = (result.returncode == 0 and isinstance(actual, list)
                       and all(type(v) is int for v in actual)
                       and actual == expected and interpreted == expected)
            records.append({"case": index, "actual": actual,
                            "expected": expected, "interpreted": interpreted,
                            "exit_code": result.returncode,
                            "stderr": result.stderr,
                            "output_error": output_error, "correct": correct})
            if result.returncode or output_error is not None:
                break
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    args = parser.parse_args()
    root, compiler = args.output.resolve(), args.compiler.resolve()
    report = json.loads((root / "report.json").read_text())
    if report["execution_pending"] is not True:
        raise ValueError("array decisions already executed")
    for name, expected in report["source_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError("array experiment source changed")
    inputs = json.loads((root / "held-inputs.json").read_text())
    clock = CostClock()
    before = digest(compiler)
    rows, negative = [], None
    for row in report["rows"]:
        entry = {**row, "python_records": [], "curlee_records": [],
                 "python_accepted": False, "curlee_accepted": False}
        tokens = row["chosen"]["tokens"]
        if tokens is not None:
            tokens = tuple(tokens)
            folder = root / row["method"] / row["id"]
            for language in ("python", "curlee"):
                (folder / language).mkdir(parents=True, exist_ok=False)
            py = folder / "python/clean_readings.py"
            cr = folder / "curlee/process.curlee"
            py.write_text(python_source(tokens))
            cr.write_text(curlee_source(tokens))
            entry.update(python_sha256=digest(py), curlee_sha256=digest(cr))
            records = python_cases(folder / "python", tokens, row["id"],
                                   inputs)
            entry["python_records"] = records
            entry["python_accepted"] = (len(records) == len(inputs)
                                        and all(x["correct"] for x in records))
            for index, values in enumerate(inputs):
                path = folder / "curlee" / f"case-{index:03}.curlee"
                path.write_text(curlee_harness(
                    tokens, values, oracle(row["id"], values)))
                checked = execute(compiler, path, "check")
                run = (execute(compiler, path, "run")
                       if checked["exit_code"] == 0 else {})
                entry["curlee_records"].append({
                    "case": index, "check": checked, "run": run,
                    "passed": passed(run), "source_sha256": digest(path)})
                if checked["exit_code"]:
                    break
            records = entry["curlee_records"]
            entry["curlee_accepted"] = (len(records) == len(inputs)
                                        and all(x["passed"] for x in records))
            if negative is None and entry["curlee_accepted"]:
                path = folder / "curlee/wrong_expected.curlee"
                path.write_text(curlee_harness(tokens, [], [123]))
                run = execute(compiler, path, "run")
                negative = {"run": run, "rejected": passed(run, 1),
                            "source_sha256": digest(path)}
        entry["accepted"] = (entry["python_accepted"]
                             and entry["curlee_accepted"])
        rows.append(entry)
        print(json.dumps({"method": row["method"], "id": row["id"],
                          "accepted": entry["accepted"]}), flush=True)
    after = digest(compiler)
    report.update(rows=rows, execution_pending=False,
                  compiler_before=before, compiler_after=after,
                  compiler_stable=before == after,
                  wrong_expected_control=negative,
                  execution_cost=clock.elapsed())
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(before != after or negative is None or not negative["rejected"])


if __name__ == "__main__":
    raise SystemExit(main())
