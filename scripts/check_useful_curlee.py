"""Check and run emitted utility code with the existing CURLEE compiler.

Run this command through CURLEE's scripts/run_limited.py resource guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from rsi_programmer.utility_harness import curlee_harness
from rsi_programmer.utility_source import emit_curlee


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def execute(compiler: Path, source: Path, stage: str) -> dict[str, Any]:
    command = [str(compiler), stage]
    if stage == "run":
        command += ["--fuel", "20000000"]
    command.append(str(source))
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=30)
    return {"exit_code": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr[:8000],
            "stderr_truncated": len(result.stderr) > 8000,
            "wall_seconds": time.monotonic() - started}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    root = args.output.resolve()
    learned = json.loads((root / "report.json").read_text())
    program = tuple(learned["program"])
    inputs = json.loads((root / "held-inputs.json").read_text())
    expected = [sorted({value for value in values if value > 0})
                for values in inputs]
    before = digest(compiler)
    source = root / "curlee/check_clean_readings.curlee"
    if ((root / "curlee/clean_readings.curlee").read_text()
            != emit_curlee(program)
            or source.read_text() != curlee_harness(program, inputs,
                                                    expected)):
        raise ValueError("emitted routine or independent assertions changed")
    batches = root / "curlee/batches"
    batches.mkdir(exist_ok=True)
    records = []
    for start in range(0, len(inputs)):
        path = batches / f"cases-{start:03}.curlee"
        chunk = inputs[start:start + 1]
        path.write_text(curlee_harness(program, chunk,
                                      expected[start:start + 1]))
        checked = execute(compiler, path, "check")
        run = (execute(compiler, path, "run")
               if not checked["exit_code"] else {})
        ok = run.get("exit_code") == 0 and bool(re.search(
            r"curlee run: result 0\b", run.get("stdout", "")))
        records.append({"first_case": start, "cases": len(chunk),
                        "source_sha256": digest(path), "check": checked,
                        "run": run, "passed": ok})
        print(json.dumps({"first_case": start, "passed": ok}), flush=True)
        if not ok:
            break
    passed = (sum(row["cases"] for row in records) == len(inputs)
              and all(row["passed"] for row in records))
    wrong = [answers[:] for answers in expected[:1]]
    wrong[0] = [123]
    control = batches / "wrong_expected.curlee"
    control.write_text(curlee_harness(program, inputs[:1], wrong))
    negative = execute(compiler, control, "run")
    negative_rejected = (negative["exit_code"] == 0 and bool(re.search(
        r"curlee run: result 1\b", negative["stdout"])))
    after = digest(compiler)
    report = {
        "scope": "Actual CURLEE check and VM execution of the learned "
                 "processing routine on 100 fresh arrays. Capacity 256; "
                 "signed integers. JSON/file CLI is Python only.",
        "compiler_sha256_before": before,
        "compiler_sha256_after": after, "compiler_stable": before == after,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cases": len(inputs), "batch_size": 1, "records": records,
        "passed": passed,
        "wrong_expected_control": negative,
        "wrong_expected_rejected": negative_rejected,
    }
    (root / "curlee-execution.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("cases", "passed", "compiler_stable",
                       "wrong_expected_rejected")}), flush=True)
    return int(not passed or not negative_rejected or before != after)


if __name__ == "__main__":
    raise SystemExit(main())
