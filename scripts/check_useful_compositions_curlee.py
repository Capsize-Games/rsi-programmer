"""Execute every generated suite composition using guarded CURLEE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from rsi_programmer.utility_harness import curlee_harness
from rsi_programmer.utility_source import emit_curlee
from rsi_programmer.utility_suite import oracle
from scripts.check_useful_curlee import digest, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compiler, root = args.compiler.resolve(), args.output.resolve()
    report = json.loads((root / "report.json").read_text())
    inputs = json.loads((root / "held-inputs.json").read_text())
    before = digest(compiler)
    rows = []
    for attempt in report["rows"]:
        name, program = attempt["id"], attempt["program"]
        entry = {"id": name, "program": program, "records": [],
                 "accepted": False, "previously_tested":
                 attempt["previously_tested"]}
        if program is not None:
            program = tuple(program)
            folder = root / name / "curlee"
            if (folder / "process.curlee").read_text() != emit_curlee(program):
                raise ValueError("recorded program source changed")
            for index, values in enumerate(inputs):
                path = folder / f"case-{index:03}.curlee"
                path.write_text(curlee_harness(
                    program, [values], [oracle(name, values)]))
                check = execute(compiler, path, "check")
                run = (execute(compiler, path, "run")
                       if check["exit_code"] == 0 else {})
                passed = run.get("exit_code") == 0 and bool(re.search(
                    r"curlee run: result 0\b", run.get("stdout", "")))
                entry["records"].append({"case": index, "check": check,
                                         "run": run, "passed": passed,
                                         "source_sha256": digest(path)})
                if check["exit_code"]:
                    break
            entry["accepted"] = len(entry["records"]) == len(inputs) and all(
                row["passed"] for row in entry["records"])
        rows.append(entry)
        print(json.dumps({"id": name, "accepted": entry["accepted"],
                          "checked_cases": len(entry["records"])}), flush=True)
    after = digest(compiler)
    result = {"scope": "Actual compiler check and VM execution for all "
                       "generated compositions, including incorrect attempts.",
              "compiler_before": before, "compiler_after": after,
              "compiler_stable": before == after, "rows": rows,
              "tasks": len(rows), "accepted": sum(row["accepted"]
                                                    for row in rows),
              "new_accepted": sum(row["accepted"] for row in rows
                                  if not row["previously_tested"])}
    (root / "curlee-execution.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("tasks", "accepted", "new_accepted",
                       "compiler_stable")}), flush=True)
    return int(before != after)


if __name__ == "__main__":
    raise SystemExit(main())
