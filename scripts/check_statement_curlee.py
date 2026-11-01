"""Check and execute every generated statement program under CURLEE guard."""
from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import re

from rsi_programmer.statement_curriculum import oracle
from rsi_programmer.statement_source import curlee_harness, curlee_source
from scripts.check_useful_curlee import digest, execute


def passed(run: dict[str, object], code: int = 0) -> bool:
    return run.get("exit_code") == 0 and bool(re.search(
        rf"curlee run: result {code}\b", str(run.get("stdout", ""))))


def main(task_oracle: Callable[[str, list[int]], int] | None = None) -> int:
    task_oracle = oracle if task_oracle is None else task_oracle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compiler, root = args.compiler.resolve(), args.output.resolve()
    report = json.loads((root / "report.json").read_text())
    inputs = json.loads((root / "held-inputs.json").read_text())
    before = digest(compiler)
    rows = []
    control = None
    for attempt in report["rows"]:
        name, tokens = attempt["id"], attempt["chosen"]["tokens"]
        entry = {"id": name, "tokens": tokens, "records": [],
                 "accepted": False}
        if tokens is not None:
            tokens = tuple(tokens)
            folder = root / name / "curlee"
            if ((folder / "process.curlee").read_text()
                    != curlee_source(tokens)):
                raise ValueError("recorded program source changed")
            for index, values in enumerate(inputs):
                path = folder / f"case-{index:03}.curlee"
                path.write_text(curlee_harness(tokens, values,
                                               task_oracle(name, values)))
                check = execute(compiler, path, "check")
                run = (execute(compiler, path, "run")
                       if check["exit_code"] == 0 else {})
                entry["records"].append({"case": index, "check": check,
                                         "run": run, "passed": passed(run),
                                         "source_sha256": digest(path)})
                if check["exit_code"]:
                    break
            entry["accepted"] = len(entry["records"]) == len(inputs) and all(
                row["passed"] for row in entry["records"])
            if control is None and entry["accepted"]:
                path = folder / "wrong_expected.curlee"
                path.write_text(curlee_harness(tokens, [],
                                               task_oracle(name, []) + 1))
                run = execute(compiler, path, "run")
                control = {"task": name, "run": run,
                           "rejected": passed(run, 1)}
        rows.append(entry)
        print(json.dumps({"id": name, "accepted": entry["accepted"],
                          "checked_cases": len(entry["records"])}), flush=True)
    after = digest(compiler)
    result = {"scope": "Actual compiler check and VM execution for every "
                       "generated statement program, including wrong "
                       "attempts.",
              "program_origin": report.get("scope"),
              "compiler_before": before, "compiler_after": after,
              "compiler_stable": before == after, "rows": rows,
              "wrong_expected_control": control,
              "tasks": len(rows), "accepted": sum(row["accepted"]
                                                    for row in rows)}
    (root / "curlee-execution.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tasks": len(rows), "accepted": result["accepted"],
                      "compiler_stable": before == after}), flush=True)
    return int(before != after or control is None or not control["rejected"])


if __name__ == "__main__":
    raise SystemExit(main())
