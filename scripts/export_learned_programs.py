"""Export every returned ANN development program to real target source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rsi_programmer.construction_check import check_graph
from rsi_programmer.target_source import emit_function

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    corpus = json.loads((ROOT / "fixtures/construction_corpus_v1.json")
                        .read_text())
    tasks = {task["id"]: task for task in corpus["splits"]["dev"]}
    rows = report["rows"]["trained_ann_beam"]
    if len(rows) != len(tasks) or {row["id"] for row in rows} != set(tasks):
        raise ValueError("complete development attempts required")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "python").mkdir()
    (args.output / "curlee").mkdir()
    functions: list[str] = []
    cases = []
    manifest = []
    for row in rows:
        task = tasks[row["id"]]
        entry = {"id": task["id"], "generated": row["graph"] is not None}
        if row["graph"] is None:
            manifest.append(entry)
            continue
        check = check_graph(
            row["graph"], tuple(tuple(r) for r in task["observed"]),
            tuple(tuple(r) for r in task["held"]),
            tuple(tuple(term) for term in task["generator"]["polynomial"]),
        )
        name = "synth_" + task["id"]
        py = emit_function(row["graph"], "python", name)
        curlee = emit_function(row["graph"], "curlee", name)
        (args.output / "python" / (name + ".py")).write_text(py)
        functions.append(py)
        all_rows = task["observed"] + task["held"]
        harness = ["", "fn main() -> Int {"]
        for number, (x, y, expected) in enumerate(all_rows, 1):
            harness += [f"    if ({name}({x}, {y}) != {expected}) {{",
                        f"        return {number};", "    }"]
        harness += ["    return 0;", "}", ""]
        (args.output / "curlee" / (name + ".curlee")).write_text(
            curlee + "\n".join(harness)
        )
        cases.append({"name": name, "observed": task["observed"],
                      "held": task["held"]})
        entry.update({"source_name": name, "ir_checker": check,
                      "python_sha256": hashlib.sha256(
                          py.encode()).hexdigest(),
                      "curlee_function_sha256": hashlib.sha256(
                          curlee.encode()).hexdigest()})
        manifest.append(entry)
    (args.output / "programs.py").write_text("\n".join(functions))
    (args.output / "cases.json").write_text(json.dumps(cases) + "\n")
    result = {"scope": "Learned arithmetic programs compiled to target "
                       "source; "
                       "language execution results are separate.",
              "attempts": len(rows), "generated": len(cases),
              "missing": len(rows) - len(cases), "programs": manifest}
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2)
                                              + "\n")
    print(json.dumps({key: result[key] for key in
                      ("attempts", "generated", "missing")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
