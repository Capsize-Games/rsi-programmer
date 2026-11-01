"""Execute frozen statement choices without loading a learner or solver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_statement_decisions import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    path = root / "report.json"
    report = json.loads(path.read_text())
    if report.get("execution_pending") is not True:
        raise ValueError("trial does not contain pending frozen choices")
    held = json.loads((root / "held-inputs.json").read_text())
    rows = evaluate(root, report["rows"], held)
    report.update(rows=rows, execution_pending=False,
                  accepted=sum(row["accepted"] for row in rows))
    roles = {row["role"] for row in rows if "role" in row}
    if roles:
        report["accepted_by_role"] = {
            role: sum(row["accepted"] for row in rows if row["role"] == role)
            for role in sorted(roles)}
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"accepted": report["accepted"], "tasks": len(rows),
                      "by_role": report.get("accepted_by_role")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
