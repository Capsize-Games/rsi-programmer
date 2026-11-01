"""Recheck early development graphs and counts without fitting a learner."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from rsi_programmer.construction_check import check_graph

ROOT = Path(__file__).resolve().parents[1]


def verify_report(report: dict[str, Any], tasks: dict[str, dict[str, Any]]
                  ) -> dict[str, int]:
    """Verify complete dev coverage and every returned graph and aggregate.

    This checks correctness/coverage, not neural action probabilities,
    learning-rule replay, timing accuracy or unseen final performance.
    """
    verified = graphs = 0
    if set(report["controls"]) != set(report["rows"]):
        raise ValueError("control and row sets differ")
    for name, rows in report["rows"].items():
        identities = [row["id"] for row in rows]
        if len(identities) != len(tasks) or set(identities) != set(tasks):
            raise ValueError("development task coverage differs")
        correct = 0
        by_size: dict[str, list[int]] = {}
        for row in rows:
            task = tasks[row["id"]]
            observed = tuple(tuple(item) for item in task["observed"])
            held = tuple(tuple(item) for item in task["held"])
            identity = tuple(tuple(item) for item in
                             task["generator"]["polynomial"])
            check = check_graph(row["graph"], observed, held, identity, 4)
            if check != row["checker"] or check["accepted"] != row["accepted"]:
                raise ValueError("saved graph verdict differs")
            correct += int(check["accepted"])
            size = str(task["generator"]["minimal_tree_ops"])
            by_size.setdefault(size, [0, 0])
            by_size[size][0] += int(check["accepted"])
            by_size[size][1] += 1
            graphs += int(row["graph"] is not None)
            verified += 1
        summary = report["controls"][name]
        if (summary["accepted"] != correct or summary["tasks"] != len(tasks)
                or summary["by_minimal_tree_ops"] != by_size):
            raise ValueError("saved aggregate differs")
    return {"verified_rows": verified, "returned_graphs": graphs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    corpus = json.loads((ROOT / "fixtures/construction_corpus_v1.json")
                        .read_text())
    tasks = {task["id"]: task for task in corpus["splits"]["dev"]}
    for path in args.reports:
        report = json.loads(path.read_text())
        result = verify_report(report, tasks)
        corruptions = []
        bad = copy.deepcopy(report)
        control = next(iter(bad["controls"]))
        bad["controls"][control]["accepted"] += 1
        corruptions.append(bad)
        bad = copy.deepcopy(report)
        control = next(iter(bad["rows"]))
        bad["rows"][control].pop()
        corruptions.append(bad)
        bad = copy.deepcopy(report)
        control = next(iter(bad["rows"]))
        bad["rows"][control][0]["accepted"] = not bad["rows"][control][0][
            "accepted"
        ]
        corruptions.append(bad)
        for bad in corruptions:
            try:
                verify_report(bad, tasks)
            except ValueError:
                continue
            raise ValueError("corrupted report was accepted")
        print(json.dumps({"report": str(path), **result,
                          "corruptions_rejected": len(corruptions)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
