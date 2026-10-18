"""Read-only diagnosis of hard-task rich surrogate groups."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from rsi_programmer import author, rich
from rsi_programmer.probe import enumerate_programs
from rsi_programmer.scale import ScaleUp
from rsi_programmer.traces import TraceRecorder

HARD = "fixtures/rsi_hard_tasks.json"


def main() -> None:
    tasks = TraceRecorder(HARD).load_family()["tasks"]
    task_map = {str(task["id"]): task for task in tasks}
    exprs: dict[str, Any] = {}
    tries: dict[str, int] = {}
    reports: dict[str, dict[str, Any]] = {}
    negatives: dict[str, tuple[rich.Group, rich.Group]] = {}
    original_report = author.solution_sigs_by_size_report

    for task in tasks:
        tid = str(task["id"])
        rows, target, bound = ScaleUp._parts(task)
        expr, tried, _ = enumerate_programs(rows, target, bound)
        exprs[tid] = expr
        tries[tid] = tried
        captured: dict[str, dict[str, Any]] = {}

        def recording_report(
                in_rows: Any, in_target: Any, in_bound: int,
                candidate_cap: int = author.SOLUTION_CANDIDATE_CAP
                ) -> tuple[dict[int, set[tuple[int, ...]]],
                           dict[str, Any]]:
            completing, report = original_report(
                in_rows, in_target, in_bound, candidate_cap)
            captured["report"] = report
            return completing, report

        with patch.object(author, "solution_sigs_by_size_report",
                          side_effect=recording_report):
            neg12, negx = rich.hard_neg(rows, target, bound)
        reports[tid] = captured["report"]
        negatives[tid] = (neg12, negx)

    def cached_hard_neg(rows: Any, target: Any,
                        bound: int) -> tuple[rich.Group, rich.Group]:
        tid = next(
            str(task["id"]) for task in tasks
            if ScaleUp._parts(task) == (rows, target, bound))
        return negatives[tid]

    with patch.object(rich, "hard_neg", side_effect=cached_hard_neg):
        groups12, groupsx, selection = rich.family_pairs(
            task_map, exprs, tries, ScaleUp._parts)
    summary = []
    for index, task in enumerate(tasks):
        tid = str(task["id"])
        neg12, negx = negatives[tid]
        report = reports[tid]
        summary.append({
            "task": tid,
            "bound": ScaleUp._parts(task)[2],
            "cap": report["candidate_cap"],
            "candidate_signature_evaluations":
                report["candidate_signature_evaluations"],
            "reachability_truncated": report["truncated"],
            "retained_negative_pairs": len(neg12),
            "rich_negative_pairs": len(negx),
            "surrogate_group_size_after_selection_slice":
                len(groups12[index]),
            "rich_group_size_after_selection_slice": len(groupsx[index]),
            "selection_size": len(selection[index]),
            "expr_available": exprs[tid] is not None,
        })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
