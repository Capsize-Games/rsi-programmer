#!/usr/bin/env python3
"""Enumerator baselines for the generated corpus (#5).

Slow gate only: solves every generated task with the enumerator
using each task's recorded IR features and reports per-family
solve-rate and cost. Kept out of `just test-fast`.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from rsi_programmer.probe import CONSTS, enumerate_programs
from rsi_programmer.probe import interpret, lower_to_dag

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = ("fixtures/rsi_gen_train.json",
            "fixtures/rsi_gen_novel.json")


def _flags(task: dict) -> dict:
    """Search flags matching the task's recorded features."""
    feats = set(task.get("features", []))
    consts = CONSTS if "const" in feats else ()
    return {"consts": consts, "branch": "branch" in feats}


class GeneratedBaselineTests(unittest.TestCase):
    def test_enumerator_baseline_per_family(self) -> None:
        """Every generated task solves within its bound."""
        for name in FAMILIES:
            with open(ROOT / name) as handle:
                tasks = json.load(handle)["tasks"]
            solved = 0
            cost = 0
            for task in tasks:
                rows = tuple((c[0], c[1]) for c in task["train"])
                target = tuple(c[2] for c in task["train"])
                bound = int(task.get("max_tree_ops", 4))
                expr, tried, _ = enumerate_programs(
                    rows, target, bound, **_flags(task))
                self.assertIsNotNone(expr, task["id"])
                assert expr is not None
                graph = lower_to_dag(expr)
                rows_all = [(c[0], c[1]) for c in
                            task["train"] + task["held_out"]]
                want_all = [c[2] for c in
                            task["train"] + task["held_out"]]
                got = [interpret(graph, x, y)
                       for x, y in rows_all]
                self.assertEqual(got, want_all, task["id"])
                solved += 1
                cost += tried
            rate = solved / len(tasks)
            print("%s solve=%.2f cost=%d tasks=%d"
                  % (name, rate, cost, len(tasks)))
            self.assertEqual(rate, 1.0)


if __name__ == "__main__":
    unittest.main()
