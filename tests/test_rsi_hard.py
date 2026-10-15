#!/usr/bin/env python3
"""Harder frozen RSI families: depth 5-7, sub-heavy, distractor (#34)."""
from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Callable

from rsi_programmer.traces import TraceRecorder
from rsi_programmer.probe import interpret

FIXTURE = (Path(__file__).resolve().parent.parent
           / "fixtures" / "rsi_hard_tasks.json")
W1_TRIED = 6807
Rule = Callable[[int, int], int]


def rule_shift_square(x: int, y: int) -> int:
    """Closed form for deep_shift_square (6 tree ops)."""
    return (x - y - y) * (x - y - y) + x


def rule_sub_square(x: int, y: int) -> int:
    """Closed form for sub_square_diff (5 tree ops)."""
    return (x - y) * (x - y) - (x + y)


def rule_mixed7(x: int, y: int) -> int:
    """Closed form for mixed_depth7 (7 tree ops)."""
    return ((x + y) * (x - y) + (x - y)) * (x + y)


def rule_distractor(x: int, y: int) -> int:
    """Closed form for distractor_cube (true rule, 7 ops)."""
    return (x - y) ** 3 + (x + y)


RULES: dict[str, Rule] = {
    "deep_shift_square": rule_shift_square,
    "sub_square_diff": rule_sub_square,
    "mixed_depth7": rule_mixed7,
    "distractor_cube": rule_distractor,
}

NEAR_MISS: dict[str, Any] = {
    "input_types": {"x": "Int", "y": "Int"},
    "nodes": [{"op": "add", "type": "Int", "args": ["x", "x"]}],
    "output": 0,
}


class HardFamilyTests(unittest.TestCase):
    recorder: TraceRecorder
    tasks: dict[str, Any]
    report: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.recorder = TraceRecorder(str(FIXTURE))
        family = cls.recorder.load_family()
        cls.tasks = {t["id"]: t for t in family["tasks"]}
        cls.report = cls.recorder.record_all()

    def test_ids_bounds_and_depths(self) -> None:
        self.assertEqual(set(self.tasks), set(RULES))
        bounds = set()
        widest = 0
        for task in self.tasks.values():
            bound = int(task["max_tree_ops"])
            self.assertGreaterEqual(bound, 5)
            self.assertLessEqual(bound, 7)
            bounds.add(bound)
            for row in task["train"] + task["held_out"]:
                widest = max(widest, abs(row[0]), abs(row[1]))
        self.assertEqual(bounds, {5, 6, 7})
        self.assertGreaterEqual(widest, 12)

    def test_digest_stable(self) -> None:
        seen = set()
        for task in self.tasks.values():
            digest = self.recorder.split_hash(task)
            self.assertEqual(digest, task["split_sha256"])
            self.assertEqual(digest, self.recorder.split_hash(task))
            seen.add(digest)
        self.assertEqual(len(seen), len(self.tasks))

    def test_expectations_recomputed(self) -> None:
        for tid, task in self.tasks.items():
            rule = RULES[tid]
            for row in task["train"] + task["held_out"]:
                self.assertEqual(row[2], rule(row[0], row[1]))

    def test_enumerator_clearly_expensive(self) -> None:
        by_task = {r["task"]: r for r in self.report["records"]}
        for tid in ("deep_shift_square", "mixed_depth7"):
            tried = int(by_task[tid]["tried"])
            print("%s tried=%d ratio=%.1fx"
                  % (tid, tried, tried / W1_TRIED))
            self.assertGreaterEqual(tried, 10 * W1_TRIED)

    def test_near_miss_passes_train_fails_held_out(self) -> None:
        task = self.tasks["distractor_cube"]
        rule = RULES["distractor_cube"]
        for row in task["train"]:
            self.assertEqual(interpret(NEAR_MISS, row[0], row[1]),
                             rule(row[0], row[1]))
        misses = 0
        for row in task["held_out"]:
            if interpret(NEAR_MISS, row[0], row[1]) != rule(row[0],
                                                            row[1]):
                misses += 1
        self.assertEqual(misses, len(task["held_out"]))
        by_task = {r["task"]: r for r in self.report["records"]}
        self.assertFalse(by_task["distractor_cube"]["solved"])

    def test_baseline_solve_rate_and_cost(self) -> None:
        baseline = self.report["baseline"]
        self.assertEqual(baseline["tasks"], 4)
        self.assertEqual(baseline["solved"], 3)
        self.assertAlmostEqual(baseline["solve_rate"], 0.75)
        for record in self.report["records"]:
            print("task=%s solved=%s tried=%d wall=%.2fs"
                  % (record["task"], record["solved"],
                     record["tried"], record["wall_s"]))
        self.assertGreater(baseline["total_wall_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
