#!/usr/bin/env python3
"""Oracle-rejection retry past near-misses (#36)."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.evolve import OracleHook, guided_search
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.traces import TraceRecorder
from rsi_programmer.probe import interpret

HARD = str(Path(__file__).resolve().parent.parent
           / "fixtures" / "rsi_hard_tasks.json")
SHADOW = str(Path(__file__).resolve().parent.parent
           / "fixtures" / "rsi_shadow_tasks.json")
TRUE_RULE = "f(x, y) = (x - y) * (x - y) * (x - y) + (x + y)"
PROBES = ((20, -7), (-20, 11), (100, 1), (0, 5), (-8, -8))
SEED = 7


def rule_distractor(x: int, y: int) -> int:
    """Closed form of the true 7-op distractor rule."""
    return (x - y) ** 3 + (x + y)


def accept_all(graph: dict[str, Any]) -> bool:
    """Permissive hook: exercise the oracle path, veto nothing."""
    return graph is not None


class RetryTests(unittest.TestCase):
    out: dict[str, Any] = {}

    @classmethod
    def setUpClass(cls) -> None:
        """One shared distractor search; ~15 minutes, run once."""
        rec = TraceRecorder(HARD)
        fam = rec.load_family()["tasks"]
        task = next(t for t in fam if t["id"] == "distractor_cube")
        train = task["train"]
        rows = tuple((c[0], c[1]) for c in train)
        target = tuple(c[2] for c in train)
        bound = int(task.get("max_tree_ops", 4))
        policy = SearchPolicy(seed=SEED)

        def holds(graph: dict[str, Any]) -> bool:
            return rec.oracle_holds(task, graph)

        cls.task = task
        cls.out = guided_search(policy, rows, target, bound,
                                10 ** 9, "distractor_cube",
                                oracle=holds)
        print("distractor tried=%d solved=%s"
              % (cls.out["tried"], cls.out["solved"]))

    def test_retry_solves_distractor_with_true_rule(self) -> None:
        """Retry passes the near-miss to the true 7-op rule."""
        self.assertEqual(self.task["task"], TRUE_RULE)
        self.assertTrue(self.out["solved"])
        graph = self.out["graph"]
        self.assertIsNotNone(graph)
        rec = TraceRecorder(HARD)
        self.assertTrue(rec.oracle_holds(self.task, graph))
        for row in self.task["train"] + self.task["held_out"]:
            self.assertEqual(interpret(graph, row[0], row[1]),
                             rule_distractor(row[0], row[1]))
        for x, y in PROBES:
            self.assertEqual(interpret(graph, x, y),
                             rule_distractor(x, y))

    def test_near_miss_exclusion_logged(self) -> None:
        """Each oracle veto names its signature class."""
        vetoes = [r for r in self.out["rejected"]
                  if r["verdict"] == "oracle-rejected"]
        self.assertGreater(len(vetoes), 0)
        self.assertIn("add", {v["op"] for v in vetoes})
        for item in vetoes:
            self.assertEqual(item["task"], "distractor_cube")
            self.assertTrue(item["op"])
            self.assertTrue(item["signature"])
        for item in self.out["rejected"]:
            self.assertTrue(item["verdict"])

    def test_hook_never_changes_rankings(self) -> None:
        """Post-hoc filter never leaks into search behavior."""
        rec = TraceRecorder(SHADOW)
        task = rec.load_family()["tasks"][0]
        train = task["train"]
        rows = tuple((c[0], c[1]) for c in train)
        target = tuple(c[2] for c in train)
        bound = int(task.get("max_tree_ops", 4))
        plain = guided_search(SearchPolicy(seed=SEED), rows,
                              target, bound, 10 ** 9)
        held = guided_search(SearchPolicy(seed=SEED), rows,
                             target, bound, 10 ** 9,
                             oracle=accept_all)
        self.assertEqual(plain["solved"], held["solved"])
        self.assertEqual(plain["tried"], held["tried"])
        self.assertEqual(plain["graph"], held["graph"])
        self.assertEqual(plain["rejected"], held["rejected"])


if __name__ == "__main__":
    unittest.main()
