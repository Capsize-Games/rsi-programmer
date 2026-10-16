#!/usr/bin/env python3
"""Tests for the RSI W4 headless loop (#33)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.headless import HeadlessLoop
from rsi_programmer.traces import TraceRecorder

FIXTURE = str(Path(__file__).resolve().parent.parent
              / "fixtures" / "rsi_shadow_tasks.json")
KEYS = ("gen", "task", "tried", "solved", "sandbox",
        "violations", "lineage", "oracle_holds", "path",
        "author_verdict")
TOKENS = ("oracle", "held_out", "heldout", "promot", "ledger")
SOLVED = {"diff_squares", "twice_product"}


def _forging_hook(first: str, width: int) -> Any:
    """Forge outputs for one task; accept everything else."""

    def hook(task_id: str, graph: dict[str, Any] | None) -> Any:
        if task_id == first and graph is not None:
            return {"outputs": [999999] * width}
        return None

    return hook


class HeadlessLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Run one fully hands-off loop over the frozen family."""
        cls.tmp = tempfile.mkdtemp(prefix="rsi-w4-")
        cls.recorder = TraceRecorder(FIXTURE)
        family = cls.recorder.load_family()
        cls.tasks = family["tasks"]
        loop = HeadlessLoop(FIXTURE, cls.tmp + "/w4.jsonl",
                            cls.tmp + "/box", seed=1)
        cls.result = loop.run()

    @classmethod
    def _records(cls, name: str) -> list[dict[str, Any]]:
        """Reload JSONL decision records from disk."""
        path = Path(cls.tmp) / name
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def test_hands_off_author_only_honest_rate(self) -> None:
        """Author path 2/4 with no rescue, zero bad executions."""
        res = self.result
        self.assertEqual(res["status"], "complete")
        self.assertEqual(res["solve_rate"], 0.5)
        self.assertEqual(res["baseline_solve_rate"], 1.0)
        self.assertLess(res["cost_tried"], res["baseline_cost"])
        self.assertEqual(res["safety_failures"], 0)
        self.assertEqual(res["paths"], {"beam": 4})
        self.assertEqual(res["reject_histogram"],
                         {"no-authored-solution": 2, "solved": 2})
        got = {d["task"] for d in res["decisions"] if d["solved"]}
        self.assertEqual(got, SOLVED)
        for decision in res["decisions"]:
            self.assertEqual(decision["path"], "beam")
            if decision["solved"]:
                self.assertEqual(decision["sandbox"], "accepted")
                self.assertTrue(decision["oracle_holds"])
            else:
                self.assertEqual(decision["author_verdict"],
                                 "no-authored-solution")
        print("solve=%.2f base=%.2f cost=%d base=%d wall=%.1fs"
              % (res["solve_rate"], res["baseline_solve_rate"],
                 res["cost_tried"], res["baseline_cost"],
                 res["wall_s"]))

    def test_every_decision_traceable(self) -> None:
        """Decisions match JSONL records; no oracle files leak."""
        recs = self._records("w4.jsonl")
        self.assertEqual(len(recs), len(self.result["decisions"]))
        for decision, record in zip(self.result["decisions"], recs):
            for key in KEYS:
                self.assertIn(key, record)
                self.assertEqual(record[key], decision[key])
            if record["solved"]:
                self.assertEqual(len(record["lineage"]), 64)
            else:
                self.assertEqual(record["lineage"], "")
        leaked = [p for p in Path(self.tmp, "box").rglob("*")
                  if any(tok in p.name for tok in TOKENS)]
        self.assertEqual(leaked, [])

    def test_no_silent_rescue(self) -> None:
        """Enumerator-solved task stays unsolved with no fallback."""
        from rsi_programmer.author import beam_search
        from rsi_programmer.policy import SearchPolicy
        task = next(t for t in self.tasks
                    if t["id"] == "square_plus")
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        policy = SearchPolicy(seed=1)
        out = beam_search(policy, rows, target, bound, 10 ** 9,
                          "square_plus", fallback=False)
        self.assertFalse(out["solved"])
        self.assertEqual(out.get("path"), "beam")
        self.assertEqual(out["verdict"], "no-authored-solution")

    def test_fallback_debug_column_labelled(self) -> None:
        """Compare flag adds the enumerator column, never headline."""
        loop = HeadlessLoop(FIXTURE, self.tmp + "/cmp.jsonl",
                            self.tmp + "/cmpbox", seed=1,
                            compare_fallback=True)
        res = loop.run()
        self.assertTrue(res["fallback_compared"])
        self.assertEqual(res["fallback_solve_rate"], 1.0)
        self.assertEqual(res["solve_rate"], 0.5)

    def test_budget_exhaustion_stops_cleanly(self) -> None:
        """One-candidate budget stops the loop without failing."""
        loop = HeadlessLoop(FIXTURE, self.tmp + "/stop.jsonl",
                            self.tmp + "/stopbox", seed=1,
                            max_cands=1)
        res = loop.run()
        self.assertEqual(res["status"], "stopped-not-failed")
        self.assertEqual(len(res["decisions"]), 1)
        self.assertEqual(len(self._records("stop.jsonl")), 1)

    def test_tampered_candidate_rejected_loop_continues(self) -> None:
        """Forged mid-run claim is rejected; the loop finishes."""
        solved = next(t for t in self.tasks
                      if t["id"] in SOLVED)
        first = str(solved["id"])
        width = len(solved["train"])
        hook = _forging_hook(first, width)
        loop = HeadlessLoop(FIXTURE, self.tmp + "/tamper.jsonl",
                            self.tmp + "/tamperbox", seed=1)
        res = loop.run(hook=hook)
        self.assertEqual(res["status"], "complete")
        self.assertEqual(len(res["decisions"]), len(self.tasks))
        hit = [d for d in res["decisions"] if d["task"] == first][0]
        self.assertFalse(hit["solved"])
        self.assertEqual(hit["sandbox"], "rejected")
        self.assertIn("forged-completion", hit["violations"])
        rest = {d["task"] for d in res["decisions"]
                if d["task"] != first and d["solved"]}
        self.assertEqual(rest, SOLVED - {first})


if __name__ == "__main__":
    unittest.main()
