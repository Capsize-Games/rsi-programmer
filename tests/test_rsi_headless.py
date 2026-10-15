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
        "violations", "lineage", "oracle_holds")
TOKENS = ("oracle", "held_out", "heldout", "promot", "ledger")


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

    def test_hands_off_green_neutral_or_better(self) -> None:
        """Solve parity at lower cost, zero bad executions."""
        res = self.result
        self.assertEqual(res["status"], "complete")
        self.assertGreaterEqual(res["solve_rate"],
                               res["baseline_solve_rate"])
        self.assertLess(res["cost_tried"], res["baseline_cost"])
        self.assertEqual(res["safety_failures"], 0)
        for decision in res["decisions"]:
            self.assertEqual(decision["sandbox"], "accepted")
            self.assertTrue(decision["solved"])
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
            self.assertTrue(record["solved"])
            self.assertEqual(len(record["lineage"]), 64)
        leaked = [p for p in Path(self.tmp, "box").rglob("*")
                  if any(tok in p.name for tok in TOKENS)]
        self.assertEqual(leaked, [])

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
        first = str(self.tasks[0]["id"])
        width = len(self.tasks[0]["train"])
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
        rest = [d for d in res["decisions"] if d["task"] != first]
        self.assertTrue(all(d["solved"] for d in rest))


if __name__ == "__main__":
    unittest.main()
