#!/usr/bin/env python3
"""Tests for the RSI W2 search-control policy loop (#31)."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from typing import Any
from rsi_programmer.evolve import PolicyLoop, guided_search
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.traces import TraceRecorder
from rsi_programmer.probe import enumerate_programs
from rsi_programmer.probe import lower_to_dag
FIXTURE = str(Path(__file__).resolve().parent.parent
              / "fixtures" / "rsi_shadow_tasks.json")
class PolicyLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Run the W2 pipeline once for every test."""
        cls.tmp = tempfile.mkdtemp(prefix="rsi-w2-")
        cls.recorder = TraceRecorder(FIXTURE)
        cls.family = cls.recorder.load_family()["tasks"]
        cls.base_rate, cls.base_cost = cls._probe_rate("held_out")
        cls.budget = sum(r["tried"] for r in
                         cls.recorder.record_all()["records"])
        cls.loop = PolicyLoop(FIXTURE, cls.tmp + "/w2.db")
        cls.result = cls.loop.run(2)
        cls.best = cls._best_policy()
        cls.held = cls._heldout_runs(cls.best)
    @classmethod
    def _probe_rate(cls, split: str) -> tuple[float, int]:
        """Probe solve rate and cost on one frozen split."""
        solved, cost = 0, 0
        for task in cls.family:
            rows = tuple((c[0], c[1]) for c in task[split])
            target = tuple(c[2] for c in task[split])
            bound = int(task.get("max_tree_ops", 4))
            expr, tried, _ = enumerate_programs(rows, target,
                                               bound)
            cost += tried
            graph = lower_to_dag(expr) if expr is not None else None
            if graph is not None and cls.recorder.oracle_holds(
                    task, graph):
                solved += 1
        return solved / len(cls.family), cost
    @classmethod
    def _best_policy(cls) -> SearchPolicy:
        """Live policy restored from the best archive entry."""
        scored = sorted(cls.result["archive"],
                        key=lambda e: (
                            e["fitness"]["correctness"],
                            e["fitness"]["efficiency"],
                            e["fitness"]["recovery"]),
                        reverse=True)
        policy = SearchPolicy(seed=0)
        policy.restore({"spec": scored[0]["spec"],
                        "weights": scored[0]["weights"]})
        return policy
    @classmethod
    def _heldout_runs(cls, policy: SearchPolicy
                       ) -> list[dict[str, Any]]:
        """Policy-guided searches over the held-out family."""
        outs = []
        for task in cls.family:
            rows = tuple((c[0], c[1]) for c in task["held_out"])
            target = tuple(c[2] for c in task["held_out"])
            bound = int(task.get("max_tree_ops", 4))
            out = guided_search(policy, rows, target, bound,
                                cls.budget, str(task["id"]))
            out["task"] = task
            outs.append(out)
        return outs
    def _check_lineage(self, entries: list[dict[str, Any]]
                       ) -> None:
        """Parents resolve; entries carry fitness and weights."""
        ids = {e["policy_id"] for e in entries}
        for entry in entries:
            self.assertIn(entry["parent_id"], ids | {""})
            for key in ("fitness", "weights", "spec",
                        "operator", "generation"):
                self.assertIn(key, entry)
    def test_heldout_matches_baseline_at_lower_cost(self) -> None:
        """Guided search keeps the solve rate for less work."""
        solved, cost = 0, 0
        for out in self.held:
            cost += out["tried"]
            graph = out["graph"]
            if out["solved"] and self.recorder.oracle_holds(
                    out["task"], graph):
                solved += 1
        rate = solved / len(self.held)
        self.assertGreaterEqual(rate, self.base_rate)
        self.assertLess(cost, self.base_cost)
    def test_two_generations_show_no_regression(self) -> None:
        """Elite archive fitness never steps back."""
        first = self.result["history"][0]
        last = self.result["history"][-1]
        for key in ("correctness", "efficiency", "recovery"):
            self.assertGreaterEqual(last[key], first[key])
        for entry in self.result["history"]:
            self.assertTrue(entry["gates_pass"])
    def test_every_rejection_carries_verdict(self) -> None:
        """No silent drops: each rejection names its verdict."""
        total = 0
        for out in self.held:
            for item in out["rejected"]:
                total += 1
                self.assertTrue(item["verdict"])
                self.assertIn("task", item)
                self.assertIn("op", item)
        self.assertGreater(total, 0)
    def test_archive_lineage_round_trips(self) -> None:
        """Lineage survives close and reopen through intents."""
        before = self.loop.load_archive()
        self.assertEqual(len(before), len(self.result["archive"]))
        self._check_lineage(before)
        path = self.tmp + "/w2.db"
        self.loop.close()
        reopened = PolicyLoop(FIXTURE, path)
        after = reopened.load_archive()
        reopened.close()
        self.assertEqual(len(after), len(before))
        self._check_lineage(after)
        self.assertEqual(after[0]["policy_id"],
                         before[0]["policy_id"])
    def test_snapshot_restore_turnover(self) -> None:
        """Temporal state round-trips across generation turnover."""
        policy = SearchPolicy(seed=7)
        feats = [[float(i % 3 == j) for j in range(3)]
                 + [0.0] * 9 for i in range(6)]
        policy.scores(feats)
        snap = policy.snapshot()
        self.assertIsNotNone(snap["temporal"])
        mid = policy.scores(feats)
        policy.perturb(0.5, 99)
        policy.restore(snap)
        self.assertEqual(mid, policy.scores(feats))
