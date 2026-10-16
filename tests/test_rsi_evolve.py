#!/usr/bin/env python3
"""Tests for the RSI W2 search-control policy loop (#31)."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from typing import Any
from rsi_programmer.author import beam_search, refine_author
from rsi_programmer.evolve import PolicyLoop, guided_search
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.proxy import ProxyScreen
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


class ScreenedLoopTests(unittest.TestCase):
    NOVEL = str(Path(__file__).resolve().parent.parent
                / "fixtures" / "rsi_novel_tasks.json")

    @classmethod
    def setUpClass(cls) -> None:
        """Baseline and screen-promoted loops on one seed."""
        tmp = tempfile.mkdtemp(prefix="rsi-w2s-")
        base = PolicyLoop(FIXTURE, tmp + "/base.db")
        cls.base = base.run(1)
        base.close()
        screen = ProxyScreen(FIXTURE, cls.NOVEL)
        loop = PolicyLoop(FIXTURE, tmp + "/screen.db",
                          screen=screen)
        cls.screened = loop.run(1)
        loop.close()

    def _evaluated(self, res: dict[str, Any]) -> int:
        """Archive entries actually fully evaluated."""
        return sum(1 for e in res["archive"]
                   if not e.get("screen_skipped"))

    def test_promotion_evaluates_no_more(self) -> None:
        """Screened loop never pays for extra full evals."""
        self.assertEqual(len(self.screened["archive"]),
                         len(self.base["archive"]))
        self.assertLessEqual(self._evaluated(self.screened),
                             self._evaluated(self.base))
        print("screened evals=%d base evals=%d skipped=%s"
              % (self._evaluated(self.screened),
                 self._evaluated(self.base),
                 [s["skipped"]
                  for s in self.screened["screened"]]))

    def test_skips_carry_parent_fitness(self) -> None:
        """Skipped kids inherit, evaluated kids measure."""
        for res in (self.base, self.screened):
            by_id = {e["policy_id"]: e for e in res["archive"]}
            for entry in res["archive"]:
                self.assertIn("screen_skipped", entry)
                if entry["screen_skipped"]:
                    parent = by_id[entry["parent_id"]]
                    self.assertEqual(entry["fitness"],
                                     parent["fitness"])
        self.assertFalse(any(e["screen_skipped"]
                             for e in self.base["archive"]))

    def test_screen_order_and_pact_reported(self) -> None:
        """Every screened generation names order and pact."""
        self.assertEqual(len(self.screened["screened"]), 2)
        for rep in self.screened["screened"]:
            self.assertEqual(len(rep["order"]), 4)
            self.assertIn("agreement", rep["pact"])
            self.assertIn("pairs", rep["pact"])


class RefinableLoopTests(unittest.TestCase):
    BUDGET = 3000

    @classmethod
    def setUpClass(cls) -> None:
        """One screened generation; elite refined on shadow."""
        tmp = tempfile.mkdtemp(prefix="rsi-w2r-")
        screen = ProxyScreen(FIXTURE, FIXTURE)
        loop = PolicyLoop(FIXTURE, tmp + "/r.db", screen=screen)
        cls.res = loop.run(1)
        loop.close()
        fam = TraceRecorder(FIXTURE).load_family()["tasks"]
        cls.rows = [(str(t["id"]),
                     tuple((c[0], c[1]) for c in t["train"]),
                     tuple(c[2] for c in t["train"]),
                     int(t.get("max_tree_ops", 4))) for t in fam]
        best = max(cls.res["archive"],
                   key=lambda e: (e["fitness"]["correctness"],
                                  e["fitness"]["efficiency"]))
        print("elite: id=%s operator=%s parent=%s corr=%.2f"
              % (best["policy_id"], best["operator"],
                 best["parent_id"],
                 best["fitness"]["correctness"]))
        cls.elite = SearchPolicy(seed=0)
        cls.elite.restore({"spec": best["spec"],
                           "weights": best["weights"]})
        cls.weights, _ = refine_author(cls.elite, cls.rows,
                                       seed=0)
        author = next(e for e in cls.res["archive"]
                      if e["operator"] == "author"
                      and e["generation"] == 0)
        cls.author = SearchPolicy(seed=0)
        cls.author.restore({"spec": author["spec"],
                            "weights": author["weights"]})
        cls.author_weights, _ = refine_author(cls.author,
                                              cls.rows, seed=0)

    def test_founders_include_author_member(self) -> None:
        """Author-trained founder joins the gene pool."""
        ops = {e["operator"] for e in self.res["archive"]
               if e["generation"] == 0}
        self.assertIn("author", ops)
        self.assertIn("train", ops)
        self.assertIn("null", ops)

    def _authored(self, policy: SearchPolicy,
                  weights: list[float]) -> tuple[int, int]:
        """Refined-policy shadow solves and tried."""
        rec = TraceRecorder(FIXTURE)
        fam = rec.load_family()["tasks"]
        solved, tried = 0, 0
        for task in fam:
            rows = tuple((c[0], c[1]) for c in task["train"])
            target = tuple(c[2] for c in task["train"])
            bound = int(task.get("max_tree_ops", 4))
            out = beam_search(policy, rows, target, bound,
                              self.BUDGET, str(task["id"]),
                              fallback=False, weights=weights)
            tried += out["tried"]
            if out["solved"] and rec.oracle_holds(task,
                                                  out["graph"]):
                solved += 1
        return solved, tried

    def test_elite_refines_and_reports_solves(self) -> None:
        """Elite plus EXTRA refinement authors shadow tasks."""
        fam = TraceRecorder(FIXTURE).load_family()["tasks"]
        elite_solved, elite_tried = self._authored(
            self.elite, self.weights)
        auth_solved, auth_tried = self._authored(
            self.author, self.author_weights)
        self.assertLessEqual(elite_tried,
                             self.BUDGET * len(fam))
        self.assertLessEqual(auth_tried, self.BUDGET * len(fam))
        print("elite+refine: solved=%d/%d tried=%d"
              % (elite_solved, len(fam), elite_tried))
        print("author+refine: solved=%d/%d tried=%d"
              % (auth_solved, len(fam), auth_tried))
