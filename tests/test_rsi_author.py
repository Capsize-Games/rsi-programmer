#!/usr/bin/env python3
"""SNN-ranked authoring and search contracts (#37)."""
from __future__ import annotations
import time
import unittest
from pathlib import Path
from typing import Any
from rsi_programmer.actions import ActionSpace
from rsi_programmer.author import TaskRow, author_search, emit_edit
from rsi_programmer.author import refine_author, train_author
from rsi_programmer.evolve import guided_search
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.traces import TraceRecorder
from rsi_programmer.probe import validate_graph
FIXTURE = (Path(__file__).resolve().parent.parent
           / "fixtures" / "rsi_shadow_tasks.json")
SEEDS = (0, 1, 2)
EPOCHS = 100
BUDGET = 10 ** 9
TOPKS = (3, 6)
def _load_tasks() -> list[TaskRow]:
    """Frozen W1 tasks as search-ready tuples."""
    rec = TraceRecorder(str(FIXTURE))
    out: list[TaskRow] = []
    for task in rec.load_family()["tasks"]:
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        out.append((str(task["id"]), rows, target,
                    int(task.get("max_tree_ops", 4))))
    return out
def _tag(row: dict[str, Any]) -> str:
    """Compare solves, then canonical candidate-signature work."""
    if row["au_solved"] and (not row["rank_solved"]
                             or row["au_work"] < row["rank_work"]):
        return "win"
    if row["rank_solved"] and (not row["au_solved"]
                               or row["au_work"] > row["rank_work"]):
        return "loss"
    return "tie"
def _run_seed(seed: int, tasks: list[TaskRow]) -> list[dict[str, Any]]:
    """Train, refine, and compare author vs rank one seed."""
    policy = SearchPolicy(seed=seed)
    train_author(policy, str(FIXTURE), epochs=EPOCHS)
    rows: list[dict[str, Any]] = []
    for topk in TOPKS:
        weights, _ = refine_author(policy, tasks, seed=seed, topk=topk)
        for tid, trows, target, bound in tasks:
            ranked = guided_search(policy, trows, target, bound, BUDGET, tid)
            authored = author_search(policy, trows, target, bound,
                                     BUDGET, tid, topk, weights,
                                     fallback=False)
            rows.append({"seed": seed, "topk": topk, "task": tid,
                         "rank_tried": ranked["tried"],
                         "rank_work": ranked["signature_evaluations"],
                         "rank_solved": ranked["solved"],
                         "au_tried": authored["tried"],
                         "au_work": authored["signature_evaluations"],
                         "au_solved": authored["solved"],
                         "au_log": authored["rejected"],
                         "au_graph": authored["graph"]})
    return rows
class AuthorTests(unittest.TestCase):
    rows: list[dict[str, Any]] = []
    @classmethod
    def setUpClass(cls) -> None:
        """Train three heads; compare on frozen tasks once."""
        cls.start = time.perf_counter()
        cls.tasks = _load_tasks()
        fam = TraceRecorder(str(FIXTURE)).load_family()["tasks"]
        cls.by_id = {t["id"]: t for t in fam}
        cls.rows = []
        for seed in SEEDS:
            cls.rows += _run_seed(seed, cls.tasks)
        cls.wall_s = time.perf_counter() - cls.start
    def test_honest_report_complete(self) -> None:
        """Every row tagged; authoring non-vacuous."""
        tags = [_tag(r) for r in self.rows]
        for row, tag in zip(self.rows, tags):
            print("s%d k%d %-12s rank_work=%5d/%s au_work=%5d/%s %s"
                  % (row["seed"], row["topk"], row["task"],
                     row["rank_work"], row["rank_solved"],
                     row["au_work"], row["au_solved"], tag))
        print("wins=%d ties=%d losses=%d wall=%.1fs"
              % (tags.count("win"), tags.count("tie"),
                 tags.count("loss"), self.wall_s))
        self.assertEqual(len(tags), 24)
        self.assertTrue(any(r["au_solved"] for r in self.rows))
    def test_every_authored_edit_verdict_carrying(self) -> None:
        """Each emitted edit names its validator verdict."""
        for row in self.rows:
            log = row["au_log"]
            self.assertGreater(len(log), 0)
            for entry in log:
                self.assertEqual(set(entry),
                                 {"op", "signature", "task", "verdict"})
                if entry["verdict"] not in ("", "signature-mismatch"):
                    self.assertTrue(entry["verdict"].startswith(
                        "validator:"))
            blanks = [e for e in log if e["verdict"] == ""]
            self.assertEqual(len(blanks), 1 if row["au_solved"] else 0)
    def test_solved_graphs_pass_frozen_oracle(self) -> None:
        """Authored solutions validate and meet expectations."""
        rec = TraceRecorder(str(FIXTURE))
        for row in self.rows:
            if not row["au_solved"]:
                continue
            validate_graph(row["au_graph"])
            self.assertTrue(rec.oracle_holds(self.by_id[row["task"]],
                                             row["au_graph"]))
    def test_malformed_edits_rejected_pre_execution(self) -> None:
        """Validator gate rejects malformed edits pre-execution."""
        space = ActionSpace()
        empty: dict[str, Any] = {"input_types": {"x": "Int", "y": "Int"},
                                 "nodes": [], "output": "x"}
        bad = [{"kind": "apply", "op": "div", "left": "x", "right": "y"},
               {"kind": "apply", "op": "add", "left": 9, "right": "y"},
               {"kind": "input", "ref": "z"}, {"kind": "jump", "ref": "x"},
               {"kind": "apply", "op": "add", "left": "x"},
               {"kind": "apply", "op": "add", "left": 0, "right": "y"}]
        for action in bad:
            graph, rec, ok = emit_edit(space, dict(empty), action, "probe", ())
            self.assertFalse(ok)
            self.assertTrue(rec["verdict"].startswith("validator:"))
            self.assertEqual(graph, empty)
        good = {"kind": "apply", "op": "add", "left": "x", "right": "y"}
        graph, rec, ok = emit_edit(space, dict(empty), good,
                                   "probe", (1, 2))
        self.assertTrue(ok)
        validate_graph(graph)
    def test_enumerator_fallback_intact(self) -> None:
        """Zero scan budget leaves both author and fallback unsolved."""
        task = self.by_id["twice_product"]
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        policy = SearchPolicy(seed=0)
        out = author_search(policy, rows, target, bound, 0,
                            "twice_product")
        plain = guided_search(policy, rows, target, bound, 0,
                              "twice_product")
        self.assertEqual(out["path"], "fallback")
        self.assertEqual(out["solved"], plain["solved"])
        self.assertEqual(out["tried"], plain["tried"])
        self.assertFalse(out["solved"])
        self.assertEqual(out["tried"], 0)
        self.assertEqual(out["examined"], 0)
    def test_runs_within_cpu_budget(self) -> None:
        """Whole module stays under 600 wall seconds."""
        print("author wall=%.1fs" % self.wall_s)
        self.assertLess(self.wall_s, 600.0)
if __name__ == "__main__":
    unittest.main()
