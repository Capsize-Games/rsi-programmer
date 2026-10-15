#!/usr/bin/env python3
"""Backtracking beam author vs ranked search (#38).

Beam keeps a ranked beam alive across layers: SNN scores
order expansion, the validator gates every commit, and
backtracking revisits deprioritized branches. One scored
triple costs one tried, exactly like ranked search.
"""
from __future__ import annotations
import time
import unittest
from pathlib import Path
from typing import Any
from rsi_programmer.actions import ActionSpace
from rsi_programmer.author import TaskRow, beam_search, emit_edit
from rsi_programmer.author import refine_author, train_author
from rsi_programmer.author import _backtrack, _beam_scan, _new_ctx
from rsi_programmer.evolve import guided_search
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.traces import TraceRecorder
from rsi_programmer.probe import validate_graph
FIXTURE = (Path(__file__).resolve().parent.parent
           / "fixtures" / "rsi_shadow_tasks.json")
SEEDS = (0, 1, 2)
EPOCHS = 100
BUDGET = 10 ** 9
WIDTHS = (3, 6)
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
    """Win when beam solves with fewer tried than rank."""
    if row["bm_solved"] and (not row["rank_solved"]
                             or row["bm_tried"] < row["rank_tried"]):
        return "win"
    if row["rank_solved"] and (not row["bm_solved"]
                               or row["bm_tried"] > row["rank_tried"]):
        return "loss"
    return "tie"
def _row(seed: int, width: int, tid: str, ranked: dict[str, Any],
         beam: dict[str, Any]) -> dict[str, Any]:
    """One beam-vs-rank comparison row for the report."""
    return {"seed": seed, "width": width, "task": tid,
            "rank_tried": ranked["tried"], "rank_solved": ranked["solved"],
            "bm_tried": beam["tried"], "bm_solved": beam["solved"],
            "bm_path": beam["path"], "bm_log": beam["rejected"],
            "bm_graph": beam["graph"]}
def _run_seed(seed: int, tasks: list[TaskRow]) -> list[dict[str, Any]]:
    """Train, refine, and compare beam vs rank one seed."""
    policy = SearchPolicy(seed=seed)
    train_author(policy, str(FIXTURE), epochs=EPOCHS)
    rows: list[dict[str, Any]] = []
    for width in WIDTHS:
        weights, _ = refine_author(policy, tasks, seed=seed, topk=width)
        for tid, trows, target, bound in tasks:
            ranked = guided_search(policy, trows, target, bound,
                                   BUDGET, tid)
            beam = beam_search(policy, trows, target, bound, BUDGET,
                               tid, width, weights, fallback=False)
            rows.append(_row(seed, width, tid, ranked, beam))
    return rows
class BeamTests(unittest.TestCase):
    rows: list[dict[str, Any]] = []
    @classmethod
    def setUpClass(cls) -> None:
        """Train three heads; compare beam vs rank once."""
        cls.start = time.perf_counter()
        cls.tasks = _load_tasks()
        fam = TraceRecorder(str(FIXTURE)).load_family()["tasks"]
        cls.by_id = {t["id"]: t for t in fam}
        cls.rows = []
        for seed in SEEDS:
            cls.rows += _run_seed(seed, cls.tasks)
        cls.wall_s = time.perf_counter() - cls.start
    def test_beam_report_with_win(self) -> None:
        """Every row tagged; beam beats rank at least once."""
        tags = [_tag(r) for r in self.rows]
        for row, tag in zip(self.rows, tags):
            print("s%d w%d %-12s rank=%4d/%s bm=%3d/%s %s"
                  % (row["seed"], row["width"], row["task"],
                     row["rank_tried"], row["rank_solved"],
                     row["bm_tried"], row["bm_solved"], tag))
        print("wins=%d ties=%d losses=%d wall=%.1fs"
              % (tags.count("win"), tags.count("tie"),
                 tags.count("loss"), self.wall_s))
        self.assertEqual(len(tags), 24)
        self.assertEqual({r["bm_path"] for r in self.rows}, {"beam"})
        self.assertGreaterEqual(tags.count("win"), 1)
    def test_beam_accounting_identical(self) -> None:
        """One scored triple is one tried, exactly like rank."""
        for row in self.rows:
            self.assertEqual(row["bm_tried"], len(row["bm_log"]))
    def test_every_beam_edit_verdict_carrying(self) -> None:
        """Each emitted edit names its validator verdict."""
        for row in self.rows:
            log = row["bm_log"]
            self.assertGreater(len(log), 0)
            for entry in log:
                self.assertEqual(set(entry),
                                 {"op", "signature", "task", "verdict"})
                if entry["verdict"] not in ("", "signature-mismatch"):
                    self.assertTrue(entry["verdict"].startswith(
                        "validator:"))
            blanks = [e for e in log if e["verdict"] == ""]
            self.assertEqual(len(blanks), 1 if row["bm_solved"] else 0)
    def test_backtrack_revisits_deprioritized(self) -> None:
        """Width-capped scan stashes; backtrack re-evaluates."""
        task = self.by_id["twice_product"]
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        policy = SearchPolicy(seed=0)
        ctx = _new_ctx()
        stash = {s: [] for s in range(1, bound + 1)}
        _beam_scan(policy, ctx, rows, target, 1, bound, BUDGET,
                   "probe", 1, [0.0] * 4, stash)
        self.assertTrue(any(stash[s] for s in stash))
        scanned = ctx["tried"]
        _backtrack(ctx, "probe", bound, BUDGET, stash, target)
        self.assertGreater(ctx["tried"], scanned)
        self.assertEqual(ctx["tried"], len(ctx["log"]))
    def test_solved_graphs_pass_frozen_oracle(self) -> None:
        """Beam solutions validate and meet expectations."""
        rec = TraceRecorder(str(FIXTURE))
        for row in self.rows:
            if not row["bm_solved"]:
                continue
            validate_graph(row["bm_graph"])
            self.assertTrue(rec.oracle_holds(self.by_id[row["task"]],
                                             row["bm_graph"]))
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
        """Zero budget falls back to ranked enumeration."""
        task = self.by_id["twice_product"]
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        policy = SearchPolicy(seed=0)
        out = beam_search(policy, rows, target, bound, 0,
                          "twice_product")
        plain = guided_search(policy, rows, target, bound, 0,
                              "twice_product")
        self.assertEqual(out["path"], "fallback")
        self.assertEqual(out["solved"], plain["solved"])
        self.assertEqual(out["tried"], plain["tried"])
        self.assertTrue(out["solved"])
    def test_runs_within_cpu_budget(self) -> None:
        """Whole module stays under 600 wall seconds."""
        print("beam wall=%.1fs" % self.wall_s)
        self.assertLess(self.wall_s, 600.0)
if __name__ == "__main__":
    unittest.main()
