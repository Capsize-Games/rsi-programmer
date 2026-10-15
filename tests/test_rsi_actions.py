#!/usr/bin/env python3
"""Tests for the RSI W1 action space and shadow traces (#30)."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.actions import ActionSpace
from rsi_programmer.traces import TraceRecorder
from rsi_programmer.probe import interpret, validate_graph

FIXTURE = (Path(__file__).resolve().parent.parent
           / "fixtures" / "rsi_shadow_tasks.json")

EMPTY = {"input_types": {"x": "Int", "y": "Int"},
         "nodes": [], "output": "x"}


class ActionSpaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = ActionSpace()

    def test_grammar_matches_probe(self) -> None:
        self.assertEqual(self.space.ops(), ("add", "sub", "mul"))
        self.assertEqual(self.space.inputs(), ("x", "y"))
        self.assertEqual(len(self.space.input_actions()), 2)

    def test_apply_covers_full_cross_product(self) -> None:
        choices = self.space.apply_actions(["x", "y", 0])
        self.assertEqual(len(choices), 27)
        ops = {c["op"] for c in choices}
        self.assertEqual(ops, {"add", "sub", "mul"})

    def test_typecheck_rejects_bad_edits(self) -> None:
        bad = [{"kind": "apply", "op": "div",
                "left": "x", "right": "y"},
               {"kind": "apply", "op": "add",
                "left": 9, "right": "y"},
               {"kind": "input", "ref": "z"},
               {"kind": "apply", "op": "add", "left": "x"},
               {"kind": "jump", "ref": "x"}]
        for action in bad:
            with self.assertRaises(ValueError):
                self.space.typecheck(dict(EMPTY), dict(action))

    def test_apply_builds_valid_graph(self) -> None:
        first = self.space.apply(dict(EMPTY),
                                 {"kind": "apply", "op": "add",
                                  "left": "x", "right": "y"})
        validate_graph(first)
        second = self.space.apply(first,
                                  {"kind": "apply", "op": "mul",
                                   "left": 0, "right": "y"})
        validate_graph(second)
        self.assertEqual(interpret(second, 2, 3), 15)

    def test_forward_reference_rejected(self) -> None:
        graph = {"input_types": {"x": "Int", "y": "Int"},
                 "nodes": [], "output": "x"}
        with self.assertRaises(ValueError):
            self.space.typecheck(graph, {"kind": "apply",
                                         "op": "add", "left": 0,
                                         "right": "y"})

    def test_const_actions_cover_ir_range(self) -> None:
        choices = self.space.const_actions()
        self.assertEqual([c["value"] for c in choices],
                         [-2, -1, 0, 1, 2, 3, 4, 5])
        graph = self.space.apply(dict(EMPTY),
                                 {"kind": "const", "value": 3})
        validate_graph(graph)
        full = self.space.apply(graph,
                                {"kind": "apply", "op": "add",
                                 "left": "x", "right": 0})
        validate_graph(full)
        self.assertEqual(interpret(full, 4, 9), 7)

    def test_const_out_of_range_rejected(self) -> None:
        for action in ({"kind": "const", "value": 99},
                       {"kind": "const", "value": "x"},
                       {"kind": "const"}):
            with self.assertRaises(ValueError):
                self.space.typecheck(dict(EMPTY), dict(action))


class TraceRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recorder = TraceRecorder(str(FIXTURE))

    def test_family_loads_with_ids(self) -> None:
        family = self.recorder.load_family()
        tasks = family["tasks"]
        self.assertGreater(len(tasks), 0)
        for task in tasks:
            self.assertIn("id", task)
            self.assertIn("train", task)
            self.assertIn("held_out", task)

    def test_split_hash_stable(self) -> None:
        family = self.recorder.load_family()
        first = family["tasks"][0]
        self.assertEqual(self.recorder.split_hash(first),
                         self.recorder.split_hash(first))
        if len(family["tasks"]) > 1:
            self.assertNotEqual(
                self.recorder.split_hash(first),
                self.recorder.split_hash(family["tasks"][1]))

    def test_oracle_uses_frozen_expectations(self) -> None:
        graph = {"input_types": {"x": "Int", "y": "Int"},
                 "nodes": [{"op": "add", "type": "Int",
                            "args": ["x", "y"]}],
                 "output": 0}
        good = {"train": [[1, 2, 3]], "held_out": [[0, 5, 5]]}
        bad = {"train": [[1, 2, 4]], "held_out": [[0, 5, 5]]}
        self.assertTrue(self.recorder.oracle_holds(good, graph))
        self.assertFalse(self.recorder.oracle_holds(bad, graph))

    def test_record_task_reports_baseline(self) -> None:
        family = self.recorder.load_family()
        record = self.recorder.record_task(family["tasks"][0])
        for key in ("choice_pairs", "solved", "split_sha256",
                    "task", "tried", "unique", "wall_s"):
            self.assertIn(key, record)
        self.assertGreater(int(record["tried"]), 0)
        self.assertGreater(len(record["choice_pairs"]), 0)
        print("task=%s solved=%s tried=%s wall=%.3fs"
              % (record["task"], record["solved"],
                 record["tried"], record["wall_s"]))

    def test_record_all_summarizes_family(self) -> None:
        report = self.recorder.record_all()
        baseline = report["baseline"]
        self.assertGreater(baseline["tasks"], 0)
        self.assertGreaterEqual(baseline["solve_rate"], 0.0)
        self.assertLessEqual(baseline["solve_rate"], 1.0)
        print("solve=%d/%d wall=%.2fs"
              % (baseline["solved"], baseline["tasks"],
                 baseline["total_wall_s"]))

    def test_const_fixture_splits_match(self) -> None:
        path = (Path(__file__).resolve().parent.parent
                / "fixtures" / "rsi_const_tasks.json")
        recorder = TraceRecorder(str(path))
        tasks = recorder.load_family()["tasks"]
        self.assertEqual([t["id"] for t in tasks],
                         ["plus_three", "shift_mul"])
        for task in tasks:
            self.assertEqual(recorder.split_hash(task),
                             task["split_sha256"])
        self.assertTrue(recorder.oracle_holds(
            tasks[0], {"input_types": {"x": "Int", "y": "Int"},
                       "nodes": [{"op": "const", "type": "Int",
                                  "args": [], "value": 3},
                                 {"op": "add", "type": "Int",
                                  "args": ["x", 0]}],
                       "output": 1}))


if __name__ == "__main__":
    unittest.main()
