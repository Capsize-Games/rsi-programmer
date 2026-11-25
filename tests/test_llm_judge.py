#!/usr/bin/env python3
"""Tests for the dual-backend gated judge and its batch runner."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from rsi_programmer.llm_bank import LlmBankGenerator
from rsi_programmer.llm_judge import JUDGE_PHASES, SealedHeldError
from rsi_programmer.llm_judge import LlmJudge
from rsi_programmer.llm_judge_batch import JudgeBatchRunner

GRAPH_TASK = "llm_dev_000"
STMT_TASK = "llm_dev_045"

GRAPH_OK = {
    "input_types": {"x": "Int", "y": "Int"},
    "nodes": [{"op": "mul", "type": "Int", "args": ["x", "x"]},
              {"op": "mul", "type": "Int", "args": ["y", "y"]},
              {"op": "add", "type": "Int", "args": [1, 1]},
              {"op": "sub", "type": "Int", "args": [2, 0]}],
    "output": 3}
GRAPH_WRONG = {
    "input_types": {"x": "Int", "y": "Int"},
    "nodes": [{"op": "add", "type": "Int", "args": ["x", "y"]}],
    "output": 0}

STMT_OK = "return acc stop"
# Empty-list trigger: correct on every visible/query row of the
# always-zero task, wrong (1) on the held [] boundary row.
STMT_SAD = ("for set sub acc acc end if eq length zero "
            "set add acc one end return acc stop")
STMT_WRONG = "return one stop"


class JudgeFixtures(unittest.TestCase):
    """Load the frozen bank once without opening sealed choices."""

    @classmethod
    def setUpClass(cls) -> None:
        gen = LlmBankGenerator()
        cls.protocol, tasks, cls.visible, cls.query = \
            gen.load_public()
        cls.tasks = {task["id"]: task for task in tasks}
        cls.sha = cls.protocol["curlee_binary"]["sha256"]

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="rsi-judge-"))
        self.judge = LlmJudge(
            self.tmp / "work", tasks=self.tasks,
            visible=self.visible, query=self.query, held=None,
            expected_compiler_sha=self.sha)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def final_judge(self) -> LlmJudge:
        """Judge bound to held rows for explicit final checks."""
        held = LlmBankGenerator().load_held(unseal=True)
        tmp = Path(tempfile.mkdtemp(prefix="rsi-judge-held-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        return LlmJudge(tmp / "work", tasks=self.tasks,
                        visible=self.visible, query=self.query,
                        held=held,
                        expected_compiler_sha=self.sha)


class HappyPathTests(JudgeFixtures):
    def test_graph_accepted_visible(self) -> None:
        """Hand-built graph passes gate and both backends."""
        verdict, _ = self.judge.judge_visible(
            GRAPH_TASK, "graph", json.dumps(GRAPH_OK))
        self.assertTrue(verdict["accepted"])
        self.assertEqual(verdict["taxonomy"], [])
        self.assertTrue(verdict["valid"])
        self.assertTrue(verdict["python_correct"])
        self.assertTrue(verdict["curlee_correct"])
        self.assertTrue(verdict["identity_ok"])
        self.assertEqual(verdict["rows_checked"], 27)
        self.assertEqual(verdict["stage"], "visible")
        self.assertEqual(verdict["counterexamples"], [])
        self.assertEqual(verdict["signature"]["op"], "sub")
        self.assertEqual(len(verdict["signature"]["sig"]), 3)

    def test_graph_accepted_final(self) -> None:
        """Hand-built graph passes 40 sealed held rows in both."""
        verdict, _ = self.final_judge().judge_final(
            GRAPH_TASK, "graph", json.dumps(GRAPH_OK))
        self.assertTrue(verdict["accepted"])
        self.assertEqual(verdict["taxonomy"], [])
        self.assertTrue(verdict["identity_ok"])
        self.assertEqual(verdict["rows_checked"], 40)
        self.assertEqual(verdict["stage"], "final")
        self.assertEqual(verdict["counterexamples"], [])
        self.assertIsNone(verdict["first_mismatch"])

    def test_statement_accepted_visible_and_final(self) -> None:
        """Hand-built tokens pass visible/query and held stages."""
        verdict, _ = self.judge.judge_visible(
            STMT_TASK, "statement", STMT_OK)
        self.assertTrue(verdict["accepted"])
        self.assertEqual(verdict["taxonomy"], [])
        self.assertTrue(verdict["python_correct"])
        self.assertTrue(verdict["curlee_correct"])
        self.assertIsNone(verdict["identity_ok"])
        self.assertEqual(verdict["rows_checked"], 27)
        final, _ = self.final_judge().judge_final(
            STMT_TASK, "statement", STMT_OK)
        self.assertTrue(final["accepted"])
        self.assertEqual(final["rows_checked"], 40)
        self.assertEqual(final["counterexamples"], [])

    def test_proposal_costs_recorded(self) -> None:
        """Proposer CPU/tokens land in the verdict cost record."""
        proposal = {"candidate_text": json.dumps(GRAPH_OK),
                    "kind": "graph",
                    "token_usage": {"prompt": 120,
                                    "completion": 40},
                    "cpu": 1.5}
        verdict, _ = self.judge.judge_visible(
            GRAPH_TASK, "graph", json.dumps(GRAPH_OK),
            proposal=proposal)
        self.assertTrue(verdict["accepted"])
        self.assertEqual(verdict["tokens"]["prompt"], 120)
        self.assertEqual(verdict["tokens"]["completion"], 40)
        self.assertAlmostEqual(verdict["proposal_cpu"], 1.5)
        self.assertAlmostEqual(
            verdict["costs"]["proposer"]["cpu_seconds"], 1.5)


class SadPathTests(JudgeFixtures):
    def test_overfit_passes_visible_then_fails_final(self) -> None:
        """Empty-list trigger survives repair rows, not held."""
        visible, _ = self.judge.judge_visible(
            STMT_TASK, "statement", STMT_SAD)
        self.assertTrue(visible["accepted"])
        self.assertEqual(visible["taxonomy"], [])
        self.assertEqual(visible["rows_checked"], 27)
        final, _ = self.final_judge().judge_final(
            STMT_TASK, "statement", STMT_SAD)
        self.assertFalse(final["accepted"])
        self.assertEqual(final["taxonomy"], ["python_mismatch"])
        self.assertEqual(final["rows_checked"], 40)
        self.assertEqual(final["mismatches"], 1)
        self.assertEqual(final["first_mismatch"], 0)
        self.assertEqual(final["counterexamples"], [])

    def test_wrong_graph_reports_counterexamples(self) -> None:
        """Repair-visible failures return the failing rows."""
        visible_rows = self.judge.counterexamples(
            GRAPH_TASK, "graph", json.dumps(GRAPH_WRONG),
            scope="visible")
        self.assertEqual(len(visible_rows), 3)
        self.assertEqual(visible_rows[0], [0, 1, 2])
        query_rows = self.judge.counterexamples(
            GRAPH_TASK, "graph", json.dumps(GRAPH_WRONG),
            scope="query")
        self.assertEqual(len(query_rows), 26)
        self.assertTrue(all(len(row) == 3 for row in query_rows))

    def test_correct_graph_has_no_counterexamples(self) -> None:
        """An accepted candidate yields no repair rows."""
        rows = self.judge.counterexamples(
            GRAPH_TASK, "graph", json.dumps(GRAPH_OK),
            scope="query")
        self.assertEqual(rows, [])


class BadPathTests(JudgeFixtures):
    def test_malformed_graph_rejected(self) -> None:
        """Bad JSON never reaches either backend."""
        verdict, _ = self.judge.judge_visible(
            GRAPH_TASK, "graph", "not json {")
        self.assertFalse(verdict["accepted"])
        self.assertEqual(verdict["taxonomy"], ["structural"])
        self.assertFalse(verdict["valid"])
        self.assertIsNone(verdict["python_correct"])
        self.assertIsNone(verdict["curlee_correct"])

    def test_malformed_statement_rejected(self) -> None:
        """An incomplete token program fails the structural gate."""
        verdict, _ = self.judge.judge_visible(
            STMT_TASK, "statement", "return")
        self.assertFalse(verdict["accepted"])
        self.assertEqual(verdict["taxonomy"], ["structural"])
        self.assertIsNone(verdict["python_correct"])

    def test_fenced_graph_rejected_before_execution(self) -> None:
        """Denied references stop the candidate at the gate."""
        verdict, _ = self.judge.judge_visible(
            GRAPH_TASK, "graph",
            json.dumps(GRAPH_OK) + " oracle_answers")
        self.assertFalse(verdict["accepted"])
        self.assertEqual(verdict["taxonomy"],
                         ["denied_reference"])
        self.assertIsNone(verdict["python_correct"])
        self.assertIsNone(verdict["curlee_correct"])

    def test_fenced_statement_rejected_before_execution(self) -> None:
        """Fenced token programs never reach the sandbox."""
        verdict, _ = self.judge.judge_visible(
            STMT_TASK, "statement", STMT_OK + " oracle")
        self.assertFalse(verdict["accepted"])
        self.assertEqual(verdict["taxonomy"],
                         ["denied_reference"])
        self.assertIsNone(verdict["python_correct"])

    def test_timeout_is_a_verdict_not_an_exception(self) -> None:
        """An exhausted budget maps to a timeout verdict."""
        with mock.patch.object(
                LlmJudge, "_run_curlee_source",
                side_effect=TimeoutError("cell cap")):
            verdict, _ = self.judge.judge_visible(
                GRAPH_TASK, "graph", json.dumps(GRAPH_OK))
        self.assertFalse(verdict["accepted"])
        self.assertEqual(verdict["taxonomy"], ["timeout"])


class HeldIsolationTests(JudgeFixtures):
    def test_final_without_held_raises(self) -> None:
        """Repair-facing judges cannot reach the final stage."""
        with self.assertRaises(SealedHeldError):
            self.judge.judge_final(
                GRAPH_TASK, "graph", json.dumps(GRAPH_OK))
        with self.assertRaises(SealedHeldError):
            self.judge.held_rows(GRAPH_TASK)

    def test_held_counterexample_scope_raises(self) -> None:
        """Repair code cannot extract held rows as feedback."""
        with self.assertRaises(SealedHeldError):
            self.judge.counterexamples(
                GRAPH_TASK, "graph", json.dumps(GRAPH_WRONG),
                scope="held")
        with self.assertRaises(ValueError):
            self.judge.counterexamples(
                GRAPH_TASK, "graph", json.dumps(GRAPH_WRONG),
                scope="sealed")

    def test_query_scope_stays_repair_visible(self) -> None:
        """Visible and query scopes work without held data."""
        rows = self.judge.counterexamples(
            GRAPH_TASK, "graph", json.dumps(GRAPH_OK),
            scope="visible")
        self.assertEqual(rows, [])


class DedupTests(JudgeFixtures):
    def test_equivalent_candidates_cluster(self) -> None:
        """Same (op, sig) keeps cheapest; drops are counted."""
        dup = json.dumps(GRAPH_OK, indent=1)
        proposals = [
            {"kind": "graph", "candidate_text": json.dumps(
                GRAPH_OK), "cpu": 2.0},
            {"kind": "graph", "candidate_text": dup, "cpu": 1.0},
            {"kind": "graph",
             "candidate_text": json.dumps(GRAPH_WRONG),
             "cpu": 0.5}]
        kept, dropped, cost = self.judge.cluster(
            GRAPH_TASK, proposals)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(kept), 2)
        by_text = {entry["candidate_text"]: entry
                   for entry in kept}
        self.assertAlmostEqual(by_text[dup]["cpu"], 1.0)
        self.assertIn("cpu_seconds", cost)
        self.assertIn("wall_seconds", cost)


def _correct_arm(task: dict[str, Any]) -> list[dict[str, Any]]:
    """One canned correct candidate for the task interface."""
    if task.get("interface") == "list":
        return [{"kind": "statement", "candidate_text": STMT_OK,
                 "cpu": 0.5}]
    return [{"kind": "graph",
             "candidate_text": json.dumps(GRAPH_OK), "cpu": 0.5}]


def _dup_arm(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Correct candidates with a dropped equivalent on graphs."""
    if task.get("interface") == "list":
        return [{"kind": "statement", "candidate_text": STMT_OK,
                 "cpu": 0.5}]
    return [{"kind": "graph", "candidate_text": json.dumps(
        GRAPH_OK), "cpu": 2.0},
            {"kind": "graph",
             "candidate_text": json.dumps(GRAPH_OK, indent=1),
             "cpu": 1.0}]


def _wrong_arm(task: dict[str, Any]) -> list[dict[str, Any]]:
    """One canned wrong candidate for the task interface."""
    if task.get("interface") == "list":
        return [{"kind": "statement", "candidate_text": STMT_WRONG,
                 "cpu": 0.5}]
    return [{"kind": "graph",
             "candidate_text": json.dumps(GRAPH_WRONG),
             "cpu": 0.5}]


class BatchTests(JudgeFixtures):
    def run_batch(self, arms: dict[str, Any],
                  tasks: list[str]) -> dict[str, Any]:
        """Run one batch into a scratch output directory."""
        outdir = self.tmp / "batch"
        runner = JudgeBatchRunner(
            outdir, tasks=self.tasks, visible=self.visible,
            query=self.query, arms=arms,
            expected_compiler_sha=self.sha)
        return runner.run(task_ids=tasks)

    def test_batch_counts_and_wrong_control_rejected(self) -> None:
        """Per-task verdicts aggregate with controls rejected."""
        summary = self.run_batch(
            {"single": _correct_arm, "wrong": _wrong_arm},
            [GRAPH_TASK, STMT_TASK])
        counts = summary["counts"]
        self.assertEqual(counts["tasks"], 2)
        self.assertEqual(counts["arms"]["single"], 2)
        self.assertEqual(counts["arms"]["wrong"], 0)
        self.assertEqual(counts["controls_rejected"], 2)
        self.assertEqual(counts["controls_total"], 2)
        self.assertEqual(counts["dropped_equivalents"], 0)
        self.assertIn("python_mismatch", summary["taxonomy"])
        families = summary["breakdown"]["by_family"]
        self.assertEqual(len(families), 2)
        for slot in families.values():
            self.assertEqual(slot["tasks"], 1)
            self.assertEqual(slot["accepted"]["single"], 1)
            self.assertEqual(slot["accepted"]["wrong"], 0)
        interfaces = summary["breakdown"]["by_interface"]
        self.assertEqual(set(interfaces), {"pair", "list"})
        compiler = summary["compiler"]
        self.assertTrue(compiler["stable"])
        self.assertEqual(compiler["sha256_before"], self.sha)
        self.assertTrue(compiler["guard"])
        self.assertTrue((self.tmp / "batch" / "summary.json"
                         ).is_file())

    def test_batch_dedup_counts_dropped_equivalents(self) -> None:
        """Duplicate (op, sig) candidates collapse to cheapest."""
        summary = self.run_batch({"dup": _dup_arm},
                                 [GRAPH_TASK, STMT_TASK])
        self.assertEqual(
            summary["counts"]["dropped_equivalents"], 1)
        self.assertEqual(summary["counts"]["arms"]["dup"], 2)
        lite = summary["per_task"][GRAPH_TASK]["arms"]["dup"]
        self.assertTrue(lite["accepted"])
        self.assertEqual(lite["cluster_size"], 1)

    def test_batch_cost_phases_and_excluded_costs(self) -> None:
        """Six judge phases plus declared exclusions are present."""
        summary = self.run_batch({"single": _correct_arm},
                                 [GRAPH_TASK, STMT_TASK])
        phases = summary["costs"]["judge_phases"]
        self.assertEqual(set(phases), set(JUDGE_PHASES))
        for slot in phases.values():
            self.assertGreaterEqual(slot["cpu_seconds"], 0.0)
            self.assertGreaterEqual(slot["wall_seconds"], 0.0)
        self.assertTrue(summary["costs"]["excluded_costs"])
        self.assertGreater(
            summary["costs"]["measured_total_wall_seconds"], 0.0)
        self.assertGreater(
            summary["costs"]["measured_total_cpu_seconds"], 0.0)
        self.assertTrue(any(
            name.startswith("curlee_check_")
            for name in summary["costs"]["command_costs"]))
        ledger = summary["costs"]["cost_ledger"]
        self.assertIn("measured_total_cpu_seconds", ledger)
        self.assertIn("command_costs", ledger)
        lite = summary["per_task"][GRAPH_TASK]["arms"]["single"]
        self.assertEqual(set(lite["costs"]), set(JUDGE_PHASES))
        self.assertGreater(
            lite["costs"]["sandbox"]["wall_seconds"], 0.0)
        self.assertGreater(
            lite["costs"]["curlee_check"]["wall_seconds"], 0.0)

    def test_no_held_rows_in_batch_artifacts(self) -> None:
        """Summaries carry counts, never held row values."""
        summary = self.run_batch(
            {"single": _correct_arm, "wrong": _wrong_arm},
            [GRAPH_TASK, STMT_TASK])
        dump = json.dumps(summary)
        self.assertNotIn("held_rows", dump)
        for task_id, entry in summary["per_task"].items():
            self.assertNotIn(task_id + "_held", dump)
            for lite in list(entry["arms"].values()) + [
                    entry["control"]]:
                self.assertEqual(lite["stage"], "visible")
                self.assertEqual(lite["counterexamples"], [])
        self.assertIn("counterexample_count", dump)


if __name__ == "__main__":
    unittest.main()
