#!/usr/bin/env python3
"""Tests for the dev-bank diagnostic runner (Phase 6, no live run)."""
from __future__ import annotations

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from rsi_programmer.cost_ledger import CostLedger
from rsi_programmer.llm_bank import LlmBankGenerator
from rsi_programmer.llm_dev_diagnostic import DevDiagnosticRunner
from rsi_programmer.llm_dev_diagnostic import DevProtocol
from rsi_programmer.llm_dev_diagnostic import FROZEN_TASK_IDS
from rsi_programmer.llm_dev_diagnostic import TaskCandidateFn
from rsi_programmer.llm_dev_diagnostic import evaluate_bar
from rsi_programmer.llm_judge import JUDGE_PHASES, SealedHeldError
from rsi_programmer.llm_judge import LlmJudge

CARD_TASKS = (
    "llm_dev_004 llm_dev_002 llm_dev_011 llm_dev_010 "
    "llm_dev_009 llm_dev_018 llm_dev_022 llm_dev_020 "
    "llm_dev_016 llm_dev_021 llm_dev_039 llm_dev_031 "
    "llm_dev_030 llm_dev_042 llm_dev_043 llm_dev_047 "
    "llm_dev_050 llm_dev_045 llm_dev_049 llm_dev_048 "
    "llm_dev_068 llm_dev_065 llm_dev_062 llm_dev_072 "
    "llm_dev_073 llm_dev_076 llm_dev_088 llm_dev_083 "
    "llm_dev_081 llm_dev_075").split()

PAIR_TASK = "llm_dev_900"
STMT_TASK = "llm_dev_901"

GOOD_PAIR = json.dumps(
    {"input_types": {"x": "Int", "y": "Int"},
     "nodes": [{"op": "add", "type": "Int",
                "args": ["x", "y"]}],
     "output": 0}, sort_keys=True)
WRONG_PAIR = json.dumps(
    {"input_types": {"x": "Int", "y": "Int"},
     "nodes": [{"op": "mul", "type": "Int",
                "args": ["x", "y"]}],
     "output": 0}, sort_keys=True)
GOOD_STMT = "for set add acc value end return acc stop"
WRONG_STMT = "return one stop"

TOY_TASKS = {
    PAIR_TASK: {"id": PAIR_TASK, "pool": "dev", "family": "A6",
                "interface": "pair"},
    STMT_TASK: {"id": STMT_TASK, "pool": "dev", "family": "S1",
                "interface": "list"},
}
TOY_VISIBLE = {
    PAIR_TASK: [[0, 1, 1], [2, -1, 1], [-2, -3, -5]],
    STMT_TASK: [[[1], 1], [[1, 2], 3], [[], 0]],
}
TOY_QUERY = {
    PAIR_TASK: [[1, 1, 2], [3, 4, 7]],
    STMT_TASK: [[[5], 5]],
}
TOY_HELD = {
    PAIR_TASK: [[5, 5, 10], [0, 0, 0]],
    STMT_TASK: [[[9], 9], [[3, 4], 7]],
}
TOY_GOOD = {PAIR_TASK: GOOD_PAIR, STMT_TASK: GOOD_STMT}
TOY_WRONG = {PAIR_TASK: WRONG_PAIR, STMT_TASK: WRONG_STMT}


def stub_candidate_fn(
        task_id: str, round_no: int, prompt: str | None,
        rounds: list[dict[str, Any]]
) -> tuple[str, dict[str, int], float]:
    """Round 0 wrong, later rounds good, fixed usage and CPU."""
    del prompt, rounds
    if round_no == 0:
        return TOY_WRONG[task_id], {"prompt": 10,
                                    "completion": 20}, 0.5
    return TOY_GOOD[task_id], {"prompt": 10,
                                "completion": 20}, 0.5


class FakeJudge:
    """Stub judge with the verdict shapes the runner consumes."""

    good: dict[str, str] = {}

    def __init__(self, workdir: Path, *, tasks: Any,
                 visible: Any, query: Any, held: Any,
                 expected_compiler_sha: str) -> None:
        """Bind row splits; ignore workdir and compiler pin."""
        del workdir, tasks, expected_compiler_sha
        self._visible = {k: [list(r) for r in v]
                         for k, v in visible.items()}
        self._query = {k: [list(r) for r in v]
                       for k, v in query.items()}
        self._held = {k: [list(r) for r in v]
                      for k, v in held.items()}
        self.ledger = CostLedger()
        self._sha = "fake-sha"

    @property
    def compiler_sha(self) -> str:
        """Pinned digest stand-in for evidence records."""
        return self._sha

    def _rows(self, task_id: str, stage: str) -> list[Any]:
        if stage == "final":
            return [list(r) for r in self._held[task_id]]
        return [list(r) for r in self._visible[task_id]] + [
            list(r) for r in self._query[task_id]]

    def _verdict(self, task_id: str, kind: str, text: str,
                 stage: str) -> dict[str, Any]:
        rows = self._rows(task_id, stage)
        accepted = text == self.good.get(task_id)
        fails = [] if accepted else [rows[0]]
        return {
            "task_id": task_id, "kind": kind, "stage": stage,
            "accepted": accepted, "valid": True,
            "python_correct": True if accepted else False,
            "curlee_correct": True if accepted else None,
            "identity_ok": True if (
                accepted and kind == "graph") else None,
            "rows_checked": len(rows),
            "mismatches": 0 if accepted else len(rows),
            "first_mismatch": None if accepted else 0,
            "taxonomy": [] if accepted else [
                "python_mismatch"],
            "detail": "stub ok" if accepted else "stub differs",
            "signature": {"op": kind, "sig": [0]},
            "counterexamples": fails if stage != "final"
            else [],
            "compiler": {"path": "fake", "sha256": self._sha,
                         "guard": False},
            "range_note": "stub",
            "costs": {phase: {"cpu_seconds": 0.0,
                              "wall_seconds": 0.0}
                      for phase in JUDGE_PHASES},
            "tokens": {"prompt": 1, "completion": 2},
            "proposal_cpu": 0.1}

    def judge_visible(
            self, task_id: str, kind: str, text: str,
            proposal: Any = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Visible verdict over canned stub rows."""
        del proposal
        return self._verdict(task_id, kind, text,
                             "visible"), {}

    def judge_final(
            self, task_id: str, kind: str, text: str,
            proposal: Any = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Final verdict over canned stub held rows."""
        del proposal
        return self._verdict(task_id, kind, text, "final"), {}

    def counterexamples(self, task_id: str, kind: str,
                        text: str, scope: str = "visible"
                        ) -> list[list[Any]]:
        """Failing stub rows; held scope is always refused."""
        del kind
        if scope == "held":
            raise SealedHeldError("held refused")
        if text == self.good.get(task_id):
            return []
        return [list(self._rows(task_id, "visible")[0])]


def make_runner(tmp: Path) -> DevDiagnosticRunner:
    """Runner with the stub judge on the two-task toy bank."""
    protocol = DevProtocol(task_ids=(PAIR_TASK, STMT_TASK))
    with mock.patch(
            "rsi_programmer.llm_dev_diagnostic.LlmJudge",
            FakeJudge):
        FakeJudge.good = dict(TOY_GOOD)
        return DevDiagnosticRunner(
            tmp / "out", tasks=dict(TOY_TASKS),
            visible={k: [list(r) for r in v]
                     for k, v in TOY_VISIBLE.items()},
            query={k: [list(r) for r in v]
                   for k, v in TOY_QUERY.items()},
            held={k: [list(r) for r in v]
                  for k, v in TOY_HELD.items()},
            expected_compiler_sha="fake-sha",
            protocol=protocol)


def drive(runner: DevDiagnosticRunner,
          candidate_fn: TaskCandidateFn = stub_candidate_fn
          ) -> dict[str, Any]:
    """Run every arm plus library and final on the toy bank."""
    runner.run_arm_a(candidate_fn)
    runner.run_arm_b(candidate_fn)
    runner.build_library()
    runner.run_arm_c(candidate_fn)
    runner.run_arm_d()
    runner.run_arm_e()
    runner.finalize()
    return runner.summary()


class ProtocolFrozenTests(unittest.TestCase):
    """Hash determinism, frozen list, and dev-only guards."""

    def test_hash_deterministic(self) -> None:
        """Two protocol instances hash identically."""
        self.assertEqual(DevProtocol().sha256(),
                         DevProtocol().sha256())
        self.assertRegex(DevProtocol().sha256(), r"^[0-9a-f]{64}$")

    def test_frozen_task_list_matches_card(self) -> None:
        """Runner ids equal the 30 frozen card ids in order."""
        self.assertEqual(list(FROZEN_TASK_IDS), CARD_TASKS)
        self.assertEqual(len(FROZEN_TASK_IDS), 30)
        self.assertEqual(len(set(FROZEN_TASK_IDS)), 30)

    def test_runner_records_frozen_sha(self) -> None:
        """Protocol sha is stable before any execution."""
        tmp = Path(tempfile.mkdtemp(prefix="rsi-devdiag-"))
        try:
            runner = make_runner(tmp)
            self.assertEqual(runner.protocol_sha256(),
                             DevProtocol(
                                 task_ids=(PAIR_TASK,
                                           STMT_TASK)).sha256())
            self.assertEqual(runner.task_ids,
                             (PAIR_TASK, STMT_TASK))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rejects_non_dev_task_ids(self) -> None:
        """Sealed-style ids never enter the protocol."""
        with self.assertRaises(ValueError):
            DevProtocol(task_ids=("llm_sealed_000",))
        tmp = Path(tempfile.mkdtemp(prefix="rsi-devdiag-"))
        try:
            with mock.patch(
                    "rsi_programmer.llm_dev_diagnostic."
                    "LlmJudge", FakeJudge):
                with self.assertRaises(ValueError):
                    DevDiagnosticRunner(
                        tmp / "out", tasks=dict(TOY_TASKS),
                        visible=dict(TOY_VISIBLE),
                        query=dict(TOY_QUERY),
                        held={**TOY_HELD,
                              "llm_sealed_000": [[0, 0, 0]]},
                        expected_compiler_sha="x",
                        protocol=DevProtocol(
                            task_ids=(PAIR_TASK, STMT_TASK)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class BarLogicTests(unittest.TestCase):
    """PROCEED/KILL on canned verdict sets, no backends."""

    def _base(self, **extra: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "llm_accepted": {
                "a": {"t1"},
                "b": {"t1", "t2"},
                "c": {"t1", "t2"}},
            "streaming_accepted": {"t1"},
            "family_of": {"t1": "A6", "t2": "A8",
                          "t3": "S2"},
            "arm_cpu": {
                "a": {"t1": 1.0, "t2": 1.0},
                "b": {"t1": 1.0, "t2": 1.0},
                "c": {"t1": 1.0, "t2": 1.0},
                "d": {"t1": 1.0, "t2": 1.0}},
        }
        body.update(extra)
        return body

    def test_proceed_when_all_prongs_hold(self) -> None:
        """Deep reach plus cheap shared cost plus delta."""
        decision = evaluate_bar(**self._base())
        self.assertEqual(decision["verdict"], "PROCEED")
        self.assertEqual(decision["best_arm"], "b")
        self.assertTrue(
            decision["prongs"]["deep_reach"])
        self.assertEqual(
            decision["prongs"]["deep_tasks"], ["t2"])

    def test_kill_without_deep_reach(self) -> None:
        """Shallow-only gains never proceed, however cheap."""
        body = self._base(
            llm_accepted={"a": {"t1"}, "b": {"t1"},
                          "c": {"t1"}})
        decision = evaluate_bar(**body)
        self.assertEqual(decision["verdict"], "KILL")
        self.assertFalse(
            decision["prongs"]["deep_reach"])

    def test_kill_when_cost_ratio_exceeds_m(self) -> None:
        """A 47x-style cost gap kills despite deep reach."""
        body = self._base(arm_cpu={
            "a": {"t1": 47.0, "t2": 1.0},
            "b": {"t1": 47.0, "t2": 1.0},
            "c": {"t1": 47.0, "t2": 1.0},
            "d": {"t1": 1.0, "t2": 1.0}})
        decision = evaluate_bar(**body)
        self.assertEqual(decision["verdict"], "KILL")
        self.assertGreater(
            decision["prongs"]["cost_ratio"], 3)

    def test_kill_without_repair_or_retrieval_delta(self) -> None:
        """Flat a/b/c counts kill even with deep reach."""
        body = self._base(
            llm_accepted={"a": {"t1", "t2"},
                          "b": {"t1", "t2"},
                          "c": {"t1", "t2"}})
        decision = evaluate_bar(**body)
        self.assertEqual(decision["verdict"], "KILL")
        self.assertFalse(decision["prongs"]["delta"])
        self.assertEqual(
            decision["prongs"]["counts"],
            {"a": 2, "b": 2, "c": 2})

    def test_kill_when_streaming_solves_nothing(self) -> None:
        """Undefined cost ratio kills when nothing is shared."""
        body = self._base(streaming_accepted=set())
        decision = evaluate_bar(**body)
        self.assertEqual(decision["verdict"], "KILL")
        self.assertIsNone(decision["prongs"]["cost_ratio"])
        self.assertEqual(
            decision["prongs"]["shared_tasks"], [])

    def test_retrieval_delta_alone_proceeds(self) -> None:
        """Flat repair but strict c-vs-b delta still proceeds."""
        body = self._base(
            llm_accepted={"a": {"t1", "t2"},
                          "b": {"t1", "t2"},
                          "c": {"t1", "t2", "t3"}})
        decision = evaluate_bar(**body)
        self.assertEqual(decision["verdict"], "PROCEED")
        self.assertEqual(decision["best_arm"], "c")


class RunnerSmokeTests(unittest.TestCase):
    """Toy-bank run with stub proposer and stub judge."""

    def setUp(self) -> None:
        """Fresh temp dir per smoke run."""
        self.tmp = Path(tempfile.mkdtemp(prefix="rsi-devdiag-"))

    def tearDown(self) -> None:
        """Remove temp output."""
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_drive_writes_summary_and_report(self) -> None:
        """All arms, library freeze, finals, verdict file."""
        runner = make_runner(self.tmp)
        sha_before = runner.protocol_sha256()
        summary = drive(runner)
        self.assertEqual(summary["protocol_sha256"],
                         sha_before)
        self.assertEqual(summary["tasks"],
                         [PAIR_TASK, STMT_TASK])
        self.assertEqual(set(summary["counts"][
            "final_accepted"]), set("abcde"))
        self.assertEqual(summary["counts"][
            "final_accepted"]["b"], 2)
        self.assertEqual(summary["counts"][
            "final_accepted"]["a"], 0)
        self.assertEqual(summary["library"]["entry_count"], 2)
        self.assertTrue(
            summary["library"]["version_hash"])
        self.assertEqual(
            summary["repair_conversion_b"]["converted"], 2)
        self.assertEqual(
            summary["retrieval_delta_c_vs_b"]["delta"], 0)
        self.assertIn(summary["decision"]["verdict"],
                      ("PROCEED", "KILL"))
        report = runner.write_report(
            self.tmp / "inversion-dev-results.md")
        text = report.read_text()
        self.assertIn("## Verdict: ", text)
        self.assertIn("per arm/family", text)
        self.assertTrue((self.tmp / "out" / "summary.json"
                         ).is_file())

    def test_arm_d_matches_pair_task(self) -> None:
        """Streaming control finds x+y on the toy pair task."""
        runner = make_runner(self.tmp)
        runner.run_arm_d()
        record = runner._records["d"][PAIR_TASK]
        self.assertTrue(record["accepted_visible"])
        self.assertGreaterEqual(record["tried"], 1)
        stmt = runner._records["d"][STMT_TASK]
        self.assertEqual(stmt["status"], "not_applicable")


class RealBackendSmokeTests(unittest.TestCase):
    """One real single-task judge call, bounded under 60 s."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the frozen bank once without opening sealed."""
        gen = LlmBankGenerator()
        cls.protocol, tasks, cls.visible, cls.query = \
            gen.load_public()
        cls.tasks = {task["id"]: task for task in tasks}
        cls.sha = cls.protocol["curlee_binary"]["sha256"]

    def test_single_visible_judge_under_budget(self) -> None:
        """Real judge_visible returns a verdict quickly."""
        from pathlib import Path as _Path
        binary = _Path(
            self.protocol["curlee_binary"]["path"])
        if not binary.is_file():
            self.skipTest("CURLEE binary unavailable")
        tmp = _Path(tempfile.mkdtemp(prefix="rsi-devdiag-"))
        try:
            judge = LlmJudge(
                tmp / "work", tasks=self.tasks,
                visible=self.visible, query=self.query,
                held=None,
                expected_compiler_sha=self.sha)
            started = time.monotonic()
            verdict, _ = judge.judge_visible(
                "llm_dev_045", "statement", "return acc stop")
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 60)
            self.assertIn("taxonomy", verdict)
            self.assertGreater(verdict["rows_checked"], 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
