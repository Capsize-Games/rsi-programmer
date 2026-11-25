#!/usr/bin/env python3
"""Tests for the bounded counterexample-guided repair loop."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.llm_bank import LlmBankGenerator
from rsi_programmer.llm_judge import LlmJudge, SealedHeldError
from rsi_programmer.llm_proposer import MODEL_ID, ProposerAdapter
from rsi_programmer.llm_repair import QUERY_REVEAL_CAP, R_MAX
from rsi_programmer.llm_repair import RepairLoop
from rsi_programmer.llm_repair import assert_disjoint_from_held
from rsi_programmer.llm_repair import template_sha256

TASK = "t1"
KIND = "statement"
VISIBLE = [[[1], 1], [[2], 2], [[0], 0]]
QUERY = [[[3], 3], [[4], 4], [[5], 5], [[6], 6]]
WRONG = "return one stop"
FIXED = "return acc stop"
FRAG = {"provenance": {"source": "test"}, "tokens": ["return"]}


def _verdict(task_id: str, kind: str, accepted: bool,
             fails: list[list[Any]]) -> dict[str, Any]:
    """Minimal judge-shaped verdict over canned failures."""
    return {"task_id": task_id, "kind": kind, "stage": "visible",
            "accepted": accepted, "valid": True,
            "python_correct": accepted,
            "curlee_correct": accepted if accepted else None,
            "identity_ok": None, "rows_checked": 7,
            "mismatches": 0 if accepted else len(fails),
            "first_mismatch": None if accepted else 0,
            "taxonomy": [] if accepted else ["python_mismatch"],
            "detail": "ok" if accepted else "rows differ",
            "signature": None, "counterexamples": fails,
            "compiler": {"path": "canned", "sha256": "canned",
                         "guard": False},
            "range_note": "canned",
            "costs": {}, "tokens": {"prompt": 0,
                                    "completion": 0},
            "proposal_cpu": 0.0}


class FakeJudge:
    """Canned repair-facing judge; no backends, no held rows."""

    def __init__(self, accept: set[str],
                 fails: dict[str, list[list[Any]]]
                 ) -> None:
        self._accept = set(accept)
        self._fails = {key: [list(row) for row in rows]
                       for key, rows in fails.items()}
        self.calls: list[tuple[str, ...]] = []

    def _fails_for(self, text: str,
                   rows: list[list[Any]]) -> list[list[Any]]:
        if text in self._accept:
            return []
        if text in self._fails:
            return [list(row) for row in self._fails[text]]
        return [list(row) for row in rows]

    def judge_visible(
            self, task_id: str, kind: str, text: str,
            proposal: Any = None) -> tuple[dict[str, Any],
                                           dict[str, Any]]:
        """Canned verdict; accepted only for listed texts."""
        self.calls.append(("judge_visible", text))
        fails = self._fails_for(
            text, VISIBLE + QUERY)
        return _verdict(task_id, kind, text in self._accept,
                        fails), {}

    def counterexamples(
            self, task_id: str, kind: str, text: str,
            scope: str = "visible") -> list[list[Any]]:
        """Canned failures; held scope is always refused."""
        self.calls.append(("counterexamples", scope, text))
        if scope == "held":
            raise SealedHeldError(
                "held rows are unavailable to repair")
        if scope == "visible":
            return self._fails_for(text, VISIBLE)
        if scope == "query":
            return self._fails_for(text, VISIBLE + QUERY)
        raise ValueError(f"unknown scope: {scope!r}")


def _loop(judge: FakeJudge, **extra: Any) -> RepairLoop:
    """One repair loop over the canned pools and fragments."""
    return RepairLoop(
        ProposerAdapter(), judge,  # type: ignore[arg-type]
        visible={TASK: VISIBLE}, query={TASK: QUERY},
        fragments={TASK: [FRAG]}, **extra)


def _script(texts: list[str]) -> Any:
    """Canned candidate source replaying one text per round."""
    def run(number: int, prompt: str | None,
            history: list[dict[str, Any]]
            ) -> tuple[str, dict[str, int], float]:
        index = min(number, len(texts) - 1)
        return texts[index], {"prompt": 10,
                              "completion": 5}, 0.1
    return run


class FailThenFixTests(unittest.TestCase):
    def test_accepted_round_one_with_provenance(self) -> None:
        """A wrong single shot is repaired and accepted."""
        judge = FakeJudge({FIXED}, {WRONG: [VISIBLE[0]]})
        loop = _loop(judge)
        got = loop.repair(TASK, KIND, _script([WRONG, FIXED]))
        self.assertTrue(got["accepted"])
        self.assertEqual(got["status"], "accepted")
        self.assertEqual(got["accepted_round"], 1)
        self.assertFalse(got["single_shot_accepted"])
        self.assertEqual(len(got["rounds"]), 2)
        first, second = got["rounds"]
        self.assertIsNone(first["prompt"])
        self.assertEqual(first["proposal"]["model"], MODEL_ID)
        self.assertEqual(
            first["proposal"]["candidate_text"], WRONG)
        sent = json.loads(second["prompt"])
        self.assertEqual(sent["template_version"],
                         "llm_repair_prompt_v1")
        self.assertEqual(sent["template_sha256"],
                         template_sha256())
        self.assertEqual(sent["failing_batch"][0], VISIBLE[0])
        self.assertIn("2**63", sent["instruction"])
        self.assertEqual(sent["fragments"], [FRAG])
        self.assertEqual(sent["diagnostics"]["taxonomy"],
                         ["python_mismatch"])
        self.assertEqual(
            sent["diagnostics"]["counterexamples"], [])
        self.assertEqual(got["revealed_count"], 0)
        self.assertEqual(got["single_shot"]["accepted"], False)

    def test_repair_prompt_is_deterministic(self) -> None:
        """Same history rebuilds the identical prompt bytes."""
        judge = FakeJudge({FIXED}, {WRONG: [VISIBLE[0]]})
        first = _loop(judge).repair(
            TASK, KIND, _script([WRONG, FIXED]))
        judge = FakeJudge({FIXED}, {WRONG: [VISIBLE[0]]})
        second = _loop(judge).repair(
            TASK, KIND, _script([WRONG, FIXED]))
        self.assertEqual(first["rounds"][1]["prompt"],
                         second["rounds"][1]["prompt"])

    def test_single_shot_recoverable(self) -> None:
        """An immediate accept keeps round 0 as the outcome."""
        judge = FakeJudge({FIXED}, {})
        got = _loop(judge).repair(TASK, KIND, _script([FIXED]))
        self.assertEqual(got["status"], "accepted")
        self.assertEqual(got["accepted_round"], 0)
        self.assertTrue(got["single_shot_accepted"])
        self.assertTrue(got["single_shot"]["accepted"])
        self.assertEqual(len(got["rounds"]), 1)
        self.assertEqual(got["revealed_count"], 0)


class CapAndLoopTests(unittest.TestCase):
    def test_never_fix_stops_at_round_cap(self) -> None:
        """Distinct failures run rounds 0..R then stop."""
        texts = [f"wrong {index} stop" for index in range(9)]
        judge = FakeJudge(set(), {})
        got = _loop(judge).repair(TASK, KIND, _script(texts))
        self.assertEqual(got["status"], "round_cap")
        self.assertFalse(got["accepted"])
        self.assertIsNone(got["accepted_round"])
        self.assertEqual(len(got["rounds"]), R_MAX + 1)
        self.assertEqual(
            [entry["round"] for entry in got["rounds"]],
            list(range(R_MAX + 1)))

    def test_repeater_declared_loop(self) -> None:
        """A repeated candidate hash stops without judging."""
        judge = FakeJudge(set(), {})
        adapter = ProposerAdapter()
        loop = RepairLoop(
            adapter, judge,  # type: ignore[arg-type]
            visible={TASK: VISIBLE}, query={TASK: QUERY})
        got = loop.repair(TASK, KIND, _script([WRONG]))
        self.assertEqual(got["status"], "loop")
        self.assertFalse(got["accepted"])
        self.assertEqual(len(got["rounds"]), 2)
        self.assertIsNone(got["rounds"][1]["verdict"])
        self.assertEqual(got["rounds"][1]["note"],
                         "loop: repeated hash")
        self.assertEqual(len(adapter.store), 1)

    def test_cost_cap_stops_repair(self) -> None:
        """Per-task CPU cap charges through the ledger."""
        def run(number: int, prompt: str | None,
                history: list[dict[str, Any]]
                ) -> tuple[str, dict[str, int], float]:
            return f"hot {number} stop", {"prompt": 10,
                                          "completion": 5}, 1.0
        judge = FakeJudge(set(), {})
        loop = _loop(judge, cpu_cap=0.5)
        got = loop.repair(TASK, KIND, run)
        self.assertEqual(got["status"], "cost_cap")
        self.assertEqual(len(got["rounds"]), 1)
        self.assertAlmostEqual(
            got["totals"]["cpu_seconds"], 1.0)
        self.assertAlmostEqual(
            got["cost_ledger"]["measured_total_cpu_seconds"],
            1.0)

    def test_token_cap_stops_repair(self) -> None:
        """Per-task token cap counts prompt plus completion."""
        judge = FakeJudge(set(), {})
        loop = _loop(judge, token_cap=5)
        got = loop.repair(TASK, KIND, _script([WRONG]))
        self.assertEqual(got["status"], "cost_cap")
        self.assertEqual(got["totals"]["prompt_tokens"], 10)
        self.assertEqual(
            got["totals"]["completion_tokens"], 5)


class HeldIsolationTests(unittest.TestCase):
    def test_repair_never_reaches_held(self) -> None:
        """Only visible/query scopes are called during repair."""
        judge = FakeJudge({FIXED}, {WRONG: [VISIBLE[0]]})
        got = _loop(judge).repair(
            TASK, KIND, _script([WRONG, FIXED]))
        self.assertTrue(got["accepted"])
        for call in judge.calls:
            self.assertIn(call[0], ("judge_visible",
                                    "counterexamples"),
                          msg=str(call))
            if call[0] == "counterexamples":
                self.assertIn(call[1], ("visible", "query"))
        scopes = [call[1] for call in judge.calls
                  if call[0] == "counterexamples"]
        self.assertIn("visible", scopes)

    def test_real_judge_without_held_stays_sealed(self) -> None:
        """A repair-facing judge cannot open the final stage."""
        gen = LlmBankGenerator()
        protocol, tasks, visible, query = gen.load_public()
        rows = {task["id"]: task for task in tasks}
        tmp = Path(tempfile.mkdtemp(prefix="rsi-repair-"))
        try:
            judge = LlmJudge(
                tmp / "work", tasks=rows, visible=visible,
                query=query, held=None,
                expected_compiler_sha=protocol[
                    "curlee_binary"]["sha256"])
            with self.assertRaises(SealedHeldError):
                judge.judge_final(
                    "llm_dev_000", "graph", "{}")
            with self.assertRaises(SealedHeldError):
                judge.held_rows("llm_dev_000")
            with self.assertRaises(SealedHeldError):
                judge.counterexamples(
                    "llm_dev_000", "graph", "{}", scope="held")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_propose_spec_refuses_held_fields(self) -> None:
        """Held/query keys cannot enter the recorded prompt."""
        adapter = ProposerAdapter()
        base = {"task_id": TASK, "kind": KIND,
                "visible_rows": VISIBLE}
        for key in ("held", "held_rows", "query", "query_rows",
                    "oracle", "sealed"):
            with self.assertRaises(ValueError, msg=key):
                adapter.propose(dict(base, **{key: []}), WRONG)

    def test_unknown_kind_and_task_rejected(self) -> None:
        """Repair validates kind and task before any proposal."""
        judge = FakeJudge(set(), {})
        loop = _loop(judge)
        with self.assertRaises(ValueError):
            loop.repair(TASK, "nope", _script([WRONG]))
        with self.assertRaises(ValueError):
            loop.repair("missing", KIND, _script([WRONG]))
        self.assertEqual(judge.calls, [])


class QueryCapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        gen = LlmBankGenerator()
        cls.protocol, tasks, cls.visible, cls.query = \
            gen.load_public()
        cls.tasks = {task["id"]: task for task in tasks}
        cls.held = gen.load_held(unseal=True)

    def test_reveal_capped_and_disjoint_from_held(self) -> None:
        """Visible-pass reveals stay capped and held-free."""
        task_id = "llm_dev_045"
        kind = "statement"
        query_rows = self.query[task_id]
        texts = [f"repair candidate {index} stop"
                 for index in range(9)]
        fails = {texts[0]: query_rows[:2],
                 texts[1]: query_rows[2:4],
                 texts[2]: query_rows[4:6],
                 texts[3]: query_rows[6:8]}
        judge = BankFake(self.visible, self.query, set(), {},
                         fails)
        loop = RepairLoop(
            ProposerAdapter(), judge,  # type: ignore[arg-type]
            visible=self.visible, query=self.query)
        got = loop.repair(task_id, kind, _script(texts))
        self.assertEqual(got["status"], "round_cap")
        self.assertLessEqual(got["revealed_count"],
                             QUERY_REVEAL_CAP)
        self.assertEqual(got["revealed_count"],
                         QUERY_REVEAL_CAP)
        query_keys = {json.dumps(row, sort_keys=True)
                      for row in query_rows}
        for row in got["revealed_query_rows"]:
            self.assertIn(json.dumps(row, sort_keys=True),
                          query_keys)
        assert_disjoint_from_held(
            got["revealed_query_rows"], self.held[task_id])

    def test_overlap_with_held_raises(self) -> None:
        """A held row in the revealed set is refused loudly."""
        task_id = "llm_dev_045"
        with self.assertRaises(ValueError):
            assert_disjoint_from_held(
                [list(self.held[task_id][0])],
                self.held[task_id])


class BankFake:
    """Canned judge over real bank pools; no backends run."""

    def __init__(self, visible: dict[str, Any],
                 query: dict[str, Any], accept: set[str],
                 fails_visible: dict[str, list[list[Any]]],
                 fails_query: dict[str, list[list[Any]]]
                 ) -> None:
        self._visible = visible
        self._query = query
        self._accept = set(accept)
        self._fails_visible = fails_visible
        self._fails_query = fails_query

    def _fails_for(self, task_id: str, text: str,
                   scope: str) -> list[list[Any]]:
        if text in self._accept:
            return []
        if scope == "visible":
            table = self._fails_visible
            if text in table:
                return [list(row) for row in table[text]]
            return []
        table = self._fails_query
        if text in table:
            return [list(row) for row in table[text]]
        pool = self._visible[task_id] + self._query[task_id]
        return [list(row) for row in pool]

    def judge_visible(
            self, task_id: str, kind: str, text: str,
            proposal: Any = None) -> tuple[dict[str, Any],
                                           dict[str, Any]]:
        """Canned verdict shaped like the real judge output."""
        fails = self._fails_for(task_id, text, "query")
        return _verdict(task_id, kind, text in self._accept,
                        fails), {}

    def counterexamples(
            self, task_id: str, kind: str, text: str,
            scope: str = "visible") -> list[list[Any]]:
        """Canned failures; held scope is always refused."""
        if scope == "held":
            raise SealedHeldError(
                "held rows are unavailable to repair")
        if scope not in ("visible", "query"):
            raise ValueError(f"unknown scope: {scope!r}")
        return self._fails_for(task_id, text, scope)


if __name__ == "__main__":
    unittest.main()
