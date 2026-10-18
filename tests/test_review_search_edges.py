"""Small edge regressions for search work accounting contracts."""
from __future__ import annotations

import math
import unittest
from unittest.mock import patch
from typing import Any

from rsi_programmer import traces
from rsi_programmer import author
from rsi_programmer.author import (_author_step, _backtrack, _new_ctx,
                                   _score_order, beam_search)
from rsi_programmer.policy import OP_ORDER, layer_triples
from rsi_programmer.probe import Expr, signature, terminals
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.search import guided_search


def _legacy_backtrack(ctx: dict[str, Any], task_id: str, bound: int,
                      budget: int, stash: dict[int, list[Any]],
                      target: Any, order: str) -> bool:
    """Pre-optimization max/pop reference, kept for differential tests."""
    for size in range(1, bound + 1):
        while stash[size] and ctx["tried"] < budget:
            if order == "score":
                idx = max(range(len(stash[size])),
                          key=lambda i: stash[size][i][0])
                _, triple, sig = stash[size].pop(idx)
            else:
                triple, sig = stash[size].pop(0)[1:]
            ctx["examined"] += 1
            if sig in ctx["sigs"]:
                continue
            ctx["tried"] += 1
            ctx["events"].append(sig)
            ctx["sigs"].append(sig)
            if sig == ctx.get("raise_on"):
                raise RuntimeError("commit test")
            if sig == ctx.get("solve_on"):
                return True
    return False


def _score_text(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return repr(value)


def _stash_state(stash: dict[int, list[Any]]) -> dict[int, list[Any]]:
    return {size: [(_score_text(score), triple, sig)
                   for score, triple, sig in entries]
            for size, entries in stash.items()}


class SearchEdgeTests(unittest.TestCase):
    def test_score_backtrack_matches_legacy_order_and_mutation(self) -> None:
        cases = [
            {"scores": [2.0, 5.0, 5.0, -math.inf,
                        math.inf, 5.0, 2.0]},
            {"scores": [math.inf, -math.inf, math.inf, -math.inf]},
            {"scores": [0.0, -0.0, 0.0]},
            {"scores": [1.0, 4.0, 3.0, 2.0, 5.0],
             "stale": {1, 3}},
            {"scores": [1.0, 4.0, 3.0, 2.0, 5.0],
             "budget": 2},
            {"scores": [1.0, 4.0, 3.0, 2.0, 5.0],
             "stale": {4}, "budget": 2},
            {"scores": [1.0, 4.0, 3.0, 2.0, 5.0],
             "solve": 4},
            {"scores": [1.0, 4.0, 3.0, 2.0, 5.0],
             "raise": 4},
            {"scores": [math.nan, 3.0, 2.0]},
            {"scores": [2.0, math.nan, 3.0]},
            {"scores": [2.0, 3.0, math.nan]},
        ]

        def run(optimized: bool, case: dict[str, Any]
                ) -> tuple[Any, dict[str, Any], dict[int, list[Any]]]:
            bucket = [(score, ("add", i, "x"), (i,))
                      for i, score in enumerate(case["scores"])]
            stash = {1: bucket, 2: []}
            ctx: dict[str, Any] = {
                "tried": 0, "examined": 0,
                "sigs": [(i,) for i in case.get("stale", set())],
                "events": [], "solve_on": (
                    (case["solve"],) if "solve" in case else None),
                "raise_on": (
                    (case["raise"],) if "raise" in case else None)}

            def commit(state: dict[str, Any], triple: Any, sig: Any,
                       task_id: str, size: int, target: Any) -> bool:
                state["events"].append(sig)
                state["sigs"].append(sig)
                if sig == state["raise_on"]:
                    raise RuntimeError("commit test")
                return sig == state["solve_on"]

            try:
                if optimized:
                    with patch.object(author, "_fresh",
                                      side_effect=lambda sig, state:
                                      sig not in state["sigs"]), \
                         patch.object(author, "_commit", side_effect=commit):
                        result = _backtrack(
                            ctx, "unit", 2,
                            case.get("budget", 100), stash, (), "score")
                else:
                    result = _legacy_backtrack(
                        ctx, "unit", 2, case.get("budget", 100),
                        stash, (), "score")
                outcome: Any = ("returned", result)
            except RuntimeError as exc:
                outcome = ("raised", type(exc).__name__, str(exc))
            return outcome, ctx, stash

        for case in cases:
            with self.subTest(scores=case["scores"], case=case):
                actual = run(True, case)
                expected = run(False, case)
                self.assertEqual(actual[0], expected[0])
                self.assertEqual(actual[1], expected[1])
                self.assertEqual(_stash_state(actual[2]),
                                 _stash_state(expected[2]))

    def test_score_backtrack_zero_budget_skips_ordering(self) -> None:
        stash = {1: [(float(i), ("add", i, "x"), (i,))
                     for i in range(3)]}
        before = _stash_state(stash)
        ctx = {"tried": 0, "examined": 0, "sigs": []}
        with patch.object(author.math, "isnan",
                          side_effect=AssertionError("unexpected sort")):
            self.assertFalse(_backtrack(ctx, "unit", 1, 0, stash,
                                        (), "score"))
            self.assertFalse(_backtrack(
                ctx, "unit", 1, 10, {1: []}, (), "score"))
        self.assertEqual(_stash_state(stash), before)
        self.assertEqual(ctx["examined"], 0)

    def test_ranker_restore_has_explicit_donor_optimizer_opt_in(self) -> None:
        donor = SearchPolicy(seed=2, lr=0.07)
        donor._opt.param_groups[0]["weight_decay"] = 0.1
        snapshot = donor.snapshot()

        recipient = SearchPolicy(seed=3, lr=0.03)
        recipient.restore(snapshot)
        self.assertEqual(recipient._opt.param_groups[0]["lr"], 0.03)
        self.assertEqual(recipient._opt.param_groups[0]["weight_decay"],
                         0.0)

        donor_options = SearchPolicy(seed=4)
        donor_options.restore(snapshot, optimizer_from_snapshot=True)
        self.assertEqual(donor_options._opt.param_groups[0]["lr"], 0.07)
        self.assertEqual(donor_options._opt.param_groups[0]["weight_decay"],
                         0.1)

    def test_trace_signature_evaluations_match_candidate_callback(
            self) -> None:
        recorder = traces.TraceRecorder()
        task = recorder.load_family()["tasks"][0]
        original_enumerate = traces.enumerate_programs
        callback_calls = 0
        enumerate_calls = 0

        def enumerated(*args: Any, **kwargs: Any) -> Any:
            nonlocal callback_calls, enumerate_calls
            enumerate_calls += 1
            callback = kwargs["on_candidate"]

            def counted(expr: Any, sig: Any, tried: int) -> None:
                nonlocal callback_calls
                callback_calls += 1
                callback(expr, sig, tried)

            kwargs["on_candidate"] = counted
            return original_enumerate(*args, **kwargs)

        with patch.object(traces, "enumerate_programs", enumerated):
            record = recorder.record_task(task)
        self.assertEqual(enumerate_calls, 1)
        self.assertGreater(callback_calls, 0)
        self.assertEqual(record["signature_evaluations"], callback_calls)
        self.assertEqual(record["tried"], callback_calls)
        self.assertEqual(len(record["choice_pairs"]), callback_calls)
        self.assertEqual(record["wall_scope"],
                         "enumeration_with_trace_capture")

    def test_author_tiny_budget_keeps_full_layer_ordinal(self) -> None:
        class TiedScores:
            def scores(self, feats: list[list[float]]) -> list[float]:
                self.feats = feats
                return [0.0] * len(feats)

        rows = ((1, 2), (2, 5))
        first = layer_triples({0: list(terminals())}, 1)[0]
        target = signature(Expr(first[0], first[1], first[2], 1), rows)
        policy = TiedScores()
        ctx = _new_ctx()
        solved = _author_step(policy, ctx, rows, target, 1, 1,
                              1, "tiny-budget", 1, [0.0] * 4)
        self.assertTrue(solved)
        self.assertEqual(ctx["tried"], 1)
        self.assertEqual(ctx["examined"], 1)
        self.assertEqual(len(policy.feats), 1)
        self.assertEqual(policy.feats[0][-2:], [1 / 12, 11 / 12])

    def test_author_budget_bounds_huge_layer_materialization(self) -> None:
        class HugeLayer:
            def __init__(self, term: Expr) -> None:
                self.term = term
                self.yielded = 0

            def __len__(self) -> int:
                return 10 ** 9

            def __iter__(self) -> Any:
                while True:
                    self.yielded += 1
                    yield self.term

        class CapturingScores:
            def scores(self, feats: list[list[float]]) -> list[float]:
                self.feats = feats
                return [0.0] * len(feats)

        rows, target = ((1, 2), (2, 5)), (3, 7)
        huge = HugeLayer(Expr("input", "x"))
        ctx = _new_ctx()
        ctx["levels"] = {0: huge}
        policy = CapturingScores()
        triples, _sigs, order, _scores = _score_order(
            policy, ctx, rows, target, 1, 1, 1, [0.0] * 4,
            candidate_limit=1)
        full_width = len(OP_ORDER) * (10 ** 9) ** 2
        self.assertEqual(len(triples), 1)
        self.assertEqual(order, [0])
        self.assertEqual(huge.yielded, 2)
        self.assertEqual(policy.feats[0][-2:],
                         [1 / full_width, 1 - 1 / full_width])
        ctx["tried"] = 1
        self.assertEqual(_score_order(
            policy, ctx, rows, target, 1, 1, 1, [0.0] * 4),
            ([], [], [], []))
        self.assertEqual(huge.yielded, 2)

    def test_guided_search_tiny_budget_can_solve_first_candidate(self) -> None:
        class ProbeOrder:
            def rank(self, feats: list[list[float]],
                     *args: Any) -> list[int]:
                return list(range(len(feats)))

        rows = ((1, 2), (2, 5))
        first = layer_triples({0: list(terminals())}, 1)[0]
        target = signature(Expr(first[0], first[1], first[2], 1), rows)
        result = guided_search(ProbeOrder(), rows, target, 1, 1)
        self.assertTrue(result["solved"])
        self.assertEqual(result["tried"], 1)
        self.assertEqual(result["output_nodes"], 1)

    def test_widen_fallback_reports_discarded_author_work(self) -> None:
        class TiedScores:
            def scores(self, feats: list[list[float]]) -> list[float]:
                return [0.0] * len(feats)
            def rank(self, feats: list[list[float]],
                     *args: Any) -> list[int]:
                return list(range(len(feats)))

        result = beam_search(TiedScores(), ((1, 2), (2, 5)),
                             (999, 999), 1, 1000, width=1,
                             widen=2, fallback=True)
        self.assertFalse(result["solved"])
        self.assertEqual(result["path"], "fallback")
        self.assertEqual(result["author_tried"], 18)
        self.assertEqual(len(result["author_log"]), 18)
        self.assertEqual(result["signature_evaluations"], 60)
        self.assertEqual(result["operand_signature_evaluations"], 122)
        self.assertEqual(result["examined"], 80)


if __name__ == "__main__":
    unittest.main()
