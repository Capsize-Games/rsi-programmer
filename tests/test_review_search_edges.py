"""Small edge regressions for search work accounting contracts."""
from __future__ import annotations

import unittest
from unittest.mock import patch
from typing import Any

from rsi_programmer import traces
from rsi_programmer.author import (_author_step, _new_ctx, _score_order,
                                   beam_search)
from rsi_programmer.policy import OP_ORDER, layer_triples
from rsi_programmer.probe import Expr, signature, terminals
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.search import guided_search


class SearchEdgeTests(unittest.TestCase):
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
