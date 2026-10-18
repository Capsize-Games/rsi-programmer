"""Tests for cheap shortlist ranking with complete fallback execution."""
from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from rsi_programmer import pre_eval, shortlist_pre_eval
from rsi_programmer.policy import encode, layer_triples
from rsi_programmer.probe import Expr, OPS, interpret, signature, terminals


class ShortlistPreEvaluationTests(unittest.TestCase):
    def test_policy_features_keep_global_coordinates(self) -> None:
        rows = ((1, 2), (3, 5))
        target = (99, 98)
        observed: list[list[list[float]]] = []

        class Capture:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                observed.append(feats)
                return list(range(len(feats)))

        result = shortlist_pre_eval.shortlist_pre_evaluation_search(
            Capture(), rows, target, 1, 12, window_size=4,
            shortlist_size=2)
        triples = layer_triples({0: list(terminals())}, 1)
        reference = [encode(
            op, signature(left, rows), signature(right, rows), target,
            left.size, right.size, 1, 1, index + 1, len(triples))
            for index, (op, left, right) in enumerate(triples)]
        for call, feats in enumerate(observed):
            block_start = call * 4
            block_end = min(block_start + 4, len(triples))
            selected = sorted(range(block_start, block_end),
                              key=lambda index: abs(OPS[triples[index][0]](
                                  signature(triples[index][1], rows)[0],
                                  signature(triples[index][2], rows)[0])
                                  - target[0]))[:2]
            expected_features = [reference[index] for index in selected]
            self.assertEqual(feats, expected_features)
        self.assertEqual(result["feature_builds"], 6)
        self.assertEqual(result["generated_descriptors"], 12)
        self.assertEqual(result["cheap_scored_descriptors"], 12)
        self.assertEqual(result["cheap_target_value_lookups"], 1)

    def test_shortlist_and_fallback_execute_each_descriptor_once(self) -> None:
        class ChooseLast:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return list(reversed(range(len(feats))))

        result = shortlist_pre_eval.shortlist_pre_evaluation_search(
            ChooseLast(), ((1, 2),), (999,), 1, 12, window_size=5,
            shortlist_size=2)
        self.assertEqual(result["signature_evaluations"], 12)
        self.assertEqual(result["shortlist_executions"], 6)
        self.assertEqual(result["fallback_executions"], 6)
        self.assertEqual(result["scored_descriptors"], 6)
        self.assertEqual(result["scored_unexecuted_descriptors"], 0)
        self.assertEqual(result["cheap_scored_unexecuted_descriptors"], 0)
        self.assertEqual(result["permanently_pruned_descriptors"], 0)
        self.assertEqual(result["skipped_descriptors"], 0)
        self.assertEqual(result["descriptor_replays"], 0)
        self.assertEqual(result["duplicate_signature_evaluations"], 0)

    def test_fallback_candidate_can_solve_after_shortlist_misses(self) -> None:
        class PreferFirst:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return list(range(len(feats)))

        rows = ((-1, 1), (2, 1))
        expected = (-2, 1)
        result = shortlist_pre_eval.shortlist_pre_evaluation_search(
            PreferFirst(), rows, expected, 1, 12, window_size=12,
            shortlist_size=1)
        self.assertTrue(result["solved"])
        self.assertGreater(result["signature_evaluations"], 1)
        self.assertLessEqual(result["signature_evaluations"], 12)
        self.assertGreater(result["fallback_executions"], 0)
        self.assertEqual(interpret(result["graph"], 3, 1), 2)

    def test_oracle_rejection_continues_to_fallback_solution(self) -> None:
        class ChooseFirst:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return list(range(len(feats)))

        calls = 0

        def oracle(graph: dict[str, Any]) -> bool:
            nonlocal calls
            calls += 1
            return calls > 1 and interpret(graph, 3, 4) == 7

        result = shortlist_pre_eval.shortlist_pre_evaluation_search(
            ChooseFirst(), ((1, 1), (2, 2)), (2, 4), 1, 12,
            oracle=oracle, window_size=12, shortlist_size=1)
        self.assertTrue(result["solved"])
        self.assertEqual(calls, 2)
        self.assertEqual(result["rejected"][0]["verdict"],
                         "oracle-rejected")
        self.assertGreater(result["fallback_executions"], 0)

    def test_completed_layer_retention_uses_generation_order(self) -> None:
        class ReverseRank:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return list(reversed(range(len(feats))))

        rows = ((1, 2),)
        target = (999,)
        result = shortlist_pre_eval.shortlist_pre_evaluation_search(
            ReverseRank(), rows, target, 2, 13, window_size=12,
            shortlist_size=12)
        self.assertEqual(result["retained_operand_vectors"], 9)
        self.assertEqual(result["layer_widths"], [12, 108])
        self.assertEqual(result["layer_generated_descriptors"], [12, 12])

    def test_retained_representatives_match_streaming(self) -> None:
        class ReverseRank:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return list(reversed(range(len(feats))))

        rows = ((1, 2), (3, 5))
        target = (999, 998)
        for window, shortlist in ((64, 4), (256, 16)):
            baseline_operands: list[tuple[str, ...]] = []
            shortlist_operands: list[tuple[str, ...]] = []
            original_layer_triples = pre_eval.layer_triples
            original_iter_triples = pre_eval._iter_triples

            def capture_baseline(levels: dict[int, list[Expr]], size: int,
                                 *, limit: int | None = None
                                 ) -> list[tuple[str, Expr, Expr]]:
                if size == 2:
                    baseline_operands.append(tuple(
                        repr(expr) for expr in levels[1]))
                return original_layer_triples(levels, size, limit=limit)

            def capture_shortlist(levels: dict[int, list[Expr]],
                                  size: int
                                  ) -> Any:
                if size == 2:
                    shortlist_operands.append(tuple(
                        repr(expr) for expr in levels[1]))
                yield from original_iter_triples(levels, size)

            with patch.object(pre_eval, "layer_triples",
                              side_effect=capture_baseline):
                pre_eval.pre_evaluation_search(
                    ReverseRank(), rows, target, 2, 1000, consts=(0,),
                    alternatives_per_signature=2)
            with patch.object(pre_eval, "_iter_triples",
                              side_effect=capture_shortlist):
                result = shortlist_pre_eval.shortlist_pre_evaluation_search(
                    ReverseRank(), rows, target, 2, 1000, consts=(0,),
                    alternatives_per_signature=2, window_size=window,
                    shortlist_size=shortlist)
            self.assertEqual(baseline_operands, shortlist_operands)
            self.assertEqual(result["layer_widths"][0], 27)

    def test_execution_ledger_contains_each_descriptor_once(self) -> None:
        for window, shortlist in ((64, 4), (256, 16)):
            generated: list[tuple[str, int, int, int]] = []
            executed: list[tuple[str, int, int, int]] = []
            original_iter = pre_eval._iter_triples
            original_expr = Expr

            def record_triples(levels: dict[int, list[Expr]], size: int
                               ) -> Any:
                for triple in original_iter(levels, size):
                    op, left, right = triple
                    generated.append((op, id(left), id(right), size))
                    yield triple

            def record_expr(op: str, left: Expr | str | int,
                            right: Expr | None = None, size: int = 0,
                            third: Expr | None = None) -> Expr:
                if isinstance(left, Expr) and isinstance(right, Expr):
                    executed.append((op, id(left), id(right), size))
                return original_expr(op, left, right, size, third)

            class RankAll:
                def rank(self, feats: list[list[float]],
                         *args: Any) -> list[int]:
                    return list(range(len(feats)))

            with patch.object(pre_eval, "_iter_triples",
                              side_effect=record_triples), patch.object(
                                  shortlist_pre_eval, "Expr",
                                  side_effect=record_expr):
                result = shortlist_pre_eval.shortlist_pre_evaluation_search(
                    RankAll(), ((1, 2),), (999,), 1, 12,
                    window_size=window, shortlist_size=shortlist)
            self.assertEqual(result["signature_evaluations"], 12)
            self.assertEqual(len(generated), 12)
            self.assertEqual(len(executed), 12)
            self.assertEqual(len(set(generated)), 12)
            self.assertEqual(len(set(executed)), 12)
            self.assertEqual(set(executed), set(generated))

    def test_policy_scored_tail_at_cap_is_counted_as_unexecuted(self) -> None:
        class RankFirst:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return list(range(len(feats)))

        result = shortlist_pre_eval.shortlist_pre_evaluation_search(
            RankFirst(), ((1, 2),), (999,), 1, 2, window_size=12,
            shortlist_size=4)
        self.assertEqual(result["scored_descriptors"], 4)
        self.assertEqual(result["shortlist_executions"], 2)
        self.assertEqual(result["scored_unexecuted_descriptors"], 2)
        self.assertEqual(result["fallback_executions"], 0)

    def test_exact_block_cap_does_not_generate_or_score_next_block(self
                                                                  ) -> None:
        class RankFirst:
            def __init__(self) -> None:
                self.calls: list[int] = []

            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                self.calls.append(len(feats))
                return list(range(len(feats)))

        policy = RankFirst()
        result = shortlist_pre_eval.shortlist_pre_evaluation_search(
            policy, ((1, 2),), (999,), 1, 4, window_size=4,
            shortlist_size=2)
        self.assertEqual(policy.calls, [2])
        self.assertEqual(result["generated_descriptors"], 4)
        self.assertEqual(result["cheap_scored_descriptors"], 4)
        self.assertEqual(result["feature_builds"], 2)
        self.assertEqual(result["signature_evaluations"], 4)
        self.assertEqual(result["retained_operand_vectors"], 0)
        self.assertTrue(result["budget_exhausted"])

    def test_cap_discards_interrupted_layer_and_counts_prefetch(self) -> None:
        class RankFirst:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return list(range(len(feats)))

        root_limited = shortlist_pre_eval.shortlist_pre_evaluation_search(
            RankFirst(), ((1, 2),), (999,), 1, 1, window_size=8,
            shortlist_size=3)
        sig_limited = shortlist_pre_eval.shortlist_pre_evaluation_search(
            RankFirst(), ((1, 2),), (999,), 1, 8, signature_budget=1,
            window_size=8, shortlist_size=3)
        for result in (root_limited, sig_limited):
            self.assertEqual(result["generated_descriptors"], 8)
            self.assertEqual(result["cheap_scored_descriptors"], 8)
            self.assertEqual(result["signature_evaluations"], 1)
            self.assertEqual(result["retained_operand_vectors"], 0)
            self.assertEqual(result["cheap_scored_unexecuted_descriptors"], 7)
            self.assertEqual(result["skipped_descriptors"], 0)
        self.assertTrue(root_limited["budget_exhausted"])
        self.assertTrue(sig_limited["signature_budget_exhausted"])

    def test_cheap_only_avoids_feature_statistics_and_policy(self) -> None:
        with patch.object(shortlist_pre_eval, "FeatureStatistics",
                          side_effect=AssertionError("feature work")):
            result = shortlist_pre_eval.shortlist_pre_evaluation_search(
                None, ((1, 2),), (999,), 1, 12, window_size=12,
                shortlist_size=2, cheap_only=True)
        self.assertEqual(result["feature_builds"], 0)
        self.assertEqual(result["scored_descriptors"], 0)
        self.assertEqual(result["policy_batches"], 0)
        self.assertEqual(result["scored_unexecuted_descriptors"], 0)
        self.assertEqual(result["signature_evaluations"], 12)

    def test_streaming_delegates_to_historical_implementation(self) -> None:
        with patch.object(pre_eval, "pre_evaluation_search",
                          wraps=pre_eval.pre_evaluation_search) as search:
            result = shortlist_pre_eval.shortlist_pre_evaluation_search(
                None, ((1, 2),), (999,), 1, 1)
        search.assert_called_once()
        self.assertEqual(result["generated_descriptors"], 2)

    def test_invalid_options_and_policy_permutation_are_rejected(self) -> None:
        for options in ({"window_size": True}, {"shortlist_size": 0},
                        {"shortlist_size": 5, "window_size": 4},
                        {"cheap_row_index": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                shortlist_pre_eval.shortlist_pre_evaluation_search(
                    object(), ((1, 2),), (999,), 1, 12, **options)

        class BadPolicy:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return [0] * len(feats)

        with self.assertRaisesRegex(ValueError, "permutation"):
            shortlist_pre_eval.shortlist_pre_evaluation_search(
                BadPolicy(), ((1, 2),), (999,), 1, 12,
                window_size=4, shortlist_size=2)


if __name__ == "__main__":
    unittest.main()
