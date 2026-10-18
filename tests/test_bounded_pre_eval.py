"""Tests for bounded-window candidate ranking."""
from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from rsi_programmer import bounded_pre_eval, pre_eval
from rsi_programmer.policy import encode, layer_triples
from rsi_programmer.probe import signature, terminals
from rsi_programmer.probe import interpret


class BoundedPreEvaluationTests(unittest.TestCase):
    def test_global_ordinals_and_full_layer_denominator(self) -> None:
        calls: list[list[list[float]]] = []

        class First:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                calls.append(feats)
                return list(range(len(feats)))

        out = bounded_pre_eval.bounded_pre_evaluation_search(
            First(), ((1, 2),), (999,), 1, 20, window_size=4,
            candidates_per_window=4)
        self.assertEqual(out["signature_evaluations"], 12)
        self.assertEqual([len(block) for block in calls], [4, 4, 4])
        fractions = [feats[-2] for block in calls for feats in block]
        self.assertEqual(fractions, [i / 12 for i in range(1, 13)])
        self.assertLessEqual(out["peak_live_descriptors"], 4)
        self.assertLessEqual(out["peak_live_features"], 4)

    def test_features_match_global_ordinal_reference_and_gap(self) -> None:
        rows = ((1, 2), (3, 5))
        target = (999, 999)
        observed: list[list[list[float]]] = []
        gaps: list[float] = []

        class Capture:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                observed.append(feats)
                return list(range(len(feats)))

        class CaptureGap(Capture):
            def rank(self, feats: list[list[float]], triples: Any = None,
                     gap: float = 0.0) -> list[int]:
                gaps.append(gap)
                return super().rank(feats, triples, gap)

        bounded_pre_eval.bounded_pre_evaluation_search(
            CaptureGap(), rows, target, 1, 12, window_size=5,
            candidates_per_window=5)
        terms = terminals()
        triples = layer_triples({0: list(terms)}, 1)
        refs = [encode(op, signature(left, rows), signature(right, rows),
                       target, left.size, right.size, 1, 1, index + 1,
                       len(triples))
                for index, (op, left, right) in enumerate(triples)]
        actual = [feat for block in observed for feat in block]
        self.assertEqual(actual, refs)
        self.assertEqual(gaps, [1.0] * 3)

    def test_completed_layer_retention_enables_next_layer(self) -> None:
        class RankAll:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return list(range(len(feats)))

        result = bounded_pre_eval.bounded_pre_evaluation_search(
            RankAll(), ((1, 2),), (999,), 2, 13, window_size=8,
            candidates_per_window=8)
        self.assertEqual(result["signature_evaluations"], 13)
        self.assertEqual(result["retained_operand_vectors"], 9)
        self.assertEqual(result["generated_descriptors"], 20)
        self.assertEqual(result["verdict"], "budget-exhausted")

    def test_window_scoring_and_root_cap_execution(self) -> None:
        class RankAll:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return list(range(len(feats)))

        out = bounded_pre_eval.bounded_pre_evaluation_search(
            RankAll(), ((1, 2),), (999,), 2, 5, window_size=8,
            candidates_per_window=3)
        self.assertEqual(out["generated_descriptors"], 12)
        self.assertEqual(out["scored_descriptors"], 12)
        self.assertEqual(out["skipped_descriptors"], 6)
        self.assertEqual(out["signature_evaluations"], 5)
        self.assertEqual(out["peak_live_descriptors"], 8)
        self.assertEqual(out["retained_operand_vectors"], 0)
        self.assertEqual(out["scored_unexecuted_descriptors"], 7)

    def test_cap_truncation_discards_partial_layer(self) -> None:
        class RankAll:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return list(range(len(feats)))

        budgeted = bounded_pre_eval.bounded_pre_evaluation_search(
            RankAll(), ((1, 2),), (999,), 1, 1, window_size=12,
            candidates_per_window=4)
        signature_capped = bounded_pre_eval.bounded_pre_evaluation_search(
            RankAll(), ((1, 2),), (999,), 1, 10, signature_budget=1,
            window_size=12, candidates_per_window=4)
        for result in (budgeted, signature_capped):
            self.assertEqual(result["layer_widths"], [12])
            self.assertEqual(result["layer_generated_descriptors"], [12])
            self.assertEqual(result["layer_executed_candidates"], [1])
            self.assertEqual(result["retained_operand_vectors"], 0)
            self.assertEqual(result["scored_unexecuted_descriptors"], 11)
            self.assertEqual(result["skipped_descriptors"], 8)
        self.assertTrue(budgeted["budget_exhausted"])
        self.assertFalse(budgeted["signature_budget_exhausted"])
        self.assertTrue(signature_capped["signature_budget_exhausted"])
        self.assertFalse(signature_capped["budget_exhausted"])

    def test_exact_layer_cap_marks_next_layer_stop(self) -> None:
        class RankAll:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return list(range(len(feats)))

        for kwargs, flag in (({"budget": 12}, "budget_exhausted"),
                             ({"budget": 20, "signature_budget": 12},
                              "signature_budget_exhausted")):
            params = dict(kwargs)
            budget = params.pop("budget")
            result = bounded_pre_eval.bounded_pre_evaluation_search(
                RankAll(), ((1, 2),), (999,), 2, budget, **params,
                window_size=12, candidates_per_window=12)
            self.assertEqual(result["layer_widths"], [12])
            self.assertEqual(result["layer_executed_candidates"], [12])
            self.assertGreater(result["retained_operand_vectors"], 0)
            self.assertTrue(result[flag])
            other = ("signature_budget_exhausted" if
                     flag == "budget_exhausted" else "budget_exhausted")
            self.assertFalse(result[other])

    def test_exact_cap_on_last_bound_finishes_naturally(self) -> None:
        class RankAll:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return list(range(len(feats)))

        result = bounded_pre_eval.bounded_pre_evaluation_search(
            RankAll(), ((1, 2),), (999,), 1, 12, window_size=12,
            candidates_per_window=12)
        self.assertEqual(result["layer_executed_candidates"], [12])
        self.assertEqual(result["retained_operand_vectors"], 9)
        self.assertFalse(result["budget_exhausted"])
        self.assertFalse(result["signature_budget_exhausted"])
        self.assertEqual(result["verdict"], "no-solution-within-bound")

    def test_does_not_materialize_complete_layer(self) -> None:
        class RankAll:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return list(range(len(feats)))

        with patch.object(pre_eval, "layer_triples",
                          side_effect=AssertionError("whole layer")):
            out = bounded_pre_eval.bounded_pre_evaluation_search(
                RankAll(), ((1, 2),), (999,), 1, 2, window_size=2,
                candidates_per_window=2)
        self.assertEqual(out["generated_descriptors"], 2)

    def test_full_width_matches_existing_policy_mode(self) -> None:
        class Ascending:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return list(range(len(feats)))

        # A block covering the layer preserves all rank features and ordering.
        first = pre_eval.pre_evaluation_search(
            Ascending(), ((1, 2), (3, 5)), (999, 999), 1, 12,
            alternatives_per_signature=2)
        second = bounded_pre_eval.bounded_pre_evaluation_search(
            Ascending(), ((1, 2), (3, 5)), (999, 999), 1, 12,
            alternatives_per_signature=2, window_size=100)
        self.assertEqual(first["generated_descriptors"],
                         second["generated_descriptors"])
        self.assertEqual(first["retained_operand_vectors"],
                         second["retained_operand_vectors"])
        self.assertEqual(first["equivalent_alternatives_dropped"],
                         second["equivalent_alternatives_dropped"])
        self.assertEqual(first["feature_builds"], second["feature_builds"])
        self.assertEqual(first["signature_evaluations"],
                         second["signature_evaluations"])

    def test_full_layer_retention_preserves_generation_order(self) -> None:
        class ReverseCapture:
            def __init__(self) -> None:
                self.layers: list[list[list[float]]] = []

            def rank(self, feats: list[list[float]],
                     *args: Any) -> list[int]:
                self.layers.append([list(row) for row in feats])
                return list(reversed(range(len(feats))))

        old_policy = ReverseCapture()
        new_policy = ReverseCapture()
        args = (((1, 2), (3, 5)), (999, 999), 2, 13)
        pre_eval.pre_evaluation_search(old_policy, *args)
        bounded_pre_eval.bounded_pre_evaluation_search(
            new_policy, *args, window_size=10000,
            candidates_per_window=10000)
        self.assertEqual(old_policy.layers[0], new_policy.layers[0])
        self.assertEqual(old_policy.layers[1], new_policy.layers[1])

    def test_streaming_delegates_to_historical_implementation(self) -> None:
        with patch.object(pre_eval, "pre_evaluation_search",
                          wraps=pre_eval.pre_evaluation_search) as search:
            result = bounded_pre_eval.bounded_pre_evaluation_search(
                None, ((1, 2),), (999,), 1, 1)
        search.assert_called_once()
        self.assertEqual(result["generated_descriptors"], 2)

    def test_oracle_rejection_retries_candidate_and_checks_alternative(
            self) -> None:
        class FirstThenRest:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return list(range(len(feats)))

        calls = 0

        def oracle(graph: dict[str, Any]) -> bool:
            nonlocal calls
            calls += 1
            return interpret(graph, 3, 4) == 7

        out = bounded_pre_eval.bounded_pre_evaluation_search(
            FirstThenRest(), ((1, 1), (2, 2)), (2, 4), 1, 12,
            oracle=oracle, window_size=12, candidates_per_window=12)
        self.assertTrue(out["solved"])
        self.assertEqual(calls, 2)
        self.assertEqual(out["rejected"][0]["verdict"], "oracle-rejected")
        self.assertEqual(out["oracle_calls"], calls)
        self.assertEqual(interpret(out["graph"], 3, 4), 7)

    def test_bad_window_and_policy_contracts(self) -> None:
        for window in (0, -1, True, 1.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                bounded_pre_eval.bounded_pre_evaluation_search(
                    object(), ((1, 2),), (999,), 1, 3,
                    window_size=window)  # type: ignore[arg-type]

        class Bad:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return [0] * len(feats)

        with self.assertRaisesRegex(ValueError, "candidates_per_window"):
            bounded_pre_eval.bounded_pre_evaluation_search(
                object(), ((1, 2),), (999,), 1, 12, window_size=4,
                candidates_per_window=5)

        with self.assertRaisesRegex(ValueError, "permutation"):
            bounded_pre_eval.bounded_pre_evaluation_search(
                Bad(), ((1, 2),), (999,), 1, 12, window_size=12,
                candidates_per_window=12)


if __name__ == "__main__":
    unittest.main()
