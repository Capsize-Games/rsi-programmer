"""Regressions for lazy candidate evaluation in pre-evaluation search."""
from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from rsi_programmer import pre_eval
from rsi_programmer.probe import interpret


class PreEvaluationSearchTests(unittest.TestCase):
    def test_policy_ranks_full_layer_before_candidate_evaluation(self) -> None:
        evaluations: list[tuple[str, tuple[int, ...], tuple[int, ...]]]
        evaluations = []
        original = pre_eval._candidate_vector

        def count(op: str, left: tuple[int, ...],
                  right: tuple[int, ...]) -> tuple[int, ...]:
            evaluations.append((op, left, right))
            return original(op, left, right)

        class First:
            def rank(self, feats: list[list[float]], triples: list[Any],
                     gap: float) -> list[int]:
                self.width = len(triples)
                self.features = feats
                self.at_rank = len(evaluations)
                return list(range(len(triples)))

        policy = First()
        with patch.object(pre_eval, "_candidate_vector", count):
            out = pre_eval.pre_evaluation_search(
                policy, ((1, 2),), (999,), 1, 2)

        self.assertEqual(policy.width, 12)
        self.assertEqual(policy.at_rank, 0)
        self.assertEqual(len(policy.features), 12)
        self.assertEqual(out["signature_evaluations"], 2)
        self.assertEqual(out["generated_descriptors"], 12)
        self.assertEqual(out["scored_descriptors"], 12)
        self.assertEqual(out["feature_builds"], 12)
        self.assertEqual(out["verdict"], "budget-exhausted")

    def test_zero_budget_does_not_rank_or_evaluate_candidates(self) -> None:
        class NeverRank:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                raise AssertionError("zero budget must skip policy ranking")

        out = pre_eval.pre_evaluation_search(
            NeverRank(), ((1, 2),), (999,), 3, 0)

        self.assertEqual(out["signature_evaluations"], 0)
        self.assertEqual(out["generated_descriptors"], 0)
        self.assertEqual(out["feature_builds"], 0)
        self.assertEqual(out["verdict"], "budget-exhausted")

    def test_budget_stops_without_expanding_future_layers(self) -> None:
        out = pre_eval.pre_evaluation_search(
            None, ((1, 2),), (999,), 3, 1)

        self.assertEqual(out["signature_evaluations"], 1)
        self.assertEqual(out["generated_descriptors"], 2)
        self.assertEqual(out["verdict"], "budget-exhausted")

    def test_budget_equal_to_one_layer_leaves_later_work_pending(
            self) -> None:
        out = pre_eval.pre_evaluation_search(
            None, ((1, 2),), (999,), 3, 12,
            alternatives_per_signature=2)

        self.assertEqual(out["signature_evaluations"], 12)
        self.assertEqual(out["generated_descriptors"], 12)
        self.assertEqual(out["verdict"], "budget-exhausted")
        self.assertEqual(out["retained_operand_vectors"], 12)
        self.assertEqual(out["operand_signature_evaluations"], 2)

    def test_constant_terminal_is_a_valid_result(self) -> None:
        out = pre_eval.pre_evaluation_search(
            None, ((4, 7), (8, 3)), (5, 5), 0, 0, consts=(5,))

        self.assertTrue(out["solved"])
        self.assertEqual(out["graph"]["nodes"][0]["value"], 5)
        self.assertEqual(out["signature_evaluations"], 0)

    def test_oracle_rejection_retries_equivalent_train_matches(self) -> None:
        class Preferred:
            def rank(self, feats: list[list[float]], triples: list[Any],
                     gap: float) -> list[int]:
                return [0, 3] + [i for i in range(len(triples))
                                if i not in {0, 3}]

        calls: list[int] = []

        def held_out(graph: dict[str, Any]) -> bool:
            calls.append(1)
            return interpret(graph, 3, 4) == 7

        out = pre_eval.pre_evaluation_search(
            Preferred(), ((1, 1), (2, 2)), (2, 4), 1, 12,
            oracle=held_out)

        self.assertTrue(out["solved"])
        self.assertEqual(out["oracle_calls"], 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(out["rejected"][0]["verdict"], "oracle-rejected")
        self.assertEqual(interpret(out["graph"], 3, 4), 7)

    def test_candidate_work_uses_cached_operands(self) -> None:
        with patch("rsi_programmer.probe.signature",
                   side_effect=AssertionError("recursive evaluation")):
            out = pre_eval.pre_evaluation_search(
                None, ((1, 2), (2, 3)), (3, 5), 1, 12)

        self.assertTrue(out["solved"])
        self.assertEqual(out["signature_evaluations"], 4)
        self.assertGreaterEqual(out["operand_signature_evaluations"], 2)

    def test_malformed_policy_orders_are_rejected(self) -> None:
        class Duplicate:
            def rank(self, feats: list[list[float]], *args: Any) -> list[int]:
                return [0] * len(feats)

        with self.assertRaisesRegex(ValueError, "permutation"):
            pre_eval.pre_evaluation_search(
                Duplicate(), ((1, 2),), (999,), 1, 12)

    def test_inputs_must_be_aligned_and_empty_rows_are_rejected(self) -> None:
        for rows, expected in (((), ()), (((1, 2),), (1, 2))):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                pre_eval.pre_evaluation_search(None, rows, expected, 1, 5)


if __name__ == "__main__":
    unittest.main()
