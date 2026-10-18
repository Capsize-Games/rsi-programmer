"""Independent cost and retention checks for pre-evaluation selection."""
from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import Mock, patch

from rsi_programmer import pre_eval, search
from rsi_programmer.probe import OPS, interpret, signature


class PreEvaluationContractTests(unittest.TestCase):
    def test_selection_changes_executed_work_before_first_solution(
            self) -> None:
        policy = Mock()
        policy.rank.side_effect = lambda feats, *_: [3] + [
            index for index in range(len(feats)) if index != 3]
        rows = ((1, 2), (3, 5))
        target = (3, 8)
        eager_calls = 0

        def counted(expr: Any, inputs: Any) -> tuple[int, ...]:
            nonlocal eager_calls
            if expr.op in OPS:
                eager_calls += 1
            return signature(expr, inputs)

        with patch.object(search, "signature", counted):
            eager = search.guided_search(policy, rows, target, 1, 1)
        candidate = pre_eval._candidate_vector
        with patch.object(pre_eval, "_candidate_vector",
                          wraps=candidate) as executed:
            lazy = pre_eval.pre_evaluation_search(
                policy, rows, target, 1, 1)
        streaming = pre_eval.pre_evaluation_search(
            None, rows, target, 1, 1)

        self.assertEqual(eager_calls, 12)
        self.assertEqual(eager["signature_evaluations"], eager_calls)
        self.assertEqual(executed.call_count, 1)
        self.assertEqual(lazy["signature_evaluations"], executed.call_count)
        self.assertTrue(eager["solved"])
        self.assertTrue(lazy["solved"])
        self.assertFalse(streaming["solved"])
        for graph in (eager["graph"], lazy["graph"]):
            self.assertEqual(interpret(graph, -4, 7), 3)

    def test_completed_layer_retention_ignores_execution_order(self) -> None:
        recorded: list[list[list[list[float]]]] = []
        for reverse in (False, True):
            seen: list[list[list[float]]] = []

            def order(feats: list[list[float]], *args: Any) -> list[int]:
                seen.append(feats)
                indices = list(range(len(feats)))
                return list(reversed(indices)) if reverse else indices

            policy = Mock()
            policy.rank.side_effect = order
            result = pre_eval.pre_evaluation_search(
                policy, ((1, 2), (3, 5)), (999, 999), 2, 13)
            self.assertEqual(result["signature_evaluations"], 13)
            self.assertEqual(result["node_row_operations"], 26)
            self.assertEqual(result["operand_signature_evaluations"], 2)
            self.assertEqual(len(seen), 2)
            recorded.append(seen)
        self.assertEqual(recorded[0], recorded[1])

    def test_secondary_signature_cap_stops_actual_execution(self) -> None:
        result = pre_eval.pre_evaluation_search(
            None, ((1, 2), (3, 5)), (999, 999), 3, 100,
            signature_budget=1)
        self.assertEqual(result["signature_evaluations"], 1)
        self.assertTrue(result["signature_budget_exhausted"])
        self.assertEqual(result["verdict"], "signature-budget-exhausted")
