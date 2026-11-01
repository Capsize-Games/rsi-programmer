"""Counterexample routing and hard bounds for program resynthesis."""
from __future__ import annotations

import unittest
from typing import Any

from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_repair import refine

GOOD = tuple("for if lt value zero set add acc one end end return acc stop"
             .split())
BAD = tuple("gt" if token == "lt" else token for token in GOOD)


class StatementRepairTests(unittest.TestCase):
    """Keep test data outside feedback and preserve rejected programs."""

    def test_counterexample_changes_examples_before_resynthesis(self) -> None:
        calls = []

        def solve(examples: list[tuple[list[int], int]]) -> dict[str, Any]:
            calls.append([(values[:], answer) for values, answer in examples])
            return {"tokens": BAD if len(examples) == 1 else GOOD}

        result = refine(solve, [([-1, 2], 1)], [[1, 2, 3], [-1]],
                        lambda values: sum(value < 0 for value in values))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][-1], ([1, 2, 3], 0))
        self.assertEqual(result["chosen"]["tokens"], GOOD)
        self.assertEqual(result["history"][0]["counterexample"]["actual"], 3)
        self.assertEqual(execute(GOOD, [-3, 2, 0, -2]), 2)

    def test_missing_candidate_uses_only_the_given_query_pool(self) -> None:
        queried = []

        def oracle(values: list[int]) -> int:
            queried.append(values[:])
            return 0

        result = refine(lambda pairs: {"tokens": None}, [([4], 0)],
                        [[1], [2], [3]], oracle, rounds=2)
        self.assertEqual(queried, [[1], [2]])
        self.assertEqual(len(result["history"]), 3)
        self.assertEqual(result["added_examples"], 2)
        self.assertIsNone(result["chosen"]["tokens"])

    def test_consistent_program_is_retained(self) -> None:
        result = refine(lambda pairs: {"tokens": GOOD}, [([-1], 1)],
                        [[1, 2], [-2, 0]],
                        lambda values: sum(value < 0 for value in values))
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["added_examples"], 0)
        self.assertEqual(result["candidate_executions"], 2)

    def test_negative_round_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            refine(lambda pairs: {"tokens": GOOD}, [([-1], 1)], [[]],
                   lambda values: 0, rounds=-1)
