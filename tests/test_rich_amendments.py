"""Focused regressions for evidence-safe rich surrogate construction."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from rsi_programmer.policy import SearchPolicy
from rsi_programmer.probe import Expr
from rsi_programmer.rich import _pairs_acc, family_pairs
from rsi_programmer.scale import ScaleUp


class RichAmendmentTests(unittest.TestCase):
    @staticmethod
    def _solution() -> Expr:
        left = Expr("input", "x", None, 0)
        right = Expr("input", "y", None, 0)
        return Expr("add", left, right, 1)

    @staticmethod
    def _task() -> dict[str, Any]:
        return {"train": [[1, 2, 3]], "max_tree_ops": 1}

    def test_complete_negatives_have_disjoint_train_and_selection_sets(
            self) -> None:
        negatives = [([float(i)] * 12, 0) for i in range(4)]
        rich_negatives = [([float(i)] * 4, 0) for i in range(4)]
        report = {"truncated": False, "evidence_status": "complete",
                  "negative_pairs": 4, "negative_cap_reached": False}
        diagnostics: list[dict[str, Any]] = []
        with patch("rsi_programmer.rich.hard_neg_with_report",
                   return_value=(negatives, rich_negatives, report)):
            groups, rich_groups, selections = family_pairs(
                {"t": self._task()}, {"t": self._solution()}, {"t": 0},
                ScaleUp._parts, diagnostics)
        self.assertEqual((len(groups), len(rich_groups), len(selections)),
                         (1, 1, 1))
        self.assertEqual(len(groups[0]), 3)
        self.assertEqual(len(rich_groups[0]), 3)
        self.assertEqual(len(selections[0]), 3)
        trained = {tuple(features) for features, label in groups[0]
                   if label == 0}
        selected = {tuple(features) for features in selections[0][:-1]}
        self.assertEqual(trained, {tuple([2.0] * 12),
                                   tuple([3.0] * 12)})
        self.assertEqual(selected, {tuple([0.0] * 12),
                                    tuple([1.0] * 12)})
        self.assertTrue(trained.isdisjoint(selected))
        self.assertEqual(diagnostics[0]["training_negative_pairs"], 2)
        self.assertEqual(diagnostics[0]["selection_negative_pairs"], 2)

    def test_truncated_and_single_negative_pools_are_omitted(self) -> None:
        one = [([0.0] * 12, 0)]
        one_rich = [([0.0] * 4, 0)]
        reports = [
            ([], [], {"truncated": True, "evidence_status": "truncated",
                      "negative_pairs": 0, "negative_cap_reached": False}),
            (one, one_rich, {"truncated": False,
                             "evidence_status": "complete",
                             "negative_pairs": 1,
                             "negative_cap_reached": False}),
        ]
        tasks = {name: self._task() for name in ("truncated", "single",
                                                  "no-solver")}
        exprs = {"truncated": self._solution(), "single": self._solution(),
                 "no-solver": None}
        diagnostics: list[dict[str, Any]] = []
        with patch("rsi_programmer.rich.hard_neg_with_report",
                   side_effect=reports), \
                patch("rsi_programmer.rich.hard_pos",
                      side_effect=AssertionError(
                          "skipped task expanded positive features")):
            result = family_pairs(tasks, exprs,
                                  {"truncated": 0, "single": 0},
                                  ScaleUp._parts, diagnostics)
        self.assertEqual(result, ([], [], []))
        self.assertEqual([row["task"] for row in diagnostics],
                         ["truncated", "single", "no-solver"])
        self.assertEqual([row["reason"] for row in diagnostics], [
            "reachability_truncated", "insufficient_negative_pairs",
            "no_train_solver"])

    def test_empty_or_positive_only_selection_is_rejected(self) -> None:
        policy = SearchPolicy(seed=0)
        with self.assertRaisesRegex(ValueError, "nonvacuous"):
            _pairs_acc(policy, [], [0.0] * 4)
        with self.assertRaisesRegex(ValueError, "nonvacuous"):
            _pairs_acc(policy, [[[0.0] * 12]], [0.0] * 4)


if __name__ == "__main__":
    unittest.main()
