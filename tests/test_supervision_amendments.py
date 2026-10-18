"""Focused regression contracts for #66/#70 supervision and ablations."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from rsi_programmer.author import canonical_solution_sigs
from rsi_programmer.author import solution_sigs_by_size_report
from rsi_programmer.author import solution_sigs_report, _label_layer
from scripts.measure_author_ablation import VARIANTS, _weights
from scripts.train import _manifest
from rsi_programmer.author import EXTRA
from rsi_programmer.policy import Rows, SearchPolicy, Vals
from rsi_programmer.probe import OPS, Expr, signature
from rsi_programmer.rich import NEG_CAP, hard_neg, neg_pair


class SupervisionAmendmentTests(unittest.TestCase):
    def test_training_manifest_identifies_actual_supervision(self) -> None:
        policy = SearchPolicy(seed=0)
        evidence = [{"task": "tiny", "included": True}]
        for mode, label in (
                ("reachability",
                 "retained_train_signature_graph_reachability"),
                ("canonical", "canonical_train_solution_subtrees")):
            with self.subTest(mode=mode):
                manifest = _manifest("tiny", policy, "fixture", 0, 1, 2,
                                     mode, evidence)
                self.assertEqual(manifest["training"]["label_semantics"],
                                 label)
                self.assertEqual(manifest["training"]["supervision_mode"],
                                 mode)
                self.assertEqual(manifest["training"]["supervision_evidence"],
                                 evidence)

    def test_reachability_labels_alternative_train_intermediates(self) -> None:
        """Reachability includes both y*y and x+y for train target five."""
        rows = ((1, 2),)
        target = (5,)
        canonical_labels = canonical_solution_sigs(rows, target, 2)
        labels, report = solution_sigs_report(rows, target, 2)

        self.assertFalse(report["truncated"])
        self.assertTrue(report["retained_graph_complete"])
        self.assertIn((4,), labels)
        self.assertIn((3,), labels)
        self.assertTrue(labels - canonical_labels)

    def test_truncated_graph_returns_no_negative_label_universe(self) -> None:
        """An interrupted layer is surfaced instead of mislabeled negative."""
        labels, report = solution_sigs_report(
            ((1, 2),), (5,), 2, candidate_cap=1)

        self.assertEqual(labels, set())
        self.assertTrue(report["truncated"])
        self.assertFalse(report["retained_graph_complete"])

    def test_evidence_cap_and_bound_require_positive_integers(self) -> None:
        """Reject booleans and floats rather than treating them as counts."""
        rows: Rows = ((1, 2),)
        target: Vals = (5,)
        for invalid in (True, 1.5, "2", 0, -1):
            with self.subTest(candidate_cap=invalid):
                with self.assertRaises(ValueError):
                    solution_sigs_by_size_report(
                        rows, target, 2, candidate_cap=invalid)
        for invalid_bound in (True, 1.5, "2", 0, -1):
            with self.subTest(bound=invalid_bound):
                with self.assertRaises(ValueError):
                    solution_sigs_by_size_report(
                        rows, target, invalid_bound)

    def test_later_within_bound_solution_adds_completing_intermediate(
            self) -> None:
        """Longer train solutions contribute labels beyond canonical tree."""
        rows = ((1, 2),)
        target = (3,)
        canonical = canonical_solution_sigs(rows, target, 2)
        labels, report = solution_sigs_report(rows, target, 2)

        self.assertIn((3,), canonical)
        self.assertNotIn((2,), canonical)
        self.assertIn((2,), labels)
        self.assertEqual(report["scope"], "all_retained_layers_through_bound")

    def test_size_specific_labels_match_tiny_tree_completion(self) -> None:
        """A signature at one size does not label another size as positive."""
        rows: Rows = ((1, 2),)
        target: Vals = (3,)
        labels, report = solution_sigs_by_size_report(rows, target, 2)
        self.assertFalse(report["truncated"])

        levels: dict[int, list[Expr]] = {
            0: [Expr("input", "x"), Expr("input", "y")]
        }
        all_exprs: list[Expr] = []
        for size in range(1, 3):
            current: list[Expr] = []
            for left_size in range(size):
                right_size = size - left_size - 1
                for left in levels[left_size]:
                    for right in levels[right_size]:
                        for op in OPS:
                            current.append(Expr(op, left, right, size))
            levels[size] = current
            all_exprs.extend(current)

        expected: dict[int, set[Vals]] = {}
        for root in all_exprs:
            if signature(root, rows) != target:
                continue
            stack = [root]
            while stack:
                node = stack.pop()
                if node.op == "input":
                    continue
                expected.setdefault(node.size, set()).add(
                    signature(node, rows))
                if isinstance(node.left, Expr):
                    stack.append(node.left)
                if isinstance(node.right, Expr):
                    stack.append(node.right)
        self.assertEqual(labels, expected)
        self.assertIn((2,), labels[1])
        self.assertNotIn((2,), labels.get(2, set()))

        x = Expr("input", "x")
        y = Expr("input", "y")
        intermediate = Expr("add", x, y, 1)
        candidate = ("sub", intermediate, x)
        group = _label_layer(
            2, [candidate], rows, target, 2, 0, 1, labels.get(2, set()))
        self.assertEqual(group[0][1], 0)

    def test_bonus_ablations_change_only_named_channels(self) -> None:
        """Exact and gap controls zero only their named author weights."""
        weights = [float(i + 1) for i in range(len(EXTRA))]
        exact = _weights(weights, "no_exact_flag")
        gap = _weights(weights, "no_open_gap")
        both = _weights(weights, "no_both")

        self.assertEqual(VARIANTS[0], "full")
        self.assertEqual(exact[EXTRA.index("exact_flag")], 0.0)
        self.assertEqual(gap[EXTRA.index("open_gap")], 0.0)
        self.assertEqual(both[EXTRA.index("exact_flag")], 0.0)
        self.assertEqual(both[EXTRA.index("open_gap")], 0.0)
        for field in ("nodes_per_bound", "self_apply"):
            index = EXTRA.index(field)
            self.assertEqual(exact[index], weights[index])
            self.assertEqual(gap[index], weights[index])
            self.assertEqual(both[index], weights[index])

    def test_rich_negatives_exclude_longer_solution_intermediate(self
                                                                ) -> None:
        """Rich negatives omit x*y because it completes a longer train path."""
        rows: Rows = ((1, 2),)
        target: Vals = (3,)
        negatives: set[Vals] = set()

        def record(op: str, left: Expr, right: Expr, size: int,
                   train_rows: Rows, expected: Vals, bound: int,
                   ordinal: int, total: int = NEG_CAP
                   ) -> tuple[object, object]:
            if size == 1:
                negatives.add(signature(Expr(op, left, right, size),
                                        train_rows))
            return neg_pair(op, left, right, size, train_rows, expected,
                            bound, ordinal, total)

        with patch("rsi_programmer.rich.neg_pair", side_effect=record):
            hard_neg(rows, target, 2)

        self.assertNotIn((2,), negatives)


if __name__ == "__main__":
    unittest.main()
