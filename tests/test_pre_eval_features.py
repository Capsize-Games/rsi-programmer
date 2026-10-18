"""Exact-semantics checks for cached pre-evaluation features."""
from __future__ import annotations

import unittest

from rsi_programmer.policy import OP_ORDER, encode
from rsi_programmer.pre_eval_features import FeatureStatistics


class PreEvaluationFeatureStatisticsTests(unittest.TestCase):
    def test_cached_features_match_reference_encoding(self) -> None:
        target = (2, 4, 6, 8)
        left = (2, 3, 6, 9)
        right = (1, 4, 5, 8)
        stats = FeatureStatistics(target)

        actual = stats.encode("add", 1, left, 2, right, 1, 0, 2, 3,
                              5, 12)
        expected = encode("add", left, right, target, 1, 0, 2, 3,
                          5, 12)

        self.assertEqual(actual, expected)

    def test_gap_matches_reference_and_reuses_hit_statistics(self) -> None:
        target = (2, 4, 6, 8)
        left = (2, 3, 6, 9)
        right = (1, 4, 5, 8)
        stats = FeatureStatistics(target)
        stats.encode("add", 1, left, 2, right, 1, 0, 1, 2, 1, 12)

        gap = stats.gap([(1, left), (2, right)])
        best = max(sum(a == b for a, b in zip(sig, target))
                   for sig in (left, right))

        self.assertEqual(gap, 1.0 - best / len(target))
        self.assertEqual(stats.cache_misses, 2)
        self.assertEqual(stats.cache_hits, 2)
        self.assertEqual(stats.target_range_elements_scanned, 8)
        self.assertEqual(stats.operand_feature_elements_scanned, 16)

    def test_repeated_operand_signatures_are_cached(self) -> None:
        stats = FeatureStatistics((0, 1))
        first = stats.operand(1, (0, 1))
        second = stats.operand(1, (0, 1))

        self.assertEqual(first, second)
        self.assertEqual(stats.cache_misses, 1)
        self.assertEqual(stats.cache_hits, 1)
        self.assertEqual(stats.cache_lookups, 2)
        self.assertEqual(stats.operand_feature_elements_scanned, 4)

    def test_reference_features_match_for_all_ops_sizes_and_signatures(
            self) -> None:
        target = (-3, 0, 2, 2, 7)
        signatures = ((-3, 1, 2, 9, 7), (0, 0, 0, 0, 0),
                      (8, -3, 2, 4, 7))
        for op_index, op in enumerate(OP_ORDER):
            for left_size, right_size, size in (
                    (0, 0, 1), (0, 1, 2), (1, 0, 2), (1, 1, 3)):
                left = signatures[op_index % len(signatures)]
                right = signatures[(op_index + size) % len(signatures)]
                actual = FeatureStatistics(target).encode(
                    op, 10, left, 11, right, left_size, right_size, size,
                    4, op_index + 1, len(OP_ORDER))
                expected = encode(op, left, right, target, left_size,
                                  right_size, size, 4, op_index + 1,
                                  len(OP_ORDER))
                with self.subTest(op=op, size=size):
                    self.assertEqual(actual, expected)

    def test_empty_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty target"):
            FeatureStatistics(())


if __name__ == "__main__":
    unittest.main()
