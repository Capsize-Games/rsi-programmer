"""Compare cached features and search with the immutable reference engine."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rsi_programmer.pre_eval import pre_evaluation_search

REFERENCE = (Path(__file__).resolve().parents[1] / "benchmarks" /
             "reference" / "pre_eval_code_v65.py")


def reference_search() -> Any:
    spec = importlib.util.spec_from_file_location("pre_eval_reference",
                                                 REFERENCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pre_evaluation_search


def recording_policy(seen: list[Any], reverse: bool) -> Any:
    def rank(features: list[list[float]], triples: Any,
             gap: float) -> list[int]:
        seen.append((features, gap))
        order = list(range(len(triples)))
        return list(reversed(order)) if reverse else order
    return SimpleNamespace(rank=rank, _steps=1)


class PreEvaluationFeatureContractTests(unittest.TestCase):
    def test_features_gap_graph_and_work_equal_reference(self) -> None:
        original = reference_search()
        rows = ((-2, 3), (0, 1), (2, -1))
        cases = (((1, 1, 1), (1,)), ((1, 1, 1), ()),
                 ((-6, 0, -2), ()), ((999, 999, 999), ()))
        for expected, consts in cases:
            for budget in (0, 1, 12, 13, 80):
                for ordering in (None, False, True):
                    observed: list[Any] = []
                    cached: list[Any] = []
                    left_policy = (None if ordering is None else
                                   recording_policy(observed, ordering))
                    right_policy = (None if ordering is None else
                                    recording_policy(cached, ordering))
                    before = original(left_policy, rows, expected, 3,
                                      budget, consts=consts)
                    after = pre_evaluation_search(
                        right_policy, rows, expected, 3, budget,
                        consts=consts)
                    with self.subTest(expected=expected, budget=budget,
                                      ordering=ordering, consts=consts):
                        self.assertEqual(observed, cached)
                        for key, value in before.items():
                            if key.endswith("_seconds"):
                                continue
                            self.assertEqual(value, after[key], key)

    def test_stream_zero_cap_and_terminal_do_no_feature_scan(self) -> None:
        rows = ((1, 2), (3, 5))
        for policy, target, budget in (
                (None, (99, 99), 13),
                (recording_policy([], False), (99, 99), 0),
                (recording_policy([], False), (1, 3), 20)):
            result = pre_evaluation_search(policy, rows, target, 2, budget)
            self.assertEqual(result["feature_stat_cache_lookups"], 0)
            self.assertEqual(
                result["feature_target_range_elements_scanned"], 0)
            self.assertEqual(result["feature_operand_elements_scanned"], 0)

    def test_actual_scans_match_retained_operand_cache_misses(self) -> None:
        rows = ((1, 2), (3, 5), (4, -2))
        result = pre_evaluation_search(
            recording_policy([], True), rows, (999, 999, 999), 3, 150)
        misses = result["feature_stat_cache_misses"]
        self.assertGreater(misses, 0)
        self.assertEqual(result["feature_operand_elements_scanned"],
                         2 * len(rows) * misses)
        self.assertEqual(result["feature_target_range_elements_scanned"],
                         2 * len(rows))
        self.assertEqual(result["feature_stat_cache_lookups"],
                         misses + result["feature_stat_cache_hits"])
        self.assertGreater(
            result["reference_feature_operand_elements_scanned"],
            result["feature_operand_elements_scanned"])


if __name__ == "__main__":
    unittest.main()
