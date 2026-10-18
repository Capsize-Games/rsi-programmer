"""Tests for paired total-work accounting helpers."""
from __future__ import annotations

import unittest

from rsi_programmer.benchmark_total_work import (
    ENGINE_NAMES,
    _summarize,
    _work_vector,
    engine_order,
)
from rsi_programmer.benchmark_v2_runner import CONTROL_NAMES


class TotalWorkAccountingTests(unittest.TestCase):
    def test_engine_order_is_balanced_and_second_repeat_reverses(self) -> None:
        first_orders = [
            engine_order(0, "trained_snn", f"task-{index}", 8192, 0)
            for index in range(100)
        ]
        self.assertEqual(
            {"baseline", "optimized"},
            {order[0] for order in first_orders},
        )
        for index, first in enumerate(first_orders):
            second = engine_order(
                0, "trained_snn", f"task-{index}", 8192, 1,
            )
            self.assertEqual(first, tuple(reversed(second)))
            self.assertEqual(set(ENGINE_NAMES), set(first))

    def test_work_vector_preserves_distinct_units(self) -> None:
        work = _work_vector(
            {"signature_evaluations": 7, "feature_builds": 3},
            {"validation_calls": 25, "node_row_operations": 90},
            "baseline", 12,
        )
        self.assertEqual(7, work["signature_evaluations"])
        self.assertEqual(3, work["feature_builds"])
        self.assertEqual(25, work["validation_calls_post_search"])
        self.assertEqual(90,
                         work["validation_node_row_operations_post_search"])
        self.assertNotIn("total_work", work)

    def test_matched_success_cost_reports_censoring_at_frozen_points(
            self) -> None:
        budgets = [512, 2048, 8192, 32768]
        records = []
        for engine in ENGINE_NAMES:
            for control in CONTROL_NAMES:
                for budget in budgets:
                    for repeat in (0, 1):
                        for task in range(2):
                            accepted = budget >= 2048 and task == 0
                            records.append({
                                "engine": engine, "control": control,
                                "budget": budget, "repeat": repeat,
                                "seed": task, "task_id": f"task-{task}",
                                "accepted": accepted,
                                "censored": False,
                                "attempt_cpu_ns_including_hashes_"
                                "and_validation":
                                    budget * 10,
                                "attempt_wall_ns_including_hashes_"
                                "and_validation":
                                    budget * 20,
                                "work": {"signature_evaluations": budget},
                            })
        protocol = {
            "evaluation": {"candidate_scan_budgets": budgets},
            "aggregation": {
                "matched_success_targets": [0.25, 0.50, 0.75, 1.00],
            },
            "amortization": {"task_counts": [48, 480, 4800]},
        }
        summary = _summarize(records, protocol, [])
        matched = summary["matched_success_cpu_cost"]
        observed = next(item for item in matched
                        if item["engine"] == "baseline"
                        and item["control"] == "streaming"
                        and item["repeat"] == 0
                        and item["target"] == 0.25)
        self.assertEqual("observed", observed["status"])
        self.assertEqual(2048, observed["budget"])
        censored = next(item for item in matched
                        if item["engine"] == "baseline"
                        and item["control"] == "streaming"
                        and item["repeat"] == 0
                        and item["target"] == 0.75)
        self.assertEqual("censored", censored["status"])
        self.assertIsNone(censored["mean_attempt_cpu_ns"])


if __name__ == "__main__":
    unittest.main()
