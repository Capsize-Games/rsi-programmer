"""Train-only capped group tests for the v2 benchmark."""
from __future__ import annotations

import unittest
from typing import Any

from rsi_programmer.benchmark_v2_training import build_training_groups
from tests.test_benchmark_v2_runner import _tiny_task


class TrainingGroupTests(unittest.TestCase):
    def test_caps_keep_original_layer_ordinal_and_denom(self) -> None:
        groups, report = build_training_groups(
            [_tiny_task()], positive_cap=1, negative_cap=2,
            sampling_seed=4,
        )
        task = report["groups"][0]
        self.assertEqual(task["status"], "ok")
        self.assertLessEqual(task["selected_counts"]["positive"], 1)
        self.assertLessEqual(task["selected_counts"]["negative"], 2)
        self.assertEqual(len(groups[0]), len(task["selected_indices"]))
        self.assertGreater(task["full_layer_width"], len(groups[0]))
        for row, selected in zip(groups[0], task["selected_candidates"]):
            index = selected["ordinal"]
            width = task["full_layer_width"]
            self.assertAlmostEqual(row[0][10], (index + 1) / width)
            self.assertAlmostEqual(row[0][11], 1 - (index + 1) / width)
            self.assertEqual(selected["feature_vector"], row[0])
            self.assertEqual(selected["root_cost"], task["solving_layer"])

    def test_training_builder_never_reads_held_out_rows(self) -> None:
        class ObservedOnly(dict[str, Any]):
            def __getitem__(self, key: str) -> Any:
                if key == "held_out":
                    raise AssertionError("held-out rows were accessed")
                return super().__getitem__(key)

        groups, report = build_training_groups([ObservedOnly(_tiny_task())])
        self.assertEqual(len(groups), 1)
        self.assertEqual(report["groups"][0]["status"], "ok")
