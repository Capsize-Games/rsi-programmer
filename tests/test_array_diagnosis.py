"""Small, weight-free checks for the array witness diagnostic."""
from __future__ import annotations

import unittest

from rsi_programmer.array_policy import ArrayPolicy
from scripts.diagnose_array_construction import diagnose


class ArrayDiagnosisTests(unittest.TestCase):
    def test_reports_coverage_and_all_control_witnesses(self) -> None:
        policy = ArrayPolicy(seed=0)
        policy.eval()
        result = diagnose(policy)
        self.assertEqual(result["teacher_coverage"]["teacher_functions"], 512)
        self.assertGreater(result["teacher_coverage"]["contains_conditions"],
                           0)
        self.assertEqual(len(result["tasks"]), 3)
        for task in result["tasks"].values():
            self.assertTrue(task["grammarable"])
            self.assertGreater(task["teacher_forced"]["action_count"], 0)
            self.assertGreater(task["teacher_forced"]["mean_rank"], 0)
            trace = task["beam_32"]
            self.assertGreaterEqual(len(trace["survival"]), 1)
            self.assertIsNotNone(trace["first_pruning_point"])
            self.assertIn("reason", trace["first_pruning_point"])


if __name__ == "__main__":
    unittest.main()
