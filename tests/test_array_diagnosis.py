"""Small, weight-free checks for the array witness diagnostic."""
from __future__ import annotations

import unittest

from rsi_programmer.array_policy import ArrayPolicy
from scripts.diagnose_array_construction import diagnose, _tree_features


class ArrayDiagnosisTests(unittest.TestCase):
    def test_condition_depth_distinguishes_sequence_from_nesting(self) -> None:
        sequential = tuple(("for if gt value zero append value end "
                            "if gt value one append value end end "
                            "return output stop").split())
        nested = tuple(("for if gt value zero if gt value one append "
                        "value end end end return output stop").split())
        sequential_features = _tree_features(sequential)
        nested_features = _tree_features(nested)
        self.assertEqual(sequential_features["if_count"], 2)
        self.assertEqual(sequential_features["max_condition_depth"], 1)
        self.assertEqual(nested_features["max_condition_depth"], 2)

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
