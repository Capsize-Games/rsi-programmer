"""Small synthetic checks for old/new control-row equivalence."""
from __future__ import annotations

import copy
import unittest
from typing import Any

from scripts.compare_online_plastic_controls import (
    _expected_rows,
    compare_row_streams,
)


class PlasticControlComparisonTests(unittest.TestCase):
    """Check matching and representative semantic-corruption rejection."""

    def _pair(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build one first-row control pair from the frozen schedule."""
        expected, _task = next(_expected_rows())
        common: dict[str, Any] = dict(expected)
        common.update({
            "graph": None,
            "observed_solved": False,
            "accepted": False,
            "held_correct": False,
            "polynomial_correct": False,
            "held_rows_executed": 0,
            "candidate_evaluations": 1,
            "generated_candidates": 1,
            "interpreter_rows": 6,
            "node_row_operations": 18,
            "graph_validation_calls": 6,
            "decision_calls": 1,
            "feedback_calls": 1,
            "budget_exhausted": False,
            "grammar_exhausted": False,
            "first_lane_correct": True,
            "rng_draw_offset_before": 0,
            "rng_draw_offset_after": 1,
            "attempts": [{
                "action": "add",
                "features": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                "graph": None,
                "predictions": [0, 0, 0, 0, 0, 0],
                "error": 1.0,
                "reward": 0.0,
                "observed_exact": False,
            }],
        })
        old = copy.deepcopy(common)
        old["weight_before"] = "readout-before"
        old["weight_after"] = "readout-after"
        new = copy.deepcopy(common)
        new["readout_hash_before"] = "readout-before"
        new["readout_hash_after"] = "readout-after"
        return old, new

    def test_identical_control_rows_match(self) -> None:
        """One valid pair matches in exact fixture order."""
        old, new = self._pair()
        self.assertEqual(
            compare_row_streams([old], [new], expected_count=1), 1
        )

    def test_action_mutation_is_rejected(self) -> None:
        """Changing the selected action breaks equivalence."""
        old, new = self._pair()
        new["attempts"][0]["action"] = "mul"
        with self.assertRaisesRegex(ValueError, "attempt action differs"):
            compare_row_streams([old], [new], expected_count=1)

    def test_readout_hash_mutation_is_rejected(self) -> None:
        """The old readout hash maps to the new readout hash fields."""
        old, new = self._pair()
        new["readout_hash_after"] = "changed"
        with self.assertRaisesRegex(ValueError, "weight_after/readout_hash"):
            compare_row_streams([old], [new], expected_count=1)

    def test_missing_control_row_is_rejected(self) -> None:
        """An incomplete control stream cannot pass a complete-row check."""
        old, new = self._pair()
        with self.assertRaisesRegex(ValueError, "missing control row"):
            compare_row_streams([], [new], expected_count=1)


if __name__ == "__main__":
    unittest.main()
