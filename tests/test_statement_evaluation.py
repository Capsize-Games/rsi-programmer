"""Independent acceptance must require actual nonempty held evidence."""
from __future__ import annotations

from pathlib import Path
import unittest

from scripts.evaluate_statement_decisions import evaluate


class StatementEvaluationTests(unittest.TestCase):
    """Prevent vacuous acceptance of generated programs and empty trials."""

    def test_no_held_rows_cannot_count_as_a_pass(self) -> None:
        decision = {"id": "positive_total",
                    "chosen": {"tokens": ("return", "zero", "stop")}}
        with self.assertRaises(ValueError):
            evaluate(Path("unused"), [decision], [])

    def test_no_program_decisions_cannot_count_as_a_trial(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(Path("unused"), [], [[1, 2]])
