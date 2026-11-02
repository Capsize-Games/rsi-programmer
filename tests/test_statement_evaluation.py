"""Independent acceptance must require actual nonempty held evidence."""
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

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

    def test_timed_out_candidate_is_rejected_without_losing_trial(self
                                                               ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            decision = {"id": "positive_total", "chosen": {
                "tokens": ("return", "zero", "stop")}}
            with patch("scripts.measure_useful_program.run_cli",
                       side_effect=subprocess.TimeoutExpired([], 5)):
                row, = evaluate(Path(temp), [decision], [[1], [2]])
            self.assertFalse(row["accepted"])
            self.assertEqual(row["python_records"][0]["exit_code"], 124)
            self.assertTrue(row["execution_stopped_on_error"])

    def test_malformed_output_is_rejected_without_losing_trial(self) -> None:
        def bad_output(root: Path, inputs: Path, output: Path, name: str,
                       result_name: str) -> subprocess.CompletedProcess[str]:
            (output / result_name).write_text("not valid JSON")
            return subprocess.CompletedProcess([], 0, "", "")

        with tempfile.TemporaryDirectory() as temp:
            decision = {"id": "positive_total", "chosen": {
                "tokens": ("return", "zero", "stop")}}
            with patch("scripts.measure_useful_program.run_cli", bad_output):
                row, = evaluate(Path(temp), [decision], [[1]])
            self.assertFalse(row["accepted"])
            self.assertIsNotNone(row["python_records"][0]["output_error"])
