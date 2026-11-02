"""Shared branch inputs retain scope, typing and verified provenance."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from rsi_programmer.statement_branch_library import instantiate, pattern
from rsi_programmer.statement_branch_library import qualify_branches
from rsi_programmer.statement_execution import execute
from rsi_programmer.stateful_tasks import oracle

ROOT = Path(__file__).resolve().parents[1]
DONORS = ROOT / "artifacts/statement-context-repair-v1-20261101"


class StatementBranchTests(unittest.TestCase):
    """These supplied arguments test the composer, not neural invention."""

    def test_shared_negation_is_used_in_condition_and_assignment(self
                                                               ) -> None:
        library = qualify_branches(DONORS)
        self.assertEqual(len(library["branches"]), 8)
        branch = next(row for row in library["branches"]
                      if row["donor"] == "maximum_floor_zero")
        tokens = instantiate(pattern(branch, branch),
                             ("sub", "zero", "value"))
        for values in ([], [0], [-7, -3], [-2, 5, -9, 1], [100] * 256):
            self.assertEqual(execute(tokens, values),
                             oracle("largest_magnitude", values))

    def test_subtracting_shared_excess_caps_each_positive_value(self) -> None:
        library = qualify_branches(DONORS)
        branch = next(row for row in library["branches"]
                      if row["donor"] == "positive_total")
        tokens = instantiate(pattern(branch, branch, "sub"),
                             ("sub", "value", "two"))
        for values in ([], [-4, 0, 1, 2, 9], [100] * 256, [-100] * 256):
            self.assertEqual(execute(tokens, values),
                             oracle("positive_values_capped_at_two_total",
                                    values))

    def test_arguments_cannot_escape_expression_depth_or_scope(self) -> None:
        library = qualify_branches(DONORS)
        branch = next(row for row in library["branches"]
                      if row["donor"] == "positive_square_total")
        with self.assertRaises(ValueError):
            instantiate(pattern(branch, branch), ("add", "value", "one"))
        with self.assertRaises(ValueError):
            instantiate(pattern(branch, branch), ("end", "return", "acc"))
        one_hole = ("for", "set", "$arg", "end", "return", "acc", "stop")
        with self.assertRaises(ValueError):
            instantiate(one_hole, ("value", "set", "zero"))

    def test_mutated_donor_source_is_rejected_before_mining(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "donors"
            shutil.copytree(DONORS, root)
            path = root / "maximum_floor_zero/python/clean_readings.py"
            path.write_text(path.read_text() + "\n# Changed after checking.\n")
            with self.assertRaises(ValueError):
                qualify_branches(root)
