"""Whole-function separation and independent suite-oracle behavior."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from rsi_programmer.statement_curriculum import oracle as statement_goal
from rsi_programmer.statement_curriculum import teachers
from rsi_programmer.statement_execution import execute as execute_statement
from rsi_programmer.utility_ops import execute
from rsi_programmer.utility_policy import teacher_programs
from rsi_programmer.utility_suite import oracle

ROOT = Path(__file__).resolve().parents[1]


class UtilitySuiteTests(unittest.TestCase):
    """Protect task oracles and exclusion of complete requested behavior."""

    def test_useful_oracles_have_independent_expected_answers(self) -> None:
        values = [-2, 0, 3, 3, 2, -2]
        expected = {
            "clean_positive": [2, 3], "clean_nonnegative": [0, 2, 3],
            "negative_magnitudes": [2, 2],
            "distinct_magnitudes_descending": [3, 2, 0],
            "positive_running_total": [3, 6, 8],
            "absolute_running_total": [2, 2, 5, 8, 10, 12],
            "distinct_positive_even": [2],
            "recent_distinct_positive": [2, 3],
            "positive_scaled_total": [6, 12, 16],
        }
        for name, answers in expected.items():
            with self.subTest(name=name):
                self.assertEqual(oracle(name, values), answers)

    def test_requested_compositions_are_removed_from_training(self) -> None:
        spec = json.loads((ROOT / "fixtures/useful_compositions_v1.json")
                          .read_text())
        excluded = [tuple(task["exclusion_program"]) for task in spec["tasks"]]
        programs = teacher_programs(excluded=excluded)
        self.assertEqual(len(programs), 641)
        for program in excluded:
            self.assertNotIn(program, programs)
        for task in spec["tasks"]:
            for values, expected in task["observed"]:
                self.assertEqual(execute(tuple(task["exclusion_program"]),
                                         values), expected)

    def test_statement_teachers_are_function_disjoint_on_declared_probes(
            self) -> None:
        names = [task["id"] for task in json.loads(
            (ROOT / "fixtures/statement_tasks_v1.json").read_text())["tasks"]]
        programs, probes = teachers(names)
        excluded = {tuple(statement_goal(name, values) for values in probes)
                    for name in names}
        self.assertEqual(len(programs), 366)
        for program in programs:
            signature = tuple(execute_statement(program, values)
                              for values in probes)
            self.assertNotIn(signature, excluded)
