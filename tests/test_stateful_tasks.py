"""Distinguish language capacity from frozen-learner generalization."""
from __future__ import annotations

import unittest

from rsi_programmer.stateful_tasks import WITNESSES, oracle
from rsi_programmer.statement_curriculum import teachers
from rsi_programmer.statement_execution import execute


class StatefulTaskTests(unittest.TestCase):
    """Check independent oracles and confirm that witness controls can run."""

    def test_positive_counter_primitive_has_independent_answers(self) -> None:
        self.assertEqual(oracle("positive_count", [-2, 0, 1, 5, 1]), 3)
        self.assertEqual(oracle("positive_count", []), 0)
        self.assertEqual(oracle("positive_count", [100] * 256), 256)

    def test_oracles_have_independent_golden_answers(self) -> None:
        values = [-2, 5, 0, 3, -1, 1]
        expected = [5, -2, 5, 5, 2]
        for name, answer in zip(WITNESSES, expected):
            with self.subTest(name=name):
                self.assertEqual(oracle(name, values), answer)
                self.assertEqual(oracle(name, []), 0)

    def test_supplied_controls_match_oracles_on_distinguishing_cases(
            self) -> None:
        cases = [[], [0], [-4, -2, 0], [1, -2, 0], [9, 1, 3, 2],
                 [-2, 5, 0, 3, -1, 1], [100] * 256]
        for name, source in WITNESSES.items():
            for values in cases:
                with self.subTest(name=name, values=values[:8]):
                    self.assertEqual(execute(tuple(source.split()), values),
                                     oracle(name, values))

    def test_targets_do_not_match_any_training_function_on_declared_probes(
            self) -> None:
        old_ids = ["positive_total", "negative_count", "even_total",
                   "positive_square_total", "odd_count", "above_two_count"]
        programs, probes = teachers(old_ids)
        signatures = {tuple(execute(program, values) for values in probes)
                      for program in programs}
        for name in WITNESSES:
            with self.subTest(name=name):
                signature = tuple(oracle(name, values) for values in probes)
                self.assertNotIn(signature, signatures)

    def test_broader_curriculum_excludes_all_eleven_functions(self) -> None:
        old_ids = ["positive_total", "negative_count", "even_total",
                   "positive_square_total", "odd_count", "above_two_count"]
        programs, probes = teachers(old_ids + list(WITNESSES), stateful=True)
        self.assertGreater(len(programs), 366)
        from rsi_programmer.statement_curriculum import oracle as all_oracles
        excluded = {tuple(all_oracles(name, values) for values in probes)
                    for name in old_ids + list(WITNESSES)}
        for program in programs:
            signature = tuple(execute(program, values) for values in probes)
            self.assertNotIn(signature, excluded)

    def test_return_variants_keep_components_but_exclude_complete_targets(
            self) -> None:
        from rsi_programmer.statement_curriculum import oracle as all_oracles
        from rsi_programmer.statement_syntax import parse
        names = ["positive_total", "negative_count", "even_total",
                 "positive_square_total", "odd_count", "above_two_count",
                 *WITNESSES]
        programs, probes = teachers(names, stateful=True, return_variants=True)
        excluded = {tuple(all_oracles(name, values) for values in probes)
                    for name in names}
        component = ("if", ("gt", ("value",), ("acc",)),
                     (("set", ("value",)),))
        retained = False
        for program in programs:
            signature = tuple(execute(program, values) for values in probes)
            self.assertNotIn(signature, excluded)
            for node in parse(program):
                if node[0] == "for" and component in node[1]:
                    retained = True
        self.assertTrue(retained)
