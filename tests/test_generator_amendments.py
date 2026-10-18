"""Task acceptance is independent of the evaluated enumerator (#54)."""
from __future__ import annotations

import random
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from rsi_programmer import families, tasks
from rsi_programmer.probe import Expr, interpret, terminals
from scripts import make_tasks


class GeneratorAmendmentTests(unittest.TestCase):
    def test_default_accepts_generator_witness_without_search(self) -> None:
        x, y = terminals()
        expr = Expr("add", x, y, 1)
        train = [(i, i) for i in range(1, 9)]
        held = [(i, i + 1) for i in range(1, 9)]
        with patch.object(tasks, "_gen_int", return_value=(expr, 1, False)), \
                patch.object(tasks, "_rows", side_effect=[train, held]), \
                patch.object(tasks, "enumerate_programs",
                             side_effect=AssertionError("search filter")):
            task = tasks.generate_task(random.Random(0), "ambiguous",
                                       1, False, False)
        self.assertFalse(task["selection_filter"]["enumerator_filter_applied"])
        select = task["selection_filter"]
        self.assertIsNone(select["enumerator_solved_within_bound"])
        witness = task["generator_witness"]
        for xvalue, yvalue, expected in task["train"] + task["held_out"]:
            self.assertEqual(interpret(witness, xvalue, yvalue), expected)
        self.assertEqual(interpret(witness, 1, 2), 3)

    def test_legacy_filter_is_explicit_and_recorded(self) -> None:
        x, y = terminals()
        expr = Expr("add", x, y, 1)
        with patch.object(tasks, "_gen_int", return_value=(expr, 1, False)), \
                patch.object(tasks, "enumerate_programs",
                             return_value=(expr, 12, 8)) as searched:
            task = tasks.generate_task(random.Random(0), "legacy",
                                       1, False, False, search_filter=True)
        searched.assert_called_once()
        select = task["selection_filter"]
        self.assertTrue(select["enumerator_filter_applied"])
        self.assertTrue(select["enumerator_solution_held_out_confirmed"])
        self.assertEqual(select["enumerator_candidates_tried"], 12)

    def test_family_records_acceptance_mode_and_frozen_witness(self) -> None:
        with patch.object(tasks, "enumerate_programs",
                          side_effect=AssertionError("search filter")):
            family = families.generate_family(34, 4, 2, 0, "independent",
                                               consts=False)
            repeated = families.generate_family(34, 4, 2, 0, "independent",
                                                 consts=False)
        self.assertEqual(family, repeated)
        self.assertEqual(family["generation_acceptance"],
                         "generating_expression")
        for task in family["tasks"]:
            select = task["selection_filter"]
            self.assertFalse(select["enumerator_filter_applied"])
            witness = task["generator_witness"]
            for xvalue, yvalue, expected in task["train"] + task["held_out"]:
                self.assertEqual(interpret(witness, xvalue, yvalue), expected)

    def test_cli_defaults_to_independent_acceptance(self) -> None:
        specs = (("unused.json", 1, 2, "tiny", False, False, 0, False, 2),)
        for arguments, expected in (([], False),
                                    (["--legacy-search-filter"], True)):
            with self.subTest(arguments=arguments), \
                    patch.object(make_tasks, "SPECS", specs), \
                    patch.object(make_tasks, "generate_family",
                                 return_value={"tasks": []}) as generate, \
                    patch.object(make_tasks, "write_family"), \
                    redirect_stdout(StringIO()):
                self.assertEqual(make_tasks.main(arguments), 0)
                self.assertEqual(generate.call_args.kwargs["search_filter"],
                                 expected)


if __name__ == "__main__":
    unittest.main()
