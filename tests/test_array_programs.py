"""Semantic and scope boundaries for list-producing statement programs."""
from __future__ import annotations

import unittest

from rsi_programmer.array_execution import execute
from rsi_programmer.array_syntax import analyze, parse
from rsi_programmer.array_tasks import CONTROL, TASKS, oracle
from rsi_programmer.array_curriculum import probes, teachers


class ArrayProgramTests(unittest.TestCase):
    def test_controls_match_independent_specs(self) -> None:
        for name in TASKS:
            for values in ([], [-3, -1, 0, 1, 2, 3], [100] * 256):
                self.assertEqual(execute(CONTROL[name], values),
                                 oracle(name, values))

    def test_two_writes_use_updated_output_state(self) -> None:
        tokens = tuple(("for append value append outlen end "
                        "return output stop").split())
        self.assertEqual(execute(tokens, [7, -2]), [7, 1, -2, 3])
        self.assertEqual(len(execute(tokens, [1] * 256)), 512)

    def test_third_write_and_scalar_return_are_rejected(self) -> None:
        for text in ("for append value append value append value end "
                     "return output stop",
                     "for append value end return value stop"):
            with self.assertRaises(ValueError):
                parse(tuple(text.split()))

    def test_invalid_numeric_execution_is_rejected(self) -> None:
        bad = tuple("for append rem value zero end return output stop".split())
        with self.assertRaises(ValueError):
            execute(bad, [1])
        for values in ([True], [1] * 257, [2 ** 63]):
            with self.assertRaises(ValueError):
                execute(CONTROL[TASKS[0]], values)

    def test_prefix_choices_preserve_array_return_type(self) -> None:
        prefix = tuple("for append value end return".split())
        self.assertEqual(analyze(prefix)[1], ("output",))

    def test_curriculum_excludes_complete_task_functions(self) -> None:
        rows = probes()
        excluded = {tuple(tuple(oracle(name, x)) for x in rows)
                    for name in TASKS}
        signatures = {tuple(tuple(execute(tokens, x)) for x in rows)
                      for tokens in teachers()}
        self.assertEqual(len(signatures), 512)
        self.assertTrue(signatures.isdisjoint(excluded))

    def test_all_teachers_keep_structural_write_bound(self) -> None:
        for tokens in teachers():
            self.assertLessEqual(tokens.count("append"), 2)
            self.assertIsNotNone(parse(tokens))
