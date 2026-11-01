"""Useful program behavior and separation of learning from code emission."""
from __future__ import annotations

import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest

import torch

from rsi_programmer.utility_harness import curlee_harness
from rsi_programmer.utility_policy import features, teacher_programs
from rsi_programmer.utility_source import emit_curlee, emit_python


class UtilityProgramTests(unittest.TestCase):
    """Verify executable primitives, held composition and input boundaries."""

    def test_emitters_work_without_neural_packages(self) -> None:
        code = ("from rsi_programmer.utility_source import emit_python; "
                "import sys; assert 'torch' not in sys.modules; "
                "assert 'def process' in emit_python(('positive',))")
        result = subprocess.run([sys.executable, "-S", "-c", code],
                                capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_empty_readings_have_finite_features(self) -> None:
        encoded = features([[]], [[]], 0)
        self.assertEqual(tuple(encoded.shape), (1, 47))
        self.assertTrue(torch.isfinite(encoded).all())

    def test_target_permutations_are_excluded_from_teachers(self) -> None:
        programs = teacher_programs()
        self.assertEqual(len(programs), 649)
        for target in itertools.permutations(
                ("positive", "unique", "ascending")):
            self.assertNotIn(target, programs)

    def test_generated_python_primitives_execute_correctly(self) -> None:
        values = [-2, 0, 3, 3, 1, -2]
        cases = {
            "positive": [3, 3, 1], "nonnegative": [0, 3, 3, 1],
            "even": [-2, 0, -2], "odd": [3, 3, 1],
            "absolute": [2, 0, 3, 3, 1, 2],
            "negate": [2, 0, -3, -3, -1, 2],
            "double": [-4, 0, 6, 6, 2, -4],
            "increment": [-1, 1, 4, 4, 2, -1],
            "unique": [-2, 0, 3, 1],
            "ascending": [-2, -2, 0, 1, 3, 3],
            "descending": [3, 3, 1, 0, -2, -2],
            "reverse": [-2, 1, 3, 3, 0, -2],
            "prefix_sum": [-2, -2, 1, 4, 5, 3],
        }
        for op, expected in cases.items():
            namespace: dict[str, Any] = {"__name__": "utility_test"}
            exec(compile(emit_python((op,)), "<generated>", "exec"),
                 namespace)
            with self.subTest(op=op):
                self.assertEqual(namespace["process"](values), expected)
                self.assertEqual(namespace["process"]([]), [])

    def test_cli_rejects_overwriting_its_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script, source = root / "utility.py", root / "input.json"
            script.write_text(emit_python(("positive",)))
            source.write_text("[-1,0,2]")
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(script), str(source),
                 str(source)], capture_output=True, text=True, timeout=3)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must differ", result.stderr)
            self.assertEqual(source.read_text(), "[-1,0,2]")

    def test_curlee_has_bounded_array_interface_and_assertions(self) -> None:
        program = ("positive", "unique", "ascending")
        source = emit_curlee(program)
        self.assertIn("input: &[Int; 256]", source)
        self.assertIn("output: &mut [Int; 256]", source)
        self.assertNotIn("Vec", source)
        harness = curlee_harness(program, [[], [3, 0, 3]], [[], [3]])
        self.assertIn("return 2;", harness)
        with self.assertRaises(ValueError):
            curlee_harness(program, [[0] * 257], [[]])
        for emitter in (emit_python, emit_curlee):
            with self.subTest(emitter=emitter.__name__):
                with self.assertRaises(ValueError):
                    emitter(("exec",))
