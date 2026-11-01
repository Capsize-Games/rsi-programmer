"""Syntax, independent execution and source-lowering contracts."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest

from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_partial import complete, execution_features
from rsi_programmer.statement_source import curlee_source, python_source
from rsi_programmer.statement_syntax import MAX_TOKENS, analyze, parse

TOTAL = tuple("for if gt value zero set add acc value end end "
              "return acc stop".split())


class StatementProgramTests(unittest.TestCase):
    """Check that tokens construct executable structure and preserve scope."""

    def test_loop_condition_and_return_are_actual_ast_nodes(self) -> None:
        ast = parse(TOTAL)
        self.assertEqual(ast[0][0], "for")
        self.assertEqual(ast[0][1][0][0], "if")
        self.assertEqual(ast[-1][0], "return")
        self.assertEqual(execute(TOTAL, [-5, 0, 3, -2, 4]), 7)
        self.assertEqual(execute(TOTAL, []), 0)
        negative = tuple("lt" if token == "gt" else token for token in TOTAL)
        self.assertEqual(execute(negative, [-5, 0, 3, -2, 4]), -7)

    def test_scope_types_and_explicit_completion_are_enforced(self) -> None:
        for invalid in (("return", "value", "stop"), TOTAL[:-1],
                        (*TOTAL, "set"), ("for", "end", "return", "acc",
                                         "stop"),
                        ("return", "eq", "one", "one", "stop")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse(invalid)
        _, choices = analyze(("for", "if"))
        self.assertIn("gt", choices)
        self.assertNotIn("set", choices)
        with self.assertRaises(ValueError):
            analyze(("set",) * (MAX_TOKENS + 1))

    def test_input_fuel_and_integer_domains_are_bounded(self) -> None:
        for values in ([True], [0] * 257, [2 ** 63]):
            with self.subTest(values=values[:3]):
                with self.assertRaises(ValueError):
                    execute(TOTAL, values)
        square = tuple("for set mul value value end return acc stop".split())
        with self.assertRaises(ValueError):
            execute(square, [2 ** 40])
        division = tuple("return rem one zero stop".split())
        with self.assertRaises(ValueError):
            execute(division, [])

    def test_python_matches_independent_integer_execution(self) -> None:
        programs = [TOTAL, tuple("for set rem value two end return acc stop"
                                 .split()),
                    tuple("return sub length one stop".split())]
        cases = [[], [-5], [-3, 0, 4, -1, 3], list(range(-20, 21))]
        for program in programs:
            namespace: dict[str, Any] = {"__name__": "statement_test"}
            exec(compile(python_source(program), "<generated>", "exec"),
                 namespace)
            for values in cases:
                with self.subTest(program=program, values=values):
                    self.assertEqual(namespace["process"](values),
                                     execute(program, values))
        curlee = curlee_source(TOTAL)
        self.assertIn("for (i in 0..count)", curlee)
        self.assertIn("acc = (acc + value)", curlee)

    def test_generated_scalar_cli_is_standalone(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = root / "total.py"
            script.write_text(python_source(TOTAL))
            source, target = root / "input.json", root / "output.json"
            source.write_text("[-2,0,4,4,-3,7]")
            result = subprocess.run([sys.executable, "-I", "-S", str(script),
                                     str(source), str(target)],
                                    capture_output=True, text=True, timeout=3)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(target.read_text().strip(), "15")

    def test_partial_execution_uses_only_observed_arrays(self) -> None:
        self.assertEqual(complete(TOTAL), TOTAL)
        features = execution_features(TOTAL, [[-2, 0, 3, 4], []], [7, 0])
        self.assertEqual(features, [[1.0, 0.0, 1.0, 1.0],
                                    [0.0, 0.0, 1.0, 1.0]])
        result = execution_features(("return", "rem", "one", "zero"),
                                    [[]], [0])
        self.assertEqual(result, [[0.0, 0.0, 0.0, 0.0]])
