"""Syntax and rejection contracts for executable language backends."""
from __future__ import annotations

import ast
import unittest

from rsi_programmer.construction_state import ConstructionState
from rsi_programmer.target_source import emit_function

ROWS = ((-2, 1, -3), (0, 3, -3), (2, -1, 3))


class TargetSourceTests(unittest.TestCase):
    """Target source preserves ordered references and refuses injection."""

    def test_python_is_a_function_with_ordered_subtraction(self) -> None:
        graph = ConstructionState(ROWS).apply({
            "op": "sub", "left": 0, "right": 1,
        }).graph()
        source = emit_function(graph, "python")
        module = ast.parse(source)
        compile(module, "<generated>", "exec")
        function = module.body[0]
        self.assertIsInstance(function, ast.FunctionDef)
        assert isinstance(function, ast.FunctionDef)
        self.assertEqual(function.name, "solve")
        assignment = function.body[0]
        self.assertIsInstance(assignment, ast.Assign)
        assert isinstance(assignment, ast.Assign)
        expression = assignment.value
        self.assertIsInstance(expression, ast.BinOp)
        assert isinstance(expression, ast.BinOp)
        self.assertIsInstance(expression.op, ast.Sub)
        self.assertEqual(ast.dump(expression.left), "Name(id='x', ctx=Load())")
        self.assertEqual(ast.dump(expression.right),
                         "Name(id='y', ctx=Load())")
        self.assertFalse(any(isinstance(node, (ast.Call, ast.Import,
                                              ast.ImportFrom, ast.Attribute))
                             for node in ast.walk(module)))

    def test_shared_intermediate_is_defined_once_in_both_targets(self) -> None:
        state = ConstructionState(ROWS).apply({
            "op": "add", "left": 0, "right": 1,
        })
        graph = state.apply({"op": "mul", "left": 2, "right": 2}).graph()
        python = emit_function(graph, "python")
        curlee = emit_function(graph, "curlee")
        self.assertEqual(python.count("n0 ="), 1)
        self.assertIn("n1 = n0 * n0", python)
        self.assertEqual(curlee.count("let n0:"), 1)
        self.assertIn("let n1: Int = n0 * n0;", curlee)
        self.assertIn("return n1;", curlee)

    def test_unsafe_names_targets_and_operations_are_rejected(self) -> None:
        graph = ConstructionState(ROWS).apply({
            "op": "sub", "left": 0, "right": 1,
        }).graph()
        for name in ("def", "fn", "solve():\n    import os", "a.b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                emit_function(graph, "python", name)
        with self.assertRaises(ValueError):
            emit_function(graph, "unknown")
        graph["nodes"][0]["op"] = "exec"
        with self.assertRaises(ValueError):
            emit_function(graph, "python")
