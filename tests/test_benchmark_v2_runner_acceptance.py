"""Independent post-search acceptance tests for benchmark v2."""
from __future__ import annotations

import unittest

from rsi_programmer.benchmark_v2_runner import validate_graph_exact
from tests.test_benchmark_v2_runner import _tiny_task


class ExactAcceptanceTests(unittest.TestCase):
    def test_equivalent_dag_is_accepted_by_exact_polynomial(self) -> None:
        task = _tiny_task()
        graph = {
            "input_types": {"x": "Int", "y": "Int"},
            "nodes": [{"op": "add", "type": "Int",
                       "args": ["y", "x"]}],
            "output": 0,
        }
        result = validate_graph_exact(graph, task)
        self.assertTrue(result["train_valid"])
        self.assertTrue(result["held_out_valid"])
        self.assertTrue(result["exact_function_match"])

    def test_matching_rows_do_not_override_exact_polynomial(self) -> None:
        task = _tiny_task()
        task["generator"]["program"] = {"op": "input", "name": "x"}
        task["train"] = [[0, 0, 0], [1, 0, 1]]
        task["held_out"] = [[2, 0, 2]]
        graph = {
            "input_types": {"x": "Int", "y": "Int"},
            "nodes": [{"op": "add", "type": "Int",
                       "args": ["x", "y"]}],
            "output": 0,
        }
        result = validate_graph_exact(graph, task)
        self.assertTrue(result["train_valid"])
        self.assertTrue(result["held_out_valid"])
        self.assertFalse(result["exact_function_match"])
