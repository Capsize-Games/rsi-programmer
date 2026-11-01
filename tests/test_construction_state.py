"""Independent checks for observed-only sequential graph construction."""
from __future__ import annotations

import copy
import unittest

from rsi_programmer.construction_check import check_graph
from rsi_programmer.construction_state import ConstructionState
from rsi_programmer.online_graph import graph_polynomial
from rsi_programmer.probe import interpret

OBSERVED = ((-2, 1, -1), (0, 3, 3), (2, -1, 1))
HELD = ((4, 5, 9), (-7, 2, -5))
SUM = ((0, 1, 1), (1, 0, 1))
ADD = {"op": "add", "left": 0, "right": 1}


class ConstructionStateTests(unittest.TestCase):
    """The controller must choose valid edits and explicitly choose STOP."""

    def test_exact_match_does_not_stop_or_mutate_its_parent(self) -> None:
        initial = ConstructionState(OBSERVED)
        added = initial.apply(ADD)
        self.assertTrue(added.observed_match())
        self.assertFalse(added.stopped)
        self.assertEqual(initial.node_count, 0)
        stopped = added.apply({"op": "stop"})
        self.assertTrue(stopped.stopped)
        self.assertFalse(added.stopped)
        with self.assertRaises(ValueError):
            stopped.apply(ADD)

    def test_cached_execution_matches_the_independent_interpreter(
        self,
    ) -> None:
        state = ConstructionState(OBSERVED)
        for action in (ADD, {"op": "mul", "left": 2, "right": 0},
                       {"op": "sub", "left": 3, "right": 1}):
            state = state.apply(action)
            outputs = tuple(interpret(state.graph(), x, y)
                            for x, y, _ in OBSERVED)
            self.assertEqual(state.vectors()[state.output_slot], outputs)
        graph = state.graph()
        graph["nodes"].clear()
        self.assertEqual(state.node_count, 3)
        self.assertEqual(len(state.graph()["nodes"]), 3)

    def test_tree_budget_counts_reused_subexpressions(self) -> None:
        state = ConstructionState(OBSERVED, max_tree_ops=2).apply(ADD)
        with self.assertRaisesRegex(ValueError, "budget"):
            state.apply({"op": "mul", "left": 2, "right": 2})
        self.assertEqual(state.node_count, 1)

    def test_bad_actions_and_specifications_are_rejected(self) -> None:
        state = ConstructionState(OBSERVED)
        for action in ({"op": "stop", "left": 0},
                       {"op": "exec", "left": 0, "right": 1},
                       {"op": "add", "left": True, "right": 1},
                       {"op": "add", "left": 2, "right": 0}):
            with self.subTest(action=action), self.assertRaises(ValueError):
                state.apply(action)
        for rows in ((), ((True, 0, 1),), ((1, 2),)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                ConstructionState(rows)  # type: ignore[arg-type]
        for bound in (True, 0, 9):
            with self.assertRaises(ValueError):
                ConstructionState(OBSERVED, bound)

    def test_independent_acceptance_needs_both_rows_and_exact_identity(
        self,
    ) -> None:
        graph = ConstructionState(OBSERVED).apply(ADD).graph()
        self.assertTrue(check_graph(graph, OBSERVED, HELD, SUM)["accepted"])
        wrong_held = ((4, 5, 10),)
        self.assertFalse(check_graph(
            graph, OBSERVED, wrong_held, SUM
        )["accepted"])
        self.assertFalse(check_graph(
            graph, OBSERVED, HELD, ((1, 0, 1),)
        )["accepted"])
        self.assertFalse(check_graph(graph, OBSERVED, (), SUM)["accepted"])
        self.assertFalse(check_graph(
            graph, OBSERVED, HELD, SUM, 0
        )["accepted"])

    def test_checker_rejects_budget_bypass_and_unsupported_operations(
        self,
    ) -> None:
        state = ConstructionState(OBSERVED, max_tree_ops=4).apply(ADD)
        state = state.apply({"op": "mul", "left": 2, "right": 2})
        graph = state.graph()
        self.assertFalse(check_graph(
            graph, OBSERVED, HELD, graph_polynomial(graph), 2
        )["valid"])
        bad = copy.deepcopy(graph)
        bad["nodes"][0]["op"] = "exec"
        self.assertFalse(check_graph(bad, OBSERVED, HELD, SUM)["valid"])
