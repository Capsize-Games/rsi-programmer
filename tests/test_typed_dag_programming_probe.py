import unittest

from rsi_programmer.probe import (
    CONSTS,
    HELD_OUT,
    MAX_TREE_OPS,
    TRAIN,
    TRAIN_EXPECTED,
    const_terms,
    enumerate_programs,
    interpret,
    lower_to_dag,
    oracle_accepts,
    render_source,
    validate_graph,
)


def oracle(x: int, y: int) -> int:
    return (x + y) * (x + y) + y


class TypedDagProgrammingProbeTests(unittest.TestCase):
    def test_enumerator_assembles_dag_that_matches_held_out_oracle(self):
        expr, examined, _ = enumerate_programs(TRAIN, TRAIN_EXPECTED, MAX_TREE_OPS)
        self.assertIsNotNone(expr)
        self.assertGreater(examined, 0)
        graph = lower_to_dag(expr)
        self.assertLess(len(graph["nodes"]), expr.size)
        self.assertEqual([interpret(graph, x, y) for x, y in TRAIN], [oracle(x, y) for x, y in TRAIN])
        self.assertEqual([interpret(graph, x, y) for x, y in HELD_OUT], [oracle(x, y) for x, y in HELD_OUT])
        self.assertIn("return", render_source(graph))

    def test_validator_rejects_unknown_operation_cycle_and_malformed_arity(self):
        invalid_graphs = [
            {"input_types": {"x": "Int", "y": "Int"}, "nodes": [{"op": "exec", "type": "Int", "args": ["x", "y"]}], "output": 0},
            {"input_types": {"x": "Int", "y": "Int"}, "nodes": [{"op": "add", "type": "Int", "args": [0, "y"]}], "output": 0},
            {"input_types": {"x": "Int", "y": "Int"}, "nodes": [{"op": "add", "type": "Int", "args": ["x"]}], "output": 0},
            {"input_types": {"x": "Int", "y": "Int"}, "nodes": [{"op": "add", "type": "Int", "args": [["x"], "y"]}], "output": 0},
            {"input_types": {"x": "Int", "y": "Int"}, "nodes": [], "output": {"ref": 0}},
        ]
        for graph in invalid_graphs:
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                validate_graph(graph)

    def test_independent_oracle_rejects_valid_but_wrong_graph(self):
        wrong_graph = {"input_types": {"x": "Int", "y": "Int"}, "nodes": [], "output": "x"}
        self.assertFalse(oracle_accepts(wrong_graph, HELD_OUT))

    def test_const_task_needs_consts_to_solve(self):
        rows = ((-3, 1), (-2, 2), (-1, -2), (0, 3), (1, 0), (2, -1))
        target = (0, 1, 2, 3, 4, 5)
        bare, _, _ = enumerate_programs(rows, target, 2)
        self.assertIsNone(bare)
        expr, _, _ = enumerate_programs(rows, target, 2, consts=CONSTS)
        self.assertIsNotNone(expr)
        graph = lower_to_dag(expr)
        self.assertEqual([interpret(graph, x, y) for x, y in rows], [x + 3 for x, _ in rows])

    def test_const_terminals_cover_ir_range(self):
        self.assertEqual([term.left for term in const_terms(CONSTS)], list(CONSTS))
        self.assertEqual(const_terms(), ())

    def test_validator_rejects_bad_constants(self):
        bad_graphs = [
            {"input_types": {"x": "Int", "y": "Int"}, "nodes": [{"op": "const", "type": "Int", "args": ["x"], "value": 1}], "output": 0},
            {"input_types": {"x": "Int", "y": "Int"}, "nodes": [{"op": "const", "type": "Int", "args": [], "value": 99}], "output": 0},
            {"input_types": {"x": "Int", "y": "Int"}, "nodes": [{"op": "const", "type": "Int", "args": []}], "output": 0},
        ]
        for graph in bad_graphs:
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                validate_graph(graph)

    def test_const_graph_renders_and_runs(self):
        graph = {"input_types": {"x": "Int", "y": "Int"}, "nodes": [{"op": "const", "type": "Int", "args": [], "value": 3}, {"op": "add", "type": "Int", "args": ["x", 0]}], "output": 1}
        validate_graph(graph)
        self.assertEqual(interpret(graph, 4, 9), 7)
        self.assertIn("(3)", render_source(graph))


if __name__ == "__main__":
    unittest.main()
