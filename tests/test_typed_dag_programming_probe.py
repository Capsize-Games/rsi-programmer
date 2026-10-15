import unittest

from rsi_programmer.probe import (
    HELD_OUT,
    MAX_TREE_OPS,
    TRAIN,
    TRAIN_EXPECTED,
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


if __name__ == "__main__":
    unittest.main()
