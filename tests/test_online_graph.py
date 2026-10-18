"""Execution, leakage and complete-fallback contracts for online search."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rsi_programmer import online_graph as engine
from rsi_programmer.probe import interpret, validate_graph


class OnlineGraphTests(unittest.TestCase):
    def test_grammar_lanes_cover_the_same_432_typed_candidates(self) -> None:
        candidates = list(engine.iter_candidates())
        self.assertEqual(len(candidates), 432)
        for op in engine.OPERATORS:
            lane = list(engine.iter_candidates(op))
            self.assertEqual(len(lane), 144)
            self.assertEqual(lane, [graph for graph in candidates
                                   if graph["nodes"][-1]["op"] == op])
        for graph in candidates:
            validate_graph(graph)
        with self.assertRaises(ValueError):
            list(engine.iter_candidates("unknown"))

    def test_symbolic_audit_matches_independent_integer_execution(self
                                                               ) -> None:
        for graph in engine.iter_candidates():
            coefficients = engine.graph_polynomial(graph)
            for x, y in ((-3, 4), (2, -5), (0, 0)):
                expanded = sum(c * x ** xp * y ** yp
                               for xp, yp, c in coefficients)
                self.assertEqual(interpret(graph, x, y), expanded)

    def test_feedback_follows_all_observed_executions(self) -> None:
        events: list[str] = []
        feature_rows: list[tuple[float, ...]] = []

        def choose(features: tuple[float, ...],
                   available: tuple[str, ...]) -> str:
            self.assertEqual(len(features), 6)
            self.assertEqual(available, engine.OPERATORS)
            events.append("choose")
            feature_rows.append(features)
            return "add"

        def execute(graph: dict, x: int, y: int) -> int:
            events.append("execute")
            return interpret(graph, x, y)

        policy = SimpleNamespace(
            start_episode=lambda identity: events.append("start"),
            choose=choose,
            observe=lambda reward: events.append("reward"),
            end_episode=lambda: events.append("end"),
        )
        with patch.object(engine, "interpret", side_effect=execute):
            out = engine.run_episode(
                policy, ((1, 2, 10000), (3, 4, 10000)), budget=2,
                episode_id="opaque-lifecycle-only")
        self.assertEqual(events, ["start", "choose", "execute",
                                  "execute", "reward", "choose",
                                  "execute", "execute", "reward", "end"])
        self.assertEqual(feature_rows[0], (1, 0, 0, 0, 1, 0))
        self.assertEqual(feature_rows[1][1:4], (1, 0, 0))
        self.assertEqual(feature_rows[1][5], 0.5)
        for attempt in out["attempts"]:
            error = sum(min(1, abs(pred - 10000) / 10001)
                        for pred in attempt["predictions"]) / 2
            self.assertAlmostEqual(attempt["reward"], 1 - error)
        self.assertEqual(out["candidate_evaluations"], 2)
        self.assertEqual(out["generated_candidates"], 2)
        self.assertEqual(out["interpreter_rows"], 4)
        self.assertEqual(out["node_row_operations"], 12)
        self.assertEqual(out["graph_validation_calls"], 4)

    def test_complete_streaming_fallback_and_hard_budget(self) -> None:
        for budget in (0, 1, 144, 432):
            out = engine.run_episode(None, ((1, 2, 99999),),
                                     budget=budget)
            self.assertFalse(out["observed_solved"])
            self.assertEqual(out["candidate_evaluations"], budget)
            self.assertEqual(out["grammar_exhausted"], budget == 432)
        graphs = [row["graph"] for row in out["attempts"]]
        self.assertEqual(graphs, list(engine.iter_candidates()))

    def test_observed_stop_cannot_learn_or_retry_from_held_rejection(self
                                                                  ) -> None:
        rewards: list[float] = []
        policy = SimpleNamespace(
            start_episode=lambda identity: None,
            choose=lambda features, available: "add",
            observe=rewards.append,
            end_episode=lambda: None,
        )
        out = engine.run_episode(policy, ((0, 0, 2),), budget=144)
        self.assertEqual(out["candidate_evaluations"], 1)
        self.assertEqual(rewards, [1.0])
        verdict = engine.check_result(
            out["graph"], ((1, 2, 999),), ((1, 1, 4),))
        self.assertFalse(verdict["accepted"])
        self.assertEqual(rewards, [1.0])
        self.assertEqual(out["candidate_evaluations"], 1)

    def test_held_row_match_alone_does_not_replace_exact_identity(self
                                                               ) -> None:
        graph = next(engine.iter_candidates())
        held = ((1, 2, interpret(graph, 1, 2)),)
        verdict = engine.check_result(graph, held, ((0, 0, 10000),))
        self.assertTrue(verdict["held_correct"])
        self.assertFalse(verdict["polynomial_correct"])
        self.assertFalse(verdict["accepted"])

    def test_invalid_inputs_and_action_do_not_execute(self) -> None:
        for rows, budget in (((), 1), (((1, 2, True),), 1),
                             (((1, 2, 3),), True),
                             (((1, 2, 3),), 433)):
            with self.assertRaises(ValueError):
                engine.run_episode(None, rows, budget=budget)
        bad = SimpleNamespace(start_episode=lambda identity: None,
                              choose=lambda features, available: "div")
        with patch.object(engine, "interpret") as execute:
            with self.assertRaisesRegex(ValueError, "unavailable"):
                engine.run_episode(bad, ((1, 2, 3),), budget=1)
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
