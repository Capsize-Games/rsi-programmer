"""Focused causality checks for bounded replay memory."""
from __future__ import annotations

import copy
import unittest

from rsi_programmer.sensing_memory import MemorySensingPredictor


def _train_replay_predictor() -> MemorySensingPredictor:
    model = MemorySensingPredictor(seed=4)
    model.configure_memory("fit", [0.2 + i / 1000 for i in range(150)])
    return model


class MemorySensingPredictorTests(unittest.TestCase):
    def test_replay_preserves_live_fast_state_and_counts_memory_work(
            self) -> None:
        model = _train_replay_predictor()
        replay_states: list[tuple[dict[str, object], dict[str, object]]] = []
        replay = model._replay

        def capture_replay() -> None:
            before = {
                key: copy.deepcopy(getattr(model, key))
                for key in ("_last_timestamp", "_pending_prediction",
                            "_pending_rates", "_pending_weight_eligibility",
                            "_pending_bias_eligibility", "_membrane",
                            "_rates",
                            "_membrane_weight_eligibility",
                            "_membrane_bias_eligibility",
                            "_rate_weight_eligibility",
                            "_rate_bias_eligibility")
            }
            replay()
            after = {
                key: copy.deepcopy(getattr(model, key))
                for key in before
            }
            replay_states.append((before, after))

        model._replay = capture_replay  # type: ignore[method-assign]
        value = 0.3
        model.start_session(0.2, value)
        for index in range(1, 301):
            value = 0.3 + (index % 17) / 100.0
            model.advance((index + 1) * 0.2, value)

        self.assertEqual(len(replay_states), 1)
        self.assertEqual(replay_states[0][0], replay_states[0][1])
        metrics = model.metrics()
        self.assertEqual(metrics["replay_passes"], 1)
        self.assertEqual(metrics["replay_targets"], 149)
        self.assertGreaterEqual(metrics["updates"], 300 + 149)
        self.assertGreaterEqual(metrics["scored_targets"], 300 + 149)

    def test_evaluator_is_frozen_and_does_not_change_source_memory(
            self) -> None:
        model = _train_replay_predictor()
        model.start_session(0.2, 0.4)
        for index in range(1, 8):
            model.advance((index + 1) * 0.2, 0.4 + index / 100.0)
        memory = list(model._memory_rows)
        metrics = model.metrics()
        hashes = model.weight_hash()

        evaluator = copy.deepcopy(model)
        evaluator.end_session()
        evaluator = evaluator.evaluation_copy()
        self.assertEqual(evaluator._memory_mode, "disabled")
        self.assertFalse(evaluator._replay_enabled)
        evaluator.start_session(2.0, 0.45)
        evaluator.advance(2.2, 0.46)

        self.assertEqual(model._memory_rows, memory)
        self.assertEqual(model.metrics(), metrics)
        self.assertEqual(model.weight_hash(), hashes)

    def test_recent_memory_contains_only_causally_observed_values(
            self) -> None:
        model = MemorySensingPredictor(seed=1)
        model.configure_memory("recent")
        model.start_session(0.2, 0.25)
        self.assertEqual(model._memory_rows, [0.25])
        model.advance(0.4, 0.35)
        self.assertEqual(model._memory_rows, [0.25, 0.35])
        self.assertEqual(len(model._memory_rows), 2)

    def test_fit_memory_requires_exact_capacity_and_valid_values(self) -> None:
        model = MemorySensingPredictor()
        with self.assertRaises(ValueError):
            model.configure_memory("fit", [0.5] * 149)
        with self.assertRaises(ValueError):
            model.configure_memory("fit", [0.5] * 149 + [float("nan")])

    def test_frozen_predictor_never_writes_recent_memory_or_replays(
            self) -> None:
        model = MemorySensingPredictor(learning=False)
        model.configure_memory("recent")
        model.start_session(0.2, 0.3)
        for index in range(1, 302):
            model.advance((index + 1) * 0.2, 0.3 + (index % 10) / 100.0)
        self.assertEqual(model._memory_rows, [])
        self.assertEqual(model.metrics()["replay_passes"], 0)
        self.assertEqual(model.metrics()["updates"], 0)

    def test_replay_respects_disabled_encoder_learning(self) -> None:
        model = MemorySensingPredictor(encoder_learning=False)
        model.configure_memory("fit", [0.2 + i / 1000 for i in range(150)])
        model.start_session(0.2, 0.3)
        for index in range(1, 301):
            model.advance((index + 1) * 0.2, 0.3 + (index % 13) / 100.0)
        self.assertEqual(model.metrics()["replay_passes"], 1)
        self.assertEqual(model.metrics()["changed_input_parameters"], 0)
