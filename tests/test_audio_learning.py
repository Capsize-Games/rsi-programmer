"""Contracts for the six-band online audio learner."""
from __future__ import annotations

import copy
import unittest

from rsi_programmer.audio_learning import AudioLearner, BAND_EDGES_HZ


class AudioLearnerTests(unittest.TestCase):
    """Check band alignment, causal updates, and boundary evaluation."""

    def test_six_bands_predict_before_scoring_targets(self) -> None:
        learner = AudioLearner(seed=9)
        first = learner.start_session(0.0, (0.2,) * 6)
        self.assertEqual(len(first), 6)
        metrics = learner.metrics()
        self.assertEqual(metrics["predictions"], 6)
        self.assertEqual(metrics["scored_targets"], 0)
        second = learner.advance(0.2, (0.8,) * 6)
        self.assertEqual(len(second), 6)
        metrics = learner.metrics()
        self.assertEqual(metrics["predictions"], 12)
        self.assertEqual(metrics["scored_targets"], 6)
        self.assertEqual(metrics["updates"], 6)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in first + second))

    def test_frozen_hash_stays_fixed_while_state_advances(self) -> None:
        reference = AudioLearner(seed=2, learning=False,
                                 encoder_learning=False)
        hashes = reference.weight_hash()
        reference.start_session(0.0, (0.4,) * 6)
        reference.advance(0.2, (0.9,) * 6)
        self.assertEqual(reference.weight_hash(), hashes)
        self.assertEqual(reference.metrics()["updates"], 0)
        self.assertEqual(reference.metrics()["predictions"], 12)

    def test_matched_reference_uses_same_initial_parameters(self) -> None:
        learner = AudioLearner(seed=13, spiking=False)
        reference = learner.frozen_reference()
        self.assertEqual(learner.weight_hash(), reference.weight_hash())
        self.assertEqual(reference.metrics()["spiking"], False)
        self.assertEqual(reference.metrics()["learning"], False)
        self.assertEqual(reference.metrics()["encoder_learning"], False)

    def test_evaluation_clone_preserves_live_pending_credit(self) -> None:
        learner = AudioLearner(seed=5)
        learner.start_session(0.0, (0.3,) * 6)
        before = copy.deepcopy([band.__dict__ for band in learner._bands])
        evaluator = learner.clone_for_evaluation()
        self.assertTrue(learner.active)
        self.assertEqual([band.__dict__ for band in learner._bands], before)
        self.assertFalse(evaluator.active)
        self.assertTrue(all(not band["learning"]
                            for band in evaluator.metrics()["bands"]))
        self.assertEqual(evaluator.weight_hash(), learner.weight_hash())

    def test_session_boundary_snapshot_and_evaluation_copy(self) -> None:
        learner = AudioLearner(seed=7)
        learner.start_session(3.0, (0.4,) * 6)
        learner.advance(3.2, (0.8,) * 6)
        learner.end_session()
        snapshot = learner.snapshot()
        self.assertEqual(snapshot["band_edges_hz"], list(BAND_EDGES_HZ))
        self.assertEqual(len(snapshot["bands"]), 6)
        evaluator = learner.evaluation_copy()
        self.assertEqual(evaluator.weight_hash(), learner.weight_hash())
        self.assertFalse(evaluator.metrics()["learning"])

    def test_invalid_feature_vector_and_timestamp_are_atomic(self) -> None:
        learner = AudioLearner()
        before = [copy.deepcopy(band.__dict__) for band in learner._bands]
        for features in ((0.5,) * 5, (0.5, 0.5, 0.5, 0.5, 0.5, float("nan")),
                         (0.5, 0.5, 0.5, 0.5, 0.5, 1.1)):
            with self.subTest(features=features):
                with self.assertRaises(ValueError):
                    learner.start_session(0.0, features)
                self.assertEqual([band.__dict__ for band in learner._bands],
                                 before)
        with self.assertRaises(ValueError):
            learner.start_session(float("inf"), (0.5,) * 6)
        self.assertEqual([band.__dict__ for band in learner._bands], before)

    def test_invalid_advance_is_atomic_for_all_bands(self) -> None:
        learner = AudioLearner()
        learner.start_session(1.0, (0.5,) * 6)
        before = [copy.deepcopy(band.__dict__) for band in learner._bands]
        for timestamp, features in (
                (1.2, (0.5, 0.5, 0.5, 0.5, 0.5, float("inf"))),
                (1.0, (0.5,) * 6), (0.9, (0.5,) * 6)):
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(ValueError):
                    learner.advance(timestamp, features)
                self.assertEqual([band.__dict__ for band in learner._bands],
                                 before)


if __name__ == "__main__":
    unittest.main()
