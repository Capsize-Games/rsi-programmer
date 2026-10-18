"""Focused checks for the independent sensing replay implementation."""
from __future__ import annotations

import copy
import unittest

from rsi_programmer.sensing_baseline import SensingBaseline
from rsi_programmer.sensing_benchmark import calibrate as production_calibrate
from rsi_programmer.sensing_benchmark import train_models
from rsi_programmer.sensing_predictor import SensingPredictor
from scripts.audit_sensing import ReferenceBaseline
from scripts.audit_sensing import ReferencePredictor
from scripts.audit_sensing import _audit_calibration
from scripts.audit_sensing import _check_fields
from scripts.audit_sensing import _event_replay
from scripts.audit_sensing import _make_sequences
from scripts.audit_sensing import _make_training_models
from scripts.audit_sensing import _process_values
from scripts.audit_sensing import _threshold
from scripts.audit_sensing import load_protocol
from scripts.sensing_reference import _model_mode


NEURAL_ARMS: tuple[str, ...] = (
    "plastic_snn", "readout_snn", "frozen_snn", "plastic_ann")
BASELINE_ARMS: tuple[str, ...] = ("persistence", "ema", "nlms")
SEEDS: tuple[int, ...] = (0, 1, 2)
SEQUENCE: tuple[float, ...] = (
    0.15, 0.20, 0.24, 0.28, 0.72, 0.76,
    0.70, 0.32, 0.25, 0.55, 0.58, 0.61)


class SensingAuditTests(unittest.TestCase):
    """Check replay isolation, corruption detection, and model equivalence."""

    def test_forecast_at_a_window_cannot_depend_on_future_values(self) -> None:
        first = [0.2] + [0.3] * 299
        altered_future = first[:]
        altered_future[1:] = [0.9] * 299
        a, _ = _process_values(ReferencePredictor(0, "plastic_snn"),
                               first, 0.1, 5)
        b, _ = _process_values(
            ReferencePredictor(0, "plastic_snn"), altered_future, 0.1, 5)
        self.assertEqual(a[0]["prediction_next"],
                         b[0]["prediction_next"])
        self.assertEqual(a[1]["prediction_before"],
                         b[1]["prediction_before"])

    def test_invalid_timestamp_leaves_reference_state_unchanged(self) -> None:
        model = ReferencePredictor(0, "plastic_snn")
        model.start(0.2, 0.25)
        before = copy.deepcopy(model.__dict__)
        with self.assertRaises(ValueError):
            model.advance(0.2, 0.4)
        self.assertEqual(model.__dict__, before)

    def test_calibration_threshold_uses_scores_not_event_labels(self) -> None:
        protocol = load_protocol()
        scores = [0.01, 0.02, 0.03, 0.04]
        threshold_a = _threshold(scores, protocol)
        alarms = [5, 76, 151, 299]
        outcome_a = _event_replay([75, 150, 225], alarms)
        outcome_b = _event_replay([], alarms)
        self.assertEqual(_threshold(scores, protocol), threshold_a)
        self.assertEqual(outcome_a["detected"], 2)
        self.assertEqual(outcome_b["detected"], 0)

    def test_replay_rejects_corrupted_forecast_score_or_hash(self) -> None:
        corruptions: tuple[tuple[str, object], ...] = (
            ("prediction_next", 0.123),
            ("score", 0.987),
            ("model_hash_after", "0" * 64),
        )
        rows, _ = _process_values(
            ReferencePredictor(0, "plastic_snn"), [0.2] * 300, 0.01, 5)
        for field, value in corruptions:
            with self.subTest(field=field, value=value):
                expected = {field: rows[5][field]}
                with self.assertRaises(ValueError):
                    _check_fields({field: value}, expected,
                                  "corrupted-window")

    def test_replay_rejects_corrupted_alarm(self) -> None:
        rows, _ = _process_values(
            ReferencePredictor(0, "plastic_snn"), [0.2] * 300, 0.01, 5)
        expected = {"alarm": rows[5]["alarm"]}
        with self.assertRaises(ValueError):
            _check_fields({"alarm": not rows[5]["alarm"]}, expected,
                          "corrupted-window")

    def test_work_counts_require_exact_integers(self) -> None:
        for invalid in (12.0, 12.0000000000001, True, 13):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _check_fields({"updates": invalid}, {"updates": 12},
                                  "corrupted-work")
        _check_fields({"updates": 12}, {"updates": 12}, "valid-work")

    def test_neural_trajectories_match_independent_reference(self) -> None:
        for arm in NEURAL_ARMS:
            spiking, learning, encoder_learning = _model_mode(arm)
            for seed in SEEDS:
                with self.subTest(arm=arm, seed=seed):
                    production = SensingPredictor(
                        seed=seed, spiking=spiking, learning=learning,
                        encoder_learning=encoder_learning)
                    reference = ReferencePredictor(seed, arm)
                    self.assertEqual(reference.spiking, spiking)
                    self.assertEqual(reference.learning, learning)
                    self.assertEqual(reference.encoder_learning,
                                     encoder_learning)
                    self.assertEqual(production.snapshot(),
                                     reference.snapshot())
                    first_time = 0.2
                    actual = production.start_session(
                        first_time, SEQUENCE[0])
                    expected = reference.start(first_time, SEQUENCE[0])
                    self.assertAlmostEqual(actual, expected, places=12)
                    self.assertEqual(production.weight_hash(),
                                     reference.weight_hash())
                    self._assert_neural_counters(production, reference)

                    for index, value in enumerate(SEQUENCE[1:], start=1):
                        timestamp = 0.2 * (index + 1)
                        actual = production.advance(timestamp, value)
                        expected = reference.advance(timestamp, value)
                        self.assertAlmostEqual(actual, expected, places=12)
                        self.assertEqual(production.weight_hash(),
                                         reference.weight_hash())
                        self._assert_neural_counters(production, reference)

                    production.end_session()
                    reference.end()
                    self.assertEqual(production.snapshot(),
                                     reference.snapshot())

    def test_baseline_trajectories_match_independent_reference(self) -> None:
        for mode in BASELINE_ARMS:
            with self.subTest(mode=mode):
                production = SensingBaseline(mode)
                reference = ReferenceBaseline(mode)
                self.assertEqual(production.snapshot(), reference.snapshot())
                actual = production.start_session(0.2, SEQUENCE[0])
                expected = reference.start(0.2, SEQUENCE[0])
                self.assertAlmostEqual(actual, expected, places=12)
                self.assertEqual(production.weight_hash(),
                                 reference.weight_hash())
                self._assert_baseline_counters(production, reference)

                for index, value in enumerate(SEQUENCE[1:], start=1):
                    timestamp = 0.2 * (index + 1)
                    actual = production.advance(timestamp, value)
                    expected = reference.advance(timestamp, value)
                    self.assertAlmostEqual(actual, expected, places=12)
                    self.assertEqual(production.weight_hash(),
                                     reference.weight_hash())
                    self._assert_baseline_counters(production, reference)

                production.end_session()
                reference.end()
                self.assertEqual(production.weight_hash(),
                                 reference.weight_hash())
                self._assert_baseline_counters(production, reference)
                self.assertEqual(production.snapshot(), reference.snapshot())

    def test_training_and_calibration_replay_preserve_trained_sources(self) \
            -> None:
        protocol = load_protocol()
        features = self._synthetic_features()
        train_sequences, calibration_sequences = _make_sequences(
            protocol, features)
        production, training_rows = train_models(features)
        records_by_key = {
            (row["arm"], row["seed"]): row for row in training_rows}
        reference = _make_training_models(
            protocol, train_sequences, records_by_key)
        for key in production:
            self.assertEqual(production[key].snapshot(),
                             reference[key].snapshot(), key)

        production_boundaries = {
            key: model.snapshot() for key, model in production.items()}
        reference_boundaries = {
            key: model.snapshot() for key, model in reference.items()}
        production_thresholds, calibration_rows = production_calibrate(
            production, features)
        audited_thresholds = _audit_calibration(
            protocol, calibration_rows, reference, calibration_sequences)
        self.assertEqual(production_thresholds, audited_thresholds)
        for key in production:
            self.assertEqual(production[key].snapshot(),
                             production_boundaries[key], key)
            self.assertEqual(reference[key].snapshot(),
                             reference_boundaries[key], key)

    def test_corrupt_training_hash_is_rejected_without_mutation(self) -> None:
        protocol = load_protocol()
        features = self._synthetic_features()
        train_sequences, _ = _make_sequences(protocol, features)
        _, training_rows = train_models(features)
        records_by_key = {
            (row["arm"], row["seed"]): row for row in training_rows}
        key = ("plastic_snn", 0)
        records_by_key[key]["trained_weight_hash"] = "0" * 64
        before = copy.deepcopy(records_by_key[key])
        with self.assertRaises(ValueError):
            _make_training_models(protocol, train_sequences, records_by_key)
        self.assertEqual(records_by_key[key], before)

    @staticmethod
    def _synthetic_features() -> dict[tuple[int, int], list[float]]:
        """Return deterministic bounded values for the frozen fit splits."""
        return {
            (1, speed): [
                0.12 + 0.72 * ((index * 7 + speed) % 23) / 22.0
                for index in range(225)
            ]
            for speed in (20, 35, 50, 65, 80, 95)
        }

    def _assert_neural_counters(self, production: SensingPredictor,
                                reference: ReferencePredictor) -> None:
        """Compare all independent-reference counters after each step."""
        actual = production.metrics()
        for name, value in reference.metrics.items():
            self.assertEqual(actual[name], value, name)

    def _assert_baseline_counters(self, production: SensingBaseline,
                                  reference: ReferenceBaseline) -> None:
        """Compare all independent-reference counters after each step."""
        actual = production.metrics()
        for name, value in reference.metrics.items():
            self.assertEqual(actual[name], value, name)


if __name__ == "__main__":
    unittest.main()
