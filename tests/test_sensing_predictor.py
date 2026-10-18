"""Contract tests for the causal online sensing predictor."""
from __future__ import annotations

import copy
import math
import unittest

from rsi_programmer.sensing_predictor import SensingPredictor


class SensingPredictorTests(unittest.TestCase):
    """Check time, credit, learning controls, and persistence contracts."""

    def test_prediction_is_scored_only_when_next_value_arrives(self) -> None:
        predictor = SensingPredictor(seed=4)
        first = predictor.start_session(0.0, 0.5)
        before = predictor.weight_hash()
        self.assertEqual(predictor.metrics()["predictions"], 1)
        self.assertEqual(predictor.metrics()["scored_targets"], 0)
        self.assertEqual(predictor.metrics()["updates"], 0)
        second = predictor.advance(0.2, 0.9)
        self.assertTrue(0.0 <= first <= 1.0)
        self.assertTrue(0.0 <= second <= 1.0)
        self.assertEqual(predictor.metrics()["predictions"], 2)
        self.assertEqual(predictor.metrics()["scored_targets"], 1)
        self.assertEqual(predictor.metrics()["updates"], 1)
        self.assertNotEqual(predictor.weight_hash(), before)

    def test_invalid_first_value_is_atomic(self) -> None:
        predictor = SensingPredictor()
        before = predictor.snapshot()
        for timestamp, value in ((float("nan"), 0.5), (0.0, -0.1),
                                 (0.0, 1.1), (True, 0.5)):
            with self.subTest(timestamp=timestamp, value=value):
                with self.assertRaises(ValueError):
                    predictor.start_session(timestamp, value)
                self.assertEqual(predictor.snapshot(), before)

    def test_invalid_advance_is_atomic(self) -> None:
        predictor = SensingPredictor()
        predictor.start_session(1.0, 0.5)
        before = copy.deepcopy(predictor.__dict__)
        for timestamp, value in ((1.0, 0.4), (0.9, 0.4),
                                 (1.2, float("inf")), (1.2, 1.1)):
            with self.subTest(timestamp=timestamp, value=value):
                with self.assertRaises(ValueError):
                    predictor.advance(timestamp, value)
                self.assertEqual(predictor.__dict__, before)

    def test_nonfinite_derived_delta_is_rejected_atomically(self) -> None:
        predictor = SensingPredictor()
        predictor.start_session(-1e308, 0.5)
        before = copy.deepcopy(predictor.__dict__)
        with self.assertRaisesRegex(ValueError, "time delta"):
            predictor.advance(1e308, 0.9)
        self.assertEqual(predictor.__dict__, before)

    def test_sessions_reset_temporal_state_but_keep_parameters(self) -> None:
        predictor = SensingPredictor()
        predictor.start_session(0.0, 0.5)
        predictor.advance(0.2, 0.9)
        trained_hash = predictor.weight_hash()
        predictor.end_session()
        self.assertTrue(all(value == 0.0 for value in predictor._membrane))
        self.assertTrue(all(value == 0.0 for value in predictor._rates))
        self.assertTrue(all(value == 0.0 for value in
                            predictor._rate_weight_eligibility))
        predictor.start_session(10.0, 0.5)
        self.assertEqual(predictor.weight_hash(), trained_hash)
        predictor.end_session()

    def test_spiking_recurrence_matches_frozen_equations(self) -> None:
        predictor = SensingPredictor(seed=2, learning=False)
        predictor._input_weights = [0.5] * 12
        predictor._input_biases = [1.5] * 12
        rates, weight_elig, bias_elig, spikes = predictor._process(0.5, 0.2)
        leak = math.exp(-1.0)
        voltage = (1.0 - leak) * 1.75
        surrogate = max(0.0, 1.0 - abs(voltage - 1.0))
        expected_spike = float(voltage >= 1.0)
        expected_rate = (1.0 - math.exp(-1.0)) * expected_spike
        expected_eu = (1.0 - leak) * 0.5
        self.assertAlmostEqual(predictor._membrane[0],
                               voltage - expected_spike)
        self.assertAlmostEqual(rates[0], expected_rate)
        self.assertAlmostEqual(weight_elig[0],
                               (1.0 - math.exp(-1.0))
                               * surrogate * expected_eu)
        self.assertAlmostEqual(bias_elig[0],
                               (1.0 - math.exp(-1.0))
                               * surrogate * (1.0 - leak))
        self.assertEqual(spikes, 4)

    def test_continuous_recurrence_uses_tanh_derivative(self) -> None:
        predictor = SensingPredictor(seed=2, spiking=False, learning=False)
        predictor._input_weights = [0.5] * 12
        predictor._input_biases = [1.5] * 12
        rates, weight_elig, bias_elig, spikes = predictor._process(0.5, 0.2)
        leak = math.exp(-1.0)
        voltage = (1.0 - leak) * 1.75
        activation = math.tanh(voltage)
        derivative = 1.0 - activation * activation
        expected_eu = (1.0 - leak) * 0.5
        filter_gain = 1.0 - math.exp(-1.0)
        self.assertAlmostEqual(predictor._membrane[0], voltage)
        self.assertAlmostEqual(rates[0], filter_gain * activation)
        self.assertAlmostEqual(weight_elig[0],
                               filter_gain * derivative * expected_eu)
        self.assertAlmostEqual(bias_elig[0],
                               filter_gain * derivative * (1.0 - leak))
        self.assertEqual(spikes, 0)

    def test_encoder_learning_flag_changes_only_input_parameters(self) -> None:
        predictor = SensingPredictor(seed=11, encoder_learning=False)
        encoder_hash = predictor.encoder_hash()
        readout_hash = predictor.weight_hash()
        predictor.start_session(0.0, 0.5)
        predictor.advance(0.2, 0.9)
        self.assertEqual(predictor.encoder_hash(), encoder_hash)
        self.assertNotEqual(predictor.weight_hash(), readout_hash)
        self.assertEqual(predictor.metrics()["input_synapse_visits"], 0)

    def test_saved_credit_uses_pre_update_readout(self) -> None:
        predictor = SensingPredictor(seed=17)
        prediction = predictor.start_session(0.0, 0.5)
        old_readout = predictor._readout_weights[:]
        old_input = predictor._input_weights[:]
        old_biases = predictor._input_biases[:]
        rates = predictor._pending_rates[:]
        weight_elig = predictor._pending_weight_eligibility[:]
        bias_elig = predictor._pending_bias_eligibility[:]
        target = 0.9
        delta = (prediction - target) * prediction * (1.0 - prediction)
        expected_readout = [predictor._clip_parameter(
            value + predictor._clip_update(-0.01 * delta * rate))
            for value, rate in zip(old_readout, rates)]
        expected_input = [predictor._clip_parameter(
            value + predictor._clip_update(-0.001 * delta * old_readout[i]
                                           * weight_elig[i]))
            for i, value in enumerate(old_input)]
        expected_biases = [predictor._clip_parameter(
            value + predictor._clip_update(-0.001 * delta * old_readout[i]
                                           * bias_elig[i]))
            for i, value in enumerate(old_biases)]
        predictor.advance(0.2, target)
        self.assertEqual(predictor._readout_weights, expected_readout)
        self.assertEqual(predictor._input_weights, expected_input)
        self.assertEqual(predictor._input_biases, expected_biases)

    def test_two_step_eligibility_matches_independent_torch_vjp(self) -> None:
        import torch

        predictor = SensingPredictor(seed=5)
        first_value = 0.5
        first_prediction = predictor.start_session(0.0, first_value)
        second_prediction = predictor.advance(0.2, first_prediction)
        old_input = predictor._input_weights[:]
        old_bias = predictor._input_biases[:]
        old_readout = predictor._readout_weights[:]
        old_output_bias = predictor._output_bias
        target = 0.9

        input_weights = torch.tensor(old_input, dtype=torch.float64,
                                     requires_grad=True)
        input_biases = torch.tensor(old_bias, dtype=torch.float64,
                                    requires_grad=True)
        readout = torch.tensor(old_readout, dtype=torch.float64,
                               requires_grad=True)
        output_bias = torch.tensor(old_output_bias, dtype=torch.float64,
                                   requires_grad=True)
        tau = torch.tensor((0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0,
                            5.0, 5.0, 5.0, 5.0), dtype=torch.float64)
        membrane = torch.zeros(12, dtype=torch.float64)
        rates = torch.zeros(12, dtype=torch.float64)
        for value in (first_value, first_prediction):
            leak = torch.exp(torch.tensor(-0.2, dtype=torch.float64) / tau)
            voltage = (leak * membrane + (1.0 - leak)
                       * (input_weights * value + input_biases))
            hard_spike = (voltage >= 1.0).to(torch.float64)
            surrogate = torch.clamp(1.0 - torch.abs(voltage - 1.0),
                                    min=0.0)
            spike = hard_spike + surrogate * (voltage - voltage.detach())
            membrane = voltage - spike
            filter_leak = math.exp(-1.0)
            rates = filter_leak * rates + (1.0 - filter_leak) * spike
        torch_prediction = torch.sigmoid(torch.sum(readout * rates)
                                         + output_bias)
        self.assertAlmostEqual(second_prediction,
                               float(torch_prediction.detach()), places=14)
        loss = 0.5 * (torch_prediction - target) ** 2
        loss.backward()

        predictor.advance(0.4, target)
        expected_readout = [predictor._clip_parameter(
            old_readout[i] - 0.01 * float(readout.grad[i]))
            for i in range(12)]
        expected_input = [predictor._clip_parameter(
            old_input[i] - 0.001 * float(input_weights.grad[i]))
            for i in range(12)]
        expected_bias = [predictor._clip_parameter(
            old_bias[i] - 0.001 * float(input_biases.grad[i]))
            for i in range(12)]
        self.assertTrue(any(a != b for a, b in zip(old_input,
                                                   expected_input)))
        self.assertTrue(any(a != b for a, b in zip(old_bias,
                                                   expected_bias)))
        for actual, expected in zip(predictor._readout_weights,
                                    expected_readout):
            self.assertAlmostEqual(actual, expected, places=12)
        for actual, expected in zip(predictor._input_weights, expected_input):
            self.assertAlmostEqual(actual, expected, places=12)
        for actual, expected in zip(predictor._input_biases, expected_bias):
            self.assertAlmostEqual(actual, expected, places=12)
        expected_output_bias = predictor._clip_parameter(
            old_output_bias - 0.01 * float(output_bias.grad))
        self.assertAlmostEqual(predictor._output_bias,
                               expected_output_bias, places=12)

    def test_parameter_change_counters_ignore_saturated_updates(self) -> None:
        predictor = SensingPredictor(seed=6)
        predictor._input_weights = [2.0] * 12
        predictor._input_biases = [2.0] * 12
        predictor._readout_weights = [2.0] * 12
        predictor._output_bias = 2.0
        predictor._pending_prediction = 0.1
        predictor._pending_rates = [1.0] * 12
        predictor._pending_weight_eligibility = [1.0] * 12
        predictor._pending_bias_eligibility = [1.0] * 12
        predictor._score_pending(1.0)
        metrics = predictor.metrics()
        self.assertEqual(metrics["changed_readout_parameters"], 0)
        self.assertEqual(metrics["changed_input_parameters"], 0)
        self.assertEqual(predictor._readout_weights, [2.0] * 12)
        self.assertEqual(predictor._input_weights, [2.0] * 12)
        self.assertEqual(predictor._input_biases, [2.0] * 12)

    def test_frozen_evaluation_copy_preserves_parameters(self) -> None:
        trained = SensingPredictor(seed=3)
        trained.start_session(0.0, 0.5)
        trained.advance(0.2, 0.9)
        trained.end_session()
        frozen = trained.evaluation_copy()
        before = frozen.weight_hash()
        frozen.start_session(5.0, 0.3)
        frozen.advance(5.2, 0.8)
        self.assertEqual(frozen.weight_hash(), before)
        self.assertEqual(frozen.metrics()["updates"], 0)
        self.assertFalse(frozen.metrics()["learning"])
        self.assertFalse(frozen.metrics()["encoder_learning"])

    def test_snapshot_restore_round_trip_and_atomic_rejection(self) -> None:
        predictor = SensingPredictor(seed=8)
        predictor.start_session(0.0, 0.5)
        predictor.advance(0.2, 0.9)
        predictor.end_session()
        snapshot = predictor.snapshot()
        restored = SensingPredictor(seed=8)
        restored.restore(snapshot)
        self.assertEqual(restored.snapshot(), snapshot)
        damaged = copy.deepcopy(snapshot)
        damaged["input_weights"][0] += 0.1
        before = restored.snapshot()
        with self.assertRaisesRegex(ValueError, "weight hash"):
            restored.restore(damaged)
        self.assertEqual(restored.snapshot(), before)

    def test_snapshot_and_evaluation_copy_require_boundary(self) -> None:
        predictor = SensingPredictor()
        predictor.start_session(0.0, 0.5)
        with self.assertRaises(RuntimeError):
            predictor.snapshot()
        with self.assertRaises(RuntimeError):
            predictor.evaluation_copy()
        with self.assertRaises(RuntimeError):
            predictor.reset_learning()
        predictor.end_session()

    def test_reset_learning_restores_parameters_and_counters(self) -> None:
        predictor = SensingPredictor(seed=9)
        initial = predictor.snapshot()
        predictor.start_session(0.0, 0.5)
        predictor.advance(0.2, 0.9)
        predictor.end_session()
        self.assertNotEqual(predictor.weight_hash(), initial["weight_hash"])
        predictor.reset_learning()
        self.assertEqual(predictor.weight_hash(), initial["weight_hash"])
        self.assertEqual(predictor.metrics()["updates"], 0)


if __name__ == "__main__":
    unittest.main()
