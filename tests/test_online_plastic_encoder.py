"""Focused tests for decision-local plastic input projections."""
from __future__ import annotations

import copy
import math
import unittest

from rsi_programmer.online_observed_context import ObservedContextPolicy
from rsi_programmer.online_plastic_encoder import OnlinePlasticEncoderPolicy

ROWS = ((1, 2, 3), (2, 1, 4), (3, 2, 5), (4, 3, 6), (5, 4, 7),
        (6, 5, 8))
FEATURES = (1.0, 0.0, 0.0, 0.0, 0.25, 0.5)
AVAILABLE = ("add", "sub", "mul")


class OnlinePlasticEncoderTests(unittest.TestCase):
    """Verify update math, persistence and inherited compatibility."""

    def _active(
        self, *, spiking: bool = True, encoder_learning: bool = True
    ) -> OnlinePlasticEncoderPolicy:
        """Create a policy with one active observed-context episode."""
        policy = OnlinePlasticEncoderPolicy(
            seed=17, spiking=spiking, encoder_learning=encoder_learning
        )
        policy.set_observed_rows(ROWS)
        policy.start_episode("test")
        return policy

    def test_fixed_encoder_flag_matches_observed_policy(self) -> None:
        """The frozen path preserves inherited matrices and sampled action."""
        old = ObservedContextPolicy(seed=9)
        fixed = OnlinePlasticEncoderPolicy(seed=9, encoder_learning=False)
        self.assertEqual(old._encoder, fixed._encoder)
        self.assertEqual(old._context_projection, fixed._context_projection)
        self.assertEqual(old._rng.getstate(), fixed._rng.getstate())
        old.set_observed_rows(ROWS)
        fixed.set_observed_rows(ROWS)
        old.start_episode("same")
        fixed.start_episode("same")
        self.assertEqual(
            old.choose(FEATURES, AVAILABLE),
            fixed.choose(FEATURES, AVAILABLE),
        )
        self.assertEqual(old._membrane, fixed._membrane)
        self.assertEqual(old._rng.getstate(), fixed._rng.getstate())

    def test_master_learning_false_keeps_both_encoders_fixed(self) -> None:
        """The master learning switch freezes readout and both projections."""
        policy = OnlinePlasticEncoderPolicy(seed=5, learning=False)
        policy._weights[0] = [0.5] * 16
        before_encoder = copy.deepcopy(policy._encoder)
        before_context = copy.deepcopy(policy._context_projection)
        policy.set_observed_rows(ROWS)
        policy.start_episode("frozen")
        policy.choose(FEATURES, AVAILABLE)
        self.assertIsNone(policy._pending_hidden_signal)
        policy.observe(1.0)
        policy.end_episode()
        self.assertEqual(policy._encoder, before_encoder)
        self.assertEqual(policy._context_projection, before_context)

    def test_scalar_lif_eligibility_matches_hand_recurrence(self) -> None:
        """The factorized local trace matches the six-tick equations."""
        policy = self._active()
        policy._encoder = [[0.0] * 6 for _ in range(16)]
        policy._context_projection = [[0.0] * 16 for _ in range(18)]
        policy._membrane = [0.75] * 16
        _, _ = policy._encode(list(FEATURES))
        membrane = 0.75
        derivative_membrane = 0.0
        derivative_rate = 0.0
        for _ in range(6):
            current = 0.8 * membrane
            derivative_current = 0.8 * derivative_membrane + 1.0
            spike = int(current >= 1.0)
            psi = 0.3 * max(0.0, 1.0 - abs(current - 1.0))
            derivative_spike = psi * derivative_current
            derivative_membrane = (
                (1 - spike) * derivative_current
                - current * derivative_spike
            )
            membrane = 0.0 if spike else current
            derivative_rate += derivative_spike / 6.0
        self.assertAlmostEqual(
            policy._last_rate_derivative[0], derivative_rate, places=14
        )
        self.assertEqual(policy._last_rate_derivative,
                         [policy._last_rate_derivative[0]] * 16)

    def test_continuous_trace_matches_finite_difference(self) -> None:
        """The continuous control uses the exact tanh derivative."""
        policy = self._active(spiking=False)
        policy._encoder = [[0.0] * 6 for _ in range(16)]
        policy._context_projection = [[0.0] * 16 for _ in range(18)]
        policy._membrane = [0.2] * 16
        policy._encode(list(FEATURES))
        actual = policy._last_rate_derivative[0]

        def rate_for(current_offset: float) -> float:
            membrane = 0.2
            total = 0.0
            for _ in range(6):
                membrane = 0.8 * membrane + current_offset
                total += math.tanh(membrane) / 6.0
            return total

        epsilon = 1e-6
        numeric = (rate_for(epsilon) - rate_for(-epsilon)) / (2 * epsilon)
        self.assertAlmostEqual(actual, numeric, places=8)

    def test_first_zero_readout_signal_does_not_change_encoder(self) -> None:
        """Zero initialized readout gives no hidden credit on first choice."""
        policy = self._active()
        policy._encoder = [[0.0] * 6 for _ in range(16)]
        policy._context_projection = [[0.0] * 16 for _ in range(18)]
        policy._membrane = [0.5] * 16
        before_encoder = copy.deepcopy(policy._encoder)
        before_context = copy.deepcopy(policy._context_projection)
        policy.choose(FEATURES, AVAILABLE)
        policy.observe(1.0)
        self.assertEqual(policy._encoder, before_encoder)
        self.assertEqual(policy._context_projection, before_context)
        policy.end_episode()

    def test_nonzero_hidden_signal_changes_both_projections(self) -> None:
        """Reward updates both inputs after the readout has learned."""
        policy = self._active()
        policy._encoder = [[0.0] * 6 for _ in range(16)]
        policy._context_projection = [[0.0] * 16 for _ in range(18)]
        policy._membrane = [0.5] * 16
        policy._weights[0] = [0.4] * 16
        policy._weights[1] = [-0.2] * 16
        before_encoder = copy.deepcopy(policy._encoder)
        before_context = copy.deepcopy(policy._context_projection)
        policy.choose(FEATURES, AVAILABLE)
        signal = policy._pending_hidden_signal
        self.assertIsNotNone(signal)
        self.assertTrue(any(value != 0.0 for value in signal or []))
        policy.observe(1.0)
        self.assertNotEqual(policy._encoder, before_encoder)
        self.assertNotEqual(policy._context_projection, before_context)
        policy.end_episode()

    def test_projection_update_is_bounded_and_context_refreshes(self) -> None:
        """Updates honor per-step bounds and affect the next encoding."""
        policy = self._active()
        policy._encoder = [[0.0] * 6 for _ in range(16)]
        policy._context_projection = [[0.0] * 16 for _ in range(18)]
        policy._membrane = [0.5] * 16
        policy._weights[0] = [5.0] * 16
        policy._weights[1] = [-5.0] * 16
        old_context_hash = policy.encoder_hash()
        policy.choose(FEATURES, AVAILABLE)
        policy.observe(1.0)
        self.assertNotEqual(policy.encoder_hash(), old_context_hash)
        self.assertTrue(all(abs(value) <= 5.0
                            for row in policy._encoder for value in row))
        self.assertTrue(all(abs(value) <= 5.0
                            for row in policy._context_projection
                            for value in row))
        first_macs = policy._metrics[
            "plastic_context_projection_macs"
        ]
        policy.choose(FEATURES, AVAILABLE)
        self.assertEqual(
            policy._metrics["plastic_context_projection_macs"] - first_macs,
            18 * 16,
        )
        policy.observe(0.0)
        policy.end_episode()

    def test_encoder_reset_restores_both_initial_projections(self) -> None:
        """Reset restores both seeded input matrices and leaves RNG intact."""
        policy = self._active()
        initial_encoder = copy.deepcopy(policy._initial_encoder)
        initial_context = copy.deepcopy(policy._initial_context_projection)
        policy._weights[0] = [0.5] * 16
        policy._encoder = [[0.0] * 6 for _ in range(16)]
        policy._context_projection = [[0.0] * 16 for _ in range(18)]
        policy._membrane = [0.5] * 16
        policy.choose(FEATURES, AVAILABLE)
        policy.observe(1.0)
        policy.end_episode()
        self.assertNotEqual(policy._encoder, initial_encoder)
        self.assertNotEqual(policy._context_projection, initial_context)
        current_rng = policy._rng.getstate()
        policy.reset_learning()
        self.assertEqual(policy._encoder, initial_encoder)
        self.assertEqual(policy._context_projection, initial_context)
        self.assertEqual(policy._rng.getstate(), current_rng)

    def test_snapshot_round_trip_and_atomic_rejection(self) -> None:
        """Learned matrices persist; invalid hashes are atomic."""
        policy = self._active()
        policy._weights[0] = [0.5] * 16
        policy.choose(FEATURES, AVAILABLE)
        policy.observe(1.0)
        policy.end_episode()
        payload = policy.snapshot()
        restored = OnlinePlasticEncoderPolicy(seed=17)
        restored.restore(payload)
        self.assertEqual(restored.encoder_hash(), policy.encoder_hash())
        self.assertEqual(restored.weight_hash(), policy.weight_hash())
        self.assertEqual(restored._encoder, policy._encoder)
        self.assertEqual(restored._context_projection,
                         policy._context_projection)
        before = restored.snapshot()
        invalid = copy.deepcopy(payload)
        invalid["context_projection"][0][0] += 0.25
        with self.assertRaisesRegex(ValueError, "encoder hash"):
            restored.restore(invalid)
        self.assertEqual(restored.snapshot(), before)

    def test_evaluation_copy_isolated_and_learning_disabled(self) -> None:
        """Boundary evaluation preserves weights and isolates the source."""
        policy = OnlinePlasticEncoderPolicy(seed=31)
        policy._weights[0] = [0.25] * 16
        frozen = policy.evaluation_copy()
        before = policy.snapshot()
        frozen.set_observed_rows(ROWS)
        frozen.start_episode("eval")
        frozen.choose(FEATURES, AVAILABLE)
        frozen.observe(1.0)
        frozen.end_episode()
        self.assertEqual(policy.snapshot(), before)
        self.assertFalse(frozen._learning)
        self.assertFalse(frozen._encoder_learning)

    def test_failed_episode_start_preserves_pending_signal(self) -> None:
        """Invalid restart attempts cannot erase an outstanding decision."""
        policy = self._active()
        policy._weights[0] = [0.25] * 16
        policy.choose(FEATURES, AVAILABLE)
        pending = policy._pending_hidden_signal[:]
        derivative = policy._last_rate_derivative[:]
        with self.assertRaises(RuntimeError):
            policy.start_episode("duplicate")
        self.assertEqual(policy._pending_hidden_signal, pending)
        self.assertEqual(policy._last_rate_derivative, derivative)
        policy.observe(0.0)
        policy.end_episode()

    def test_fresh_snapshot_round_trip(self) -> None:
        """Initial integer projections restore from parsed float matrices."""
        policy = OnlinePlasticEncoderPolicy(seed=37)
        payload = policy.snapshot()
        restored = OnlinePlasticEncoderPolicy(seed=37)
        restored.restore(payload)
        self.assertEqual(restored.encoder_hash(), policy.encoder_hash())
        self.assertEqual(restored.snapshot(), payload)

    def test_feedback_and_episode_boundaries_clear_credit(self) -> None:
        """Eligibility is current-decision only and never crosses episodes."""
        policy = self._active()
        policy._weights[0] = [0.25] * 16
        policy.choose(FEATURES, AVAILABLE)
        self.assertTrue(any(policy._last_rate_derivative))
        policy.observe(0.5)
        self.assertEqual(policy._last_rate_derivative, [0.0] * 16)
        policy.choose(FEATURES, AVAILABLE)
        policy.observe(0.5)
        policy.end_episode()
        self.assertIsNone(policy._pending_hidden_signal)
        self.assertEqual(policy._last_rate_derivative, [0.0] * 16)


if __name__ == "__main__":
    unittest.main()
