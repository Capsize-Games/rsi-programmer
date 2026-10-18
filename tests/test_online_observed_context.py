"""Focused contracts for the observed-context policy prototype."""
from __future__ import annotations

import copy
import math
import unittest

from rsi_programmer.online_diagnostics import inspect_policy
from rsi_programmer.online_observed_context import ObservedContextPolicy
from rsi_programmer.online_spiking import OnlineSpikingPolicy

ZERO_ROWS = ((0, 0, 0),) * 6
ROWS_A = ((1, 2, 3), (-2, 1, 4), (3, 0, 9), (0, -1, 1),
          (2, 2, 16), (-1, 3, 7))
ROWS_B = ((-3, 4, 5), (2, -2, 0), (0, 3, 9), (1, 1, 4),
          (-2, -1, 9), (4, 0, 16))
FEATURES = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


class ObservedContextPolicyTests(unittest.TestCase):
    """Check context validation, projection isolation, and policy reuse."""

    def test_rows_are_validated_without_advancing_action_rng(self) -> None:
        policy = ObservedContextPolicy(seed=3)
        base = OnlineSpikingPolicy(seed=3)
        rng_state = policy._rng.getstate()
        self.assertEqual(policy._encoder, base._encoder)
        self.assertEqual(rng_state, base._rng.getstate())
        for rows in ([], ROWS_A[:5], ((True, 0, 1),) * 6,
                     ((1.0, 0, 1),) * 6):
            with self.assertRaises(ValueError):
                policy.set_observed_rows(rows)
        self.assertEqual(policy._rng.getstate(), rng_state)
        policy.set_observed_rows(ROWS_A)
        self.assertEqual(policy._rng.getstate(), rng_state)
        with self.assertRaises(RuntimeError):
            policy.set_observed_rows(ROWS_B)

    def test_failed_start_preserves_pending_context_and_random_state(
        self,
    ) -> None:
        policy = ObservedContextPolicy(seed=5)
        policy.set_observed_rows(ROWS_A)
        pending = policy._pending_context[:]
        context_current = policy._context_current[:]
        rng_state = policy._rng.getstate()
        metrics = policy.metrics()
        with self.assertRaises(ValueError):
            policy.start_episode("")
        self.assertEqual(policy._pending_context, pending)
        self.assertEqual(policy._context_current, context_current)
        self.assertEqual(policy._rng.getstate(), rng_state)
        self.assertEqual(policy.metrics(), metrics)
        policy.start_episode("retry")
        policy.end_episode()

    def test_cold_diagnostic_is_pure_with_pending_context(self) -> None:
        policy = ObservedContextPolicy(seed=6)
        policy.set_observed_rows(ROWS_A)
        before = {
            "rng": policy._rng.getstate(),
            "metrics": policy.metrics(),
            "membrane": policy._membrane[:],
            "weights": [row[:] for row in policy._weights],
            "bias": policy._bias[:],
            "baseline": policy._baseline,
            "pending": policy._pending_context[:],
            "current": policy._context_current[:],
        }
        report = inspect_policy(policy)
        self.assertAlmostEqual(sum(report["probabilities"]), 1.0)
        self.assertEqual(policy._rng.getstate(), before["rng"])
        self.assertEqual(policy.metrics(), before["metrics"])
        self.assertEqual(policy._membrane, before["membrane"])
        self.assertEqual(policy._weights, before["weights"])
        self.assertEqual(policy._bias, before["bias"])
        self.assertEqual(policy._baseline, before["baseline"])
        self.assertEqual(policy._pending_context, before["pending"])
        self.assertEqual(policy._context_current, before["current"])

    def test_projection_is_seeded_separately_and_snapshot_pins_its_hash(
        self,
    ) -> None:
        first = ObservedContextPolicy(seed=7)
        same = ObservedContextPolicy(seed=7)
        other = ObservedContextPolicy(seed=8)
        self.assertEqual(first._context_projection, same._context_projection)
        self.assertNotEqual(first._context_projection,
                            other._context_projection)
        self.assertEqual(first._rng.getstate(),
                         OnlineSpikingPolicy(seed=7)._rng.getstate())
        with self.assertRaises(RuntimeError):
            first.start_episode("missing-context")
        first.set_observed_rows(ROWS_A)
        with self.assertRaises(RuntimeError):
            first.snapshot()
        with self.assertRaises(RuntimeError):
            first.restore(OnlineSpikingPolicy(seed=7).snapshot())
        first.start_episode("snapshot")
        first.choose(tuple(FEATURES), ("add", "sub", "mul"))
        first.observe(0.5)
        first.end_episode()
        saved = first.snapshot()
        self.assertNotIn("context_projection", saved)
        digest = saved["config"]["observed_context_projection_sha256"]
        self.assertEqual(digest, first._context_projection_hash)

        restored = ObservedContextPolicy(seed=7)
        restored.restore(saved)
        self.assertEqual(restored.snapshot(), saved)
        before = restored.snapshot()
        corrupt = copy.deepcopy(saved)
        corrupt["config"]["observed_context_projection_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            restored.restore(corrupt)
        self.assertEqual(restored.snapshot(), before)
        restored.set_observed_rows(ROWS_B)
        with self.assertRaises(RuntimeError):
            restored.restore(saved)

    def test_zero_context_matches_baseline_encoder(self) -> None:
        base = OnlineSpikingPolicy(seed=11)
        contextual = ObservedContextPolicy(seed=11)
        contextual.set_observed_rows(ZERO_ROWS)
        base_rates, base_spikes = base._encode(FEATURES)
        context_rates, context_spikes = contextual._encode(FEATURES)
        self.assertEqual(context_rates, base_rates)
        self.assertEqual(context_spikes, base_spikes)

    def test_manual_encoder_reference_for_spiking_and_continuous_modes(
        self,
    ) -> None:
        for spiking in (True, False):
            with self.subTest(spiking=spiking):
                policy = ObservedContextPolicy(seed=13, spiking=spiking)
                policy.set_observed_rows(ROWS_A)
                rates, spikes = policy._encode(FEATURES)
                manual_membrane = [0.0] * 16
                manual_rates = [0.0] * 16
                manual_spikes = 0
                for _ in range(6):
                    for neuron in range(16):
                        base_current = sum(
                            weight * value for weight, value in zip(
                                policy._encoder[neuron], FEATURES
                            )
                        )
                        current = (base_current
                                   + policy._context_current[neuron])
                        membrane = 0.8 * manual_membrane[neuron] + current
                        spike = 0
                        if spiking and membrane >= 1.0:
                            spike = 1
                            membrane = 0.0
                        manual_membrane[neuron] = membrane
                        if spiking:
                            manual_rates[neuron] += spike
                        else:
                            manual_rates[neuron] += math.tanh(membrane)
                        manual_spikes += spike
                expected_rates = [value / 6.0 for value in manual_rates]
                self.assertEqual(rates, expected_rates)
                self.assertEqual(spikes, manual_spikes)

    def test_context_changes_encode_and_is_cleared_at_episode_end(
        self,
    ) -> None:
        first = ObservedContextPolicy(seed=19)
        second = ObservedContextPolicy(seed=19)
        first.set_observed_rows(ROWS_A)
        second.set_observed_rows(ROWS_B)
        rates_a, _ = first._encode(FEATURES)
        rates_b, _ = second._encode(FEATURES)
        self.assertNotEqual(rates_a, rates_b)

        first.start_episode("a")
        first.choose(tuple(FEATURES), ("add", "sub", "mul"))
        first.observe(1.0)
        first.end_episode()
        with self.assertRaises(RuntimeError):
            first.start_episode("stale")
        first.set_observed_rows(ROWS_B)
        first.start_episode("fresh")
        first.end_episode()

    def test_opaque_episode_ids_do_not_change_choices_or_learning(
        self,
    ) -> None:
        snapshots = []
        actions = []
        for episode_id in ("opaque-alpha", "unrelated-metadata-beta"):
            policy = ObservedContextPolicy(seed=29)
            policy.set_observed_rows(ROWS_A)
            policy.start_episode(episode_id)
            actions.append(policy.choose(tuple(FEATURES),
                                         ("add", "sub", "mul")))
            policy.observe(0.73)
            policy.end_episode()
            snapshots.append(policy.snapshot())
        self.assertEqual(actions[0], actions[1])
        self.assertEqual(snapshots[0], snapshots[1])

    def test_frozen_policy_and_single_action_mask_contract(self) -> None:
        frozen = ObservedContextPolicy(seed=23, learning=False)
        initial_hash = frozen.weight_hash()
        frozen.set_observed_rows(ROWS_A)
        frozen.start_episode("frozen")
        self.assertEqual(
            frozen.choose(tuple(FEATURES), ("sub",)), "sub"
        )
        frozen.observe(1.0)
        frozen.end_episode()
        self.assertEqual(frozen.weight_hash(), initial_hash)
        metrics = frozen.metrics()
        self.assertEqual(metrics["observation_scalars"], 18)
        self.assertEqual(metrics["projection_mac"], 18 * 16)
        self.assertEqual(metrics["context_adds"], 6 * 16)
        self.assertEqual(metrics["updates"], 0)


if __name__ == "__main__":
    unittest.main()
