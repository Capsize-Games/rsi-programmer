"""Contracts for reward-modulated online spiking lane policy."""

from __future__ import annotations

import copy
import unittest

from rsi_programmer.online_spiking import OnlineSpikingPolicy


class OnlineSpikingPolicyTests(unittest.TestCase):
    """Exercise lifecycle, masking, plasticity, and persistence contracts."""

    def test_episode_and_single_pending_observation_contract(self) -> None:
        policy = OnlineSpikingPolicy(seed=7)
        features = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        with self.assertRaises(RuntimeError):
            policy.choose(features, ("add", "sub", "mul"))
        with self.assertRaises(RuntimeError):
            policy.observe(1.0)
        policy.start_episode("episode-a")
        action = policy.choose(features, ("add", "sub", "mul"))
        self.assertIn(action, ("add", "sub", "mul"))
        with self.assertRaises(RuntimeError):
            policy.choose(features, ("add", "sub", "mul"))
        with self.assertRaises(RuntimeError):
            policy.end_episode()
        policy.observe(1.0)
        self.assertTrue(
            all(value == 0.0 for row in policy._eligibility_weights
                for value in row)
        )
        self.assertTrue(all(value == 0.0
                            for value in policy._eligibility_bias))
        policy.end_episode()
        with self.assertRaises(RuntimeError):
            policy.choose(features, ("add", "sub", "mul"))

    def test_action_mask_and_input_contract(self) -> None:
        policy = OnlineSpikingPolicy(seed=3)
        policy.start_episode("masked")
        self.assertEqual(
            policy.choose((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), ("sub",)),
            "sub",
        )
        self.assertEqual(policy.metrics()["readout_mac"], 3 * 16)
        policy.observe(0.0)
        with self.assertRaises(ValueError):
            policy.choose((1.0, 0.0, 0.0, 0.0, 1.0), ("add",))
        with self.assertRaises(ValueError):
            policy.choose((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), ("div",))
        with self.assertRaises(ValueError):
            policy.choose((1.0, 0.0, 1.0, 1.0, 1.0, 0.0), ("add",))
        with self.assertRaises(ValueError):
            policy.choose((1.0, 0.0, 0.0, 0.0, 1.1, 0.0), ("add",))
        with self.assertRaises(ValueError):
            policy.choose((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), ())
        policy.end_episode()

    def test_observed_reward_changes_only_adaptive_readout(self) -> None:
        learning = OnlineSpikingPolicy(seed=11, learning=True)
        frozen = OnlineSpikingPolicy(seed=11, learning=False)
        initial_learned = learning.weight_hash()
        initial_frozen = frozen.weight_hash()
        features = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        for policy in (learning, frozen):
            policy.start_episode("rewarded")
            action = policy.choose(features, ("add", "sub", "mul"))
            self.assertIn(action, ("add", "sub", "mul"))
            policy.observe(1.0)
            policy.end_episode()
        self.assertNotEqual(learning.weight_hash(), initial_learned)
        self.assertEqual(frozen.weight_hash(), initial_frozen)
        learned_metrics = learning.metrics()
        frozen_metrics = frozen.metrics()
        self.assertEqual(learned_metrics["updates"], 1)
        self.assertGreater(learned_metrics["update_synapses"], 0)
        self.assertEqual(learned_metrics["update_synapse_visits"], 3 * 17)
        self.assertEqual(frozen_metrics["updates"], 0)
        self.assertEqual(frozen_metrics["reward_observations"], 1)

    def test_reward_clipping_and_invalid_reward(self) -> None:
        policy = OnlineSpikingPolicy(seed=2)
        policy.start_episode("rewards")
        features = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        policy.choose(features, ("add", "sub", "mul"))
        with self.assertRaises(ValueError):
            policy.observe(float("nan"))
        self.assertEqual(policy.observe(8.0), 1.0)
        policy.choose(features, ("add", "sub", "mul"))
        self.assertEqual(policy.observe(-8.0), -1.0 - 0.05)
        policy.end_episode()

    def test_good_and_bad_rewards_credit_sampled_action(self) -> None:
        positive = OnlineSpikingPolicy(seed=23)
        negative = OnlineSpikingPolicy(seed=23)
        features = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        for policy in (positive, negative):
            policy.start_episode("credit")
        good_action = positive.choose(features, ("add", "sub", "mul"))
        bad_action = negative.choose(features, ("add", "sub", "mul"))
        self.assertEqual(good_action, bad_action)
        positive.observe(1.0)
        negative.observe(-1.0)
        positive.end_episode()
        negative.end_episode()
        good_bias = positive.snapshot()["bias"]
        bad_bias = negative.snapshot()["bias"]
        chosen = ("add", "sub", "mul").index(good_action)
        self.assertAlmostEqual(good_bias[chosen], 0.05 * 2.0 / 3.0)
        self.assertAlmostEqual(bad_bias[chosen], -0.05 * 2.0 / 3.0)
        for index in range(3):
            if index != chosen:
                self.assertAlmostEqual(good_bias[index], -0.05 / 3.0)
                self.assertAlmostEqual(bad_bias[index], 0.05 / 3.0)

    def test_each_observation_consumes_only_its_decision_trace(self) -> None:
        with self.assertRaises(ValueError):
            OnlineSpikingPolicy(eligibility_decay=0.8)
        policy = OnlineSpikingPolicy(seed=31)
        features = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        policy.start_episode("one-decision-credit")
        policy.choose(features, ("add", "sub", "mul"))
        policy.observe(-1.0)
        after_first = policy._bias[:]
        policy.choose(features, ("add", "sub", "mul"))
        policy.observe(1.0)
        after_second = policy._bias[:]
        self.assertNotEqual(after_first, after_second)
        self.assertTrue(
            all(value == 0.0 for row in policy._eligibility_weights
                for value in row)
        )
        self.assertTrue(all(value == 0.0
                            for value in policy._eligibility_bias))
        policy.end_episode()

    def test_snapshot_restore_preserves_learning_and_sampling(self) -> None:
        first = OnlineSpikingPolicy(seed=19)
        features = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        first.start_episode("initial")
        first.choose(features, ("add", "sub", "mul"))
        first.observe(0.75)
        first.end_episode()
        snapshot = first.snapshot()

        second = OnlineSpikingPolicy(seed=19)
        second.restore(snapshot)
        first.start_episode("next-a")
        second.start_episode("next-b")
        action_a = first.choose(features, ("add", "sub", "mul"))
        action_b = second.choose(features, ("add", "sub", "mul"))
        self.assertEqual(action_a, action_b)
        first.observe(-0.25)
        second.observe(-0.25)
        first.end_episode()
        second.end_episode()
        self.assertEqual(first.snapshot(), second.snapshot())

    def test_episode_boundary_clears_membrane_and_eligibility(self) -> None:
        first = OnlineSpikingPolicy(seed=29)
        features = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        first.start_episode("before")
        first.choose(features, ("add", "sub", "mul"))
        first.observe(0.6)
        first.end_episode()
        boundary = first.snapshot()
        second = OnlineSpikingPolicy(seed=29)
        second.restore(boundary)
        first.start_episode("after-first")
        second.start_episode("after-restored")
        first.choose(features, ("add", "sub", "mul"))
        second.choose(features, ("add", "sub", "mul"))
        first.observe(-0.2)
        second.observe(-0.2)
        first.end_episode()
        second.end_episode()
        self.assertEqual(first.snapshot(), second.snapshot())

    def test_snapshot_rejects_bad_version_config_shape_and_hash(self) -> None:
        policy = OnlineSpikingPolicy(seed=5)
        snapshot = policy.snapshot()
        for mutation in (
            lambda data: data.update(version=99),
            lambda data: data["config"].update(steps=99),
            lambda data: data.update(weights=[[float("nan")]]),
            lambda data: data.update(weight_hash="0" * 64),
        ):
            bad = copy.deepcopy(snapshot)
            mutation(bad)
            with self.assertRaises(ValueError):
                policy.restore(bad)
        before = policy.snapshot()
        corrupt = copy.deepcopy(snapshot)
        corrupt["weights"][0][0] = 0.5
        corrupt["weight_hash"] = "0" * 64
        with self.assertRaises(ValueError):
            policy.restore(corrupt)
        self.assertEqual(policy.snapshot(), before)

    def test_reset_learning_restores_initial_readout_and_metrics(self) -> None:
        policy = OnlineSpikingPolicy(seed=13)
        initial_hash = policy.weight_hash()
        policy.start_episode("train")
        policy.choose(
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            ("add", "sub", "mul"),
        )
        policy.observe(1.0)
        policy.end_episode()
        self.assertNotEqual(policy.weight_hash(), initial_hash)
        policy.reset_learning()
        self.assertEqual(policy.weight_hash(), initial_hash)
        metrics = policy.metrics()
        self.assertEqual(metrics["updates"], 0)
        self.assertEqual(metrics["reward_observations"], 0)

    def test_continuous_encoder_control_shares_plastic_readout_contract(
        self,
    ) -> None:
        policy = OnlineSpikingPolicy(seed=17, spiking=False)
        policy.start_episode("continuous")
        policy.choose(
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            ("add", "sub", "mul"),
        )
        policy.observe(1.0)
        policy.end_episode()
        metrics = policy.metrics()
        self.assertEqual(metrics["spikes"], 0)
        self.assertEqual(metrics["neuron_ticks"], 6 * 16)
        self.assertEqual(metrics["readout_mac"], 3 * 16)
        self.assertEqual(metrics["encoder_mode"], "continuous_leaky")
        self.assertEqual(metrics["updates"], 1)
        self.assertGreater(metrics["update_synapses"], 0)


if __name__ == "__main__":
    unittest.main()
