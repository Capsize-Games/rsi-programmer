"""Repeatability and training semantics for frozen ranking controls."""
from __future__ import annotations

import math
import random
import unittest

import torch

from rsi_programmer.benchmark_mlp import MLPControl
from rsi_programmer.benchmark_policy_heuristic import (
    ClassicalHeuristicPolicy,
)
from rsi_programmer.benchmark_policy_random import SeededRandomPolicy
from rsi_programmer.benchmark_recurrent import RecurrentControl


def _features(hit: float, inside: float, left_size: float = 0.0,
              right_size: float = 0.0) -> list[float]:
    """Build a valid row in the current 12-field encoding."""
    return [1.0, 0.0, 0.0, left_size, right_size, 1.0,
            hit, inside, hit, inside, 0.25, 0.75]


class BenchmarkPolicyTests(unittest.TestCase):
    """Check fixed-control behavior, fit updates and fresh scoring state."""

    def test_heuristic_prefers_operand_agreement_and_simplicity(self) -> None:
        policy = ClassicalHeuristicPolicy()
        feats = [_features(0.0, 0.0), _features(1.0, 1.0, 1.0, 1.0)]
        self.assertEqual(policy.rank(feats), [1, 0])
        before = policy.weights()
        before_fingerprint = policy.fingerprint()
        self.assertEqual(policy.metadata()["trainable_parameters"], 0)
        self.assertEqual(policy.fit([], epochs=1), 0.0)
        self.assertEqual(policy.weights(), before)
        self.assertEqual(policy.fingerprint(), before_fingerprint)

    def test_seeded_random_repeats_without_global_rng_changes(self) -> None:
        feats = [_features(float(i) / 9.0, 0.5) for i in range(10)]
        random.seed(57)
        before = random.getstate()
        first = SeededRandomPolicy(17)
        second = SeededRandomPolicy(17)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(first.rank(feats), second.rank(feats))
        before = first.weights()
        fingerprint = first.fingerprint()
        self.assertEqual(first.fit([], epochs=1), 0.0)
        self.assertEqual(first.weights(), before)
        self.assertEqual(first.fingerprint(), fingerprint)
        self.assertEqual(first.fingerprint(), second.fingerprint())
        other = SeededRandomPolicy(18)
        self.assertNotEqual(first.rank(feats), other.rank(feats))

    def test_neural_initialization_preserves_torch_rng(self) -> None:
        torch.manual_seed(91)
        before = torch.random.get_rng_state().clone()
        MLPControl(seed=4)
        RecurrentControl(seed=4)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_neural_learning_rate_is_explicit_in_optimizer_and_metadata(self):
        for policy_type in (MLPControl, RecurrentControl):
            with self.subTest(policy=policy_type.__name__):
                policy = policy_type(seed=2, lr=0.01)
                self.assertEqual(policy._opt.param_groups[0]["lr"], 0.01)
                self.assertEqual(policy.metadata()["learning_rate"], 0.01)
                self.assertEqual(policy.metadata()["fit_mode"],
                                 "adam_margin_lr_0.01")

    def test_neural_fit_changes_weights_from_a_repeatable_seed(self) -> None:
        groups = [[(_features(1.0, 1.0), 1),
                   (_features(0.0, 0.0), 0)]]
        for policy_type in (MLPControl, RecurrentControl):
            with self.subTest(policy=policy_type.__name__):
                first = policy_type(seed=8)
                second = policy_type(seed=8)
                if policy_type is MLPControl:
                    self.assertEqual(
                        first.metadata()["trainable_parameters"], 225,
                    )
                initial = first.weights()
                self.assertEqual(first.fingerprint(), second.fingerprint())
                loss = first.fit(groups, epochs=2)
                second_loss = second.fit(groups, epochs=2)
                self.assertTrue(math.isfinite(loss))
                self.assertAlmostEqual(loss, second_loss)
                self.assertNotEqual(first.fingerprint(),
                                    state_fingerprint_for_test(initial,
                                                               first))
                self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_neural_scores_have_fresh_state_and_batch_order_independence(self):
        rows = [_features(0.1, 0.2), _features(0.9, 0.8),
                _features(0.4, 0.7)]
        for policy_type in (MLPControl, RecurrentControl):
            with self.subTest(policy=policy_type.__name__):
                policy = policy_type(seed=3)
                single = policy._forward(torch.tensor([rows[1]])).item()
                batch = policy._forward(torch.tensor(rows)).tolist()[1]
                reversed_batch = policy._forward(
                    torch.tensor(list(reversed(rows)))
                ).tolist()[1]
                self.assertAlmostEqual(single, batch, places=6)
                self.assertAlmostEqual(single, reversed_batch, places=6)

    def test_ineligible_training_data_is_not_reported_as_a_fit(self) -> None:
        group = [[(_features(1.0, 1.0), 1)]]
        for policy_type in (MLPControl, RecurrentControl):
            with self.subTest(policy=policy_type.__name__):
                with self.assertRaisesRegex(ValueError, "both labels"):
                    policy_type(seed=1).fit(group, epochs=1)


def state_fingerprint_for_test(weights: dict[str, torch.Tensor],
                               policy: MLPControl | RecurrentControl) -> str:
    """Fingerprint a pre-fit state using the policy's metadata contract."""
    from rsi_programmer.benchmark_policy import state_fingerprint

    return state_fingerprint(weights, policy.metadata())


if __name__ == "__main__":
    unittest.main()
