"""Visible-prefix execution and invalid-state information boundaries."""
from __future__ import annotations

import unittest

import torch

from rsi_programmer.array_partial import complete, outputs
from rsi_programmer.array_policy import ArrayPolicy


class ArrayExecutionFeatureTests(unittest.TestCase):
    def test_completion_uses_fixed_syntax_and_actual_inputs(self) -> None:
        prefix = tuple("for append sub zero".split())
        self.assertEqual(complete(prefix), tuple((
            "for append sub zero value end return output stop").split()))
        self.assertEqual(outputs(prefix, [[-1, 2], []]),
                         ([[1, -2], []], [1.0, 1.0]))

    def test_invalid_execution_is_separate_from_empty_output(self) -> None:
        prefix = tuple("for append rem value zero".split())
        self.assertEqual(outputs(prefix, [[1]]), ([[]], [0.0]))
        valid = tuple("for if gt value zero append value".split())
        self.assertEqual(outputs(valid, [[-1]]), ([[]], [1.0]))

    def test_prefix_keeps_observed_examples_fixed(self) -> None:
        model = ArrayPolicy(execution_context=True)
        inputs, targets = [[-1, 2]], [[1, -2]]
        good = model.observed_features(
            inputs, targets, tuple("for append sub zero value".split()))
        bad = model.observed_features(
            inputs, targets, tuple("for append rem value zero".split()))
        self.assertEqual(tuple(good.shape), (1, 95))
        self.assertTrue(torch.equal(good[:, :47], bad[:, :47]))
        self.assertEqual(float(good[0, -1]), 1.0)
        self.assertEqual(float(bad[0, -1]), 0.0)
