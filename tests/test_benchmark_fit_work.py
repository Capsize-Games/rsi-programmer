"""Checks for benchmark model-fit work instrumentation."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from rsi_programmer.benchmark_fit_work import fit_with_work
from rsi_programmer.benchmark_mlp import MLPControl
from rsi_programmer.benchmark_policy import Group


def _group() -> Group:
    """Return one complete group with two positives and three negatives."""
    positive = [([float(index)] * 12, 1) for index in (1, 2)]
    negative = [([float(index)] * 12, 0) for index in (3, 4, 5)]
    return positive + negative


class BenchmarkFitWorkTests(unittest.TestCase):
    def test_counts_actual_updates_batches_rows_and_margin_pairs(self) -> None:
        model = MLPControl(seed=13)
        loss, work = fit_with_work(model, [_group()], 3, model._steps)

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(work["requested_epochs"], 3)
        self.assertEqual(work["eligible_groups"], 1)
        self.assertEqual(work["positive_features"], 2)
        self.assertEqual(work["negative_features"], 3)
        self.assertEqual(work["margin_pairs_per_epoch"], 6)
        self.assertEqual(work["optimizer_updates"], 3)
        self.assertEqual(work["forward_batches"], 6)
        self.assertEqual(work["scored_feature_rows"], 15)
        self.assertEqual(work["simulation_row_steps"], 15)

    def test_instrumentation_preserves_loss_and_fit_weights(self) -> None:
        instrumented = MLPControl(seed=27)
        plain = MLPControl(seed=27)
        wrapped_loss, _work = fit_with_work(
            instrumented, [_group()], 2, instrumented._steps)
        plain_loss = plain.fit([_group()], 2)

        self.assertEqual(wrapped_loss, plain_loss)
        for name, value in instrumented.weights().items():
            self.assertTrue(torch.equal(value, plain.weights()[name]))
        self.assertNotIn("_forward", instrumented.__dict__)
        self.assertNotIn("step", instrumented._opt.__dict__)

    def test_wrappers_are_restored_when_fit_raises(self) -> None:
        model = MLPControl(seed=31)

        def fail_fit(groups: list[Group], epochs: int) -> float:
            del groups, epochs
            model._forward(torch.ones((2, 12)))
            model._opt.step()
            raise RuntimeError("fit failed")

        with patch.object(model, "fit", side_effect=fail_fit):
            with self.assertRaisesRegex(RuntimeError, "fit failed"):
                fit_with_work(model, [_group()], 1, model._steps)

        self.assertNotIn("_forward", model.__dict__)
        self.assertNotIn("step", model._opt.__dict__)
        self.assertTrue(torch.isfinite(torch.tensor(model.fit([_group()], 1))))


if __name__ == "__main__":
    unittest.main()
