"""Decision, learning and state contracts for the construction policies."""
from __future__ import annotations

import copy
import unittest

import torch

from rsi_programmer.construction_beam import beam_construct
from rsi_programmer.construction_features import encode_states
from rsi_programmer.construction_greedy import greedy_construct
from rsi_programmer.construction_policy import ConstructionPolicy
from rsi_programmer.construction_state import ConstructionState
from rsi_programmer.construction_training import fit_constructor

ROWS = ((-2, 1, -1), (0, 3, 3), (2, -1, 1))
TASK = {
    "split": "train", "observed": [list(row) for row in ROWS],
    "max_tree_ops": 4,
    "generator": {"program": {
        "nodes": [{"op": "add", "type": "Int", "args": ["x", "y"]}],
    }},
}


class ConstructionPolicyTests(unittest.TestCase):
    """Check real neural updates and row/reference decision semantics."""

    def test_seed_initialization_preserves_caller_rng_and_capacity(
        self,
    ) -> None:
        before = torch.get_rng_state().clone()
        ann = ConstructionPolicy(seed=2, hidden=8)
        snn = ConstructionPolicy(seed=2, backend="snn", hidden=8)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(ann.metadata()["parameter_count"],
                         snn.metadata()["parameter_count"])
        self.assertEqual(ann.fingerprint(),
                         ConstructionPolicy(seed=2, hidden=8).fingerprint())

    def test_observed_row_order_is_invariant_and_padding_is_masked(
        self,
    ) -> None:
        state = ConstructionState(ROWS)
        reordered = ConstructionState(tuple(reversed(ROWS)))
        grown = state.apply({"op": "add", "left": 0, "right": 1})
        model = ConstructionPolicy(hidden=8)
        with torch.inference_mode():
            first, _ = model.encode(encode_states([state]))
            other, _ = model.encode(encode_states([reordered]))
            padded, references = model.encode(encode_states([state, grown]))
        torch.testing.assert_close(first, other)
        torch.testing.assert_close(first[0], padded[0])
        with torch.inference_mode():
            mask = encode_states([state, grown])[1]
            logits = model.left_logits(padded, references, mask,
                                       torch.tensor([0, 0]))
        self.assertTrue(torch.isneginf(logits[0, 2]))

    def test_ann_and_snn_training_change_weights_without_reading_held(
        self,
    ) -> None:
        poisoned = copy.deepcopy(TASK)
        poisoned["held"] = object()
        for backend in ("ann", "snn"):
            model = ConstructionPolicy(backend=backend, hidden=8)
            before = model.row_input.weight.detach().clone()
            result = fit_constructor(model, [poisoned], epochs=2,
                                     batch_size=2)
            self.assertNotEqual(result["initial_hash"], result["final_hash"])
            self.assertFalse(torch.equal(before, model.row_input.weight))
            self.assertEqual(result["updates"], 2)
            if backend == "snn":
                self.assertGreater(result["neural_work"]["spikes"], 0)

    def test_rejected_development_labels_cannot_partially_update(self) -> None:
        model = ConstructionPolicy(hidden=8)
        before = model.fingerprint()
        wrong = copy.deepcopy(TASK)
        wrong["split"] = "dev"
        with self.assertRaises(ValueError):
            fit_constructor(model, [TASK, wrong], epochs=1)
        self.assertEqual(model.fingerprint(), before)

    def test_actual_neural_stop_and_evaluation_leave_weights_unchanged(
        self,
    ) -> None:
        model = ConstructionPolicy(hidden=8)
        with torch.no_grad():
            model.operation.weight.zero_()
            model.operation.bias.fill_(-100)
            model.operation.bias[3] = 100
        before = model.fingerprint()
        first = greedy_construct(model, ROWS)
        second = greedy_construct(model, ROWS)
        self.assertEqual(first, second)
        self.assertTrue(first["stopped"])
        self.assertFalse(first["observed_match"])
        self.assertEqual(first["actions"], [{"op": "stop"}])
        self.assertEqual(model.fingerprint(), before)

    def test_beam_execution_cap_and_random_control_are_reproducible(
        self,
    ) -> None:
        model = ConstructionPolicy(hidden=8)
        before = model.fingerprint()
        out = beam_construct(model, ROWS, budget=3)
        self.assertLessEqual(out["executions"], 3)
        self.assertEqual(before, model.fingerprint())
        self.assertEqual(beam_construct(None, ROWS, seed=4),
                         beam_construct(None, ROWS, seed=4))
