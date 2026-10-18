#!/usr/bin/env python3
"""Tests for the shared-encoder bit-slice team (lattice v32)."""
from __future__ import annotations

import unittest

import torch

from rsi_programmer.lattice import BitLattice
from rsi_programmer.modules import ModularPolicy
from rsi_programmer.policy import FEATURE_ENCODING, encode, layer_triples
from rsi_programmer.probe import signature, terminals
from rsi_programmer.traces import TraceRecorder
from rsi_programmer.words import unpack

FIXTURE = "fixtures/rsi_shadow_tasks.json"


def _shadow_feats() -> list[list[float]]:
    """Real size-1 search features for every shadow task."""
    feats: list[list[float]] = []
    fam = TraceRecorder(FIXTURE).load_family()["tasks"]
    for task in fam:
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        levels = {0: list(terminals())}
        triples = layer_triples(levels, 1)
        for i, (op, left, right) in enumerate(triples):
            feats.append(encode(op, signature(left, rows),
                                signature(right, rows), target,
                                left.size, right.size, 1, bound,
                                i + 1, len(triples)))
    return feats


class LatticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """One modular donor and one lattice donee, seed 0."""
        cls.donor = ModularPolicy(seed=0)
        cls.lat = BitLattice(seed=0)
        cls.loaded = cls.lat.load_encoder(cls.donor.weights())
        cls.feats = _shadow_feats()

    def test_encoder_transfer_count_and_bytes(self) -> None:
        """Fourteen stage keys land identical in the donee."""
        self.assertEqual(self.loaded, 14)
        mine = self.lat.weights()
        for key, value in self.donor.weights().items():
            if key.split(".")[0] in ("enc_fc", "enc", "st_fc",
                                     "st", "fb_fc"):
                self.assertTrue(torch.allclose(mine[key],
                                               value))

    def test_struct_parity_on_shadow_family(self) -> None:
        """Shared struct gate matches the donor on shadow."""
        mine = self.lat.head_rates(self.feats)["st"]
        hers = self.donor.head_rates(self.feats)["st"]
        self.assertEqual(len(mine), len(hers))
        worst = max(abs(a - b) for m, h in zip(mine, hers)
                    for a, b in zip(m, h))
        print("shadow rows=%d st-worst-diff=%.2e"
              % (len(mine), worst))
        self.assertLess(worst, 1e-6)

    def test_independent_lattice_drives_start_fresh(self) -> None:
        """A prior same-sized batch cannot change lattice output."""
        first = self.lat.head_rates(self.feats)
        self.lat.head_rates([[0.0] * 12 for _ in self.feats])
        second = self.lat.head_rates(self.feats)
        self.assertEqual(first, second)

    def test_lesion_flips_bit_two_only(self) -> None:
        """Silencing head 2 flips bit 2, nothing else."""
        drive = [[0.0] * 12]
        state = self.lat.weights()
        state["b2_fc.weight"] = torch.ones_like(
            state["b2_fc.weight"]) * 2.0
        state["b2_fc.bias"] = torch.ones_like(
            state["b2_fc.bias"]) * 5.0
        self.lat._net.load_state_dict(state)
        self.lat.reset()
        before = unpack(self.lat.bits(drive)[0])
        self.assertEqual(before[2], 1)
        state["b2_fc.weight"] = torch.zeros_like(
            state["b2_fc.weight"])
        state["b2_fc.bias"] = torch.zeros_like(
            state["b2_fc.bias"])
        self.lat._net.load_state_dict(state)
        self.lat.reset()
        after = unpack(self.lat.bits(drive)[0])
        self.assertEqual(after[2], 0)
        self.assertEqual(before[:2] + before[3:],
                         after[:2] + after[3:])

    def test_struct_gate_steers_commit(self) -> None:
        """Biased struct weights force commit or reject."""
        drive = [[0.5] * 12]
        state = self.lat.weights()
        keep = {k: v.clone() for k, v in state.items()}
        state["st_fc.weight"] = torch.zeros_like(
            state["st_fc.weight"])
        state["st_fc.bias"] = torch.tensor([-5.0, 5.0])
        self.lat._net.load_state_dict(state)
        self.lat.reset()
        self.assertEqual(self.lat.struct(drive), [False])
        state["st_fc.bias"] = torch.tensor([5.0, -5.0])
        self.lat._net.load_state_dict(state)
        self.lat.reset()
        self.assertEqual(self.lat.struct(drive), [True])
        keep_state = {k: v for k, v in keep.items()}
        self.lat._net.load_state_dict(keep_state)

    def test_snapshot_restore_round_trips(self) -> None:
        """Snapshots reload weights and reproduce words."""
        drive = [[0.25] * 12]
        self.lat.reset()
        before = self.lat.bits(drive)
        snap = self.lat.snapshot()
        self.assertEqual(snap["feature_encoding"], FEATURE_ENCODING)
        twin = BitLattice(seed=9)
        twin.restore(snap)
        for key, value in self.lat.weights().items():
            self.assertTrue(torch.allclose(twin.weights()[key],
                                           value))
        twin.reset()
        self.assertEqual(twin.bits(drive), before)

    def test_encoder_freeze_api(self) -> None:
        """Freeze locks shared stages, never the heads."""
        self.lat.set_encoder_frozen(True)
        for name, param in self.lat._net.named_parameters():
            locked = name.split(".")[0] in ("enc_fc", "enc",
                                            "st_fc", "st",
                                            "fb_fc")
            self.assertEqual(param.requires_grad, not locked,
                             name)
        self.lat.set_encoder_frozen(False)
        for param in self.lat._net.parameters():
            self.assertTrue(param.requires_grad)


if __name__ == "__main__":
    unittest.main()
