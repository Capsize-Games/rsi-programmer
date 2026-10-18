"""Fast regressions for lattice review findings #52, #56-58, and #65."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from rsi_programmer.lattice import BitLattice
from rsi_programmer.lprecision import split_groups, sweep_thresholds
from rsi_programmer.ltrain import (_calibrate, _margins, _report,
                                   _pack_objective, train_lattice)
from rsi_programmer.mgroups import module_report
from rsi_programmer.modules import ModularPolicy


class ReviewLatticeTests(unittest.TestCase):
    def test_modular_independent_drives_start_fresh(self) -> None:
        policy = ModularPolicy(seed=4)
        feats = [[float((i + j) % 3) / 2.0 for j in range(12)]
                 for i in range(5)]
        first = policy.head_rates(feats)
        policy.head_rates([[0.0] * 12 for _ in feats])
        second = policy.head_rates(feats)
        self.assertEqual(first, second)

    def test_modular_continuation_is_explicit(self) -> None:
        policy = ModularPolicy(seed=4)
        feats = [[float((i + j) % 3) / 2.0 for j in range(12)]
                 for i in range(5)]
        fresh = ModularPolicy(seed=4)
        expected = fresh.head_rates(feats)
        policy.head_rates(feats)
        continued = policy.head_rates(feats, continue_state=True)
        self.assertEqual(expected, policy.head_rates(feats))
        self.assertNotEqual(expected, continued)

    def test_constant_bit_target_uses_squared_error(self) -> None:
        out = torch.tensor([0.2, 0.8], requires_grad=True)
        loss = _margins(out, torch.zeros(2))
        self.assertAlmostEqual(loss.item(), 0.34, places=6)

    def test_pack_loss_is_mean_plus_max_of_head_losses(self) -> None:
        terms = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertAlmostEqual(_pack_objective(terms, 0.2).item(), 4.7,
                               places=6)

    def test_split_keeps_validation_groups_untouched(self) -> None:
        groups = [[([float(i)], i)] for i in range(20)]
        train, calibration, evaluation = split_groups(groups)
        self.assertTrue(train and calibration and evaluation)
        self.assertFalse(set(map(id, train)) & set(map(id, calibration)))
        self.assertFalse(set(map(id, train)) & set(map(id, evaluation)))
        self.assertFalse(set(map(id, calibration)) &
                         set(map(id, evaluation)))
        self.assertEqual(len(train) + len(calibration) + len(evaluation),
                         len(groups))

    def test_small_split_has_no_phantom_evaluation_group(self) -> None:
        groups = [[([0.0], 1)], [([1.0], 0)]]
        train, calibration, evaluation = split_groups(groups)
        self.assertEqual(train, groups)
        self.assertEqual(calibration, [])
        self.assertEqual(evaluation, [])
        report = _report(BitLattice(seed=2), torch.empty((0, 13)),
                         torch.empty((0, 4)), "synthetic", 1, 0,
                         torch.empty((0,)))
        self.assertIsNone(report["word_bit_exact_accuracy"])
        self.assertIsNone(report["joint_word_emit_accuracy"])

    def test_joint_word_and_emit_metric_matches_membership(self) -> None:
        lat = BitLattice(seed=2)
        outputs = {"b%d" % i: torch.tensor([[1.0], [0.0]])
                   for i in range(4)}
        outputs["st"] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        with patch.object(lat, "_drive", return_value=outputs):
            report = _report(
                lat, torch.zeros((2, 13)),
                torch.tensor([[1.0] * 4, [0.0] * 4]),
                "synthetic", 1, 2, torch.tensor([1.0, 0.0]))
        self.assertEqual(report["emit_membership_accuracy"], 1.0)
        self.assertEqual(report["joint_word_emit_accuracy"], 1.0)

    def test_calibration_sweep_reports_joint_word_emit_accuracy(self) -> None:
        lat = BitLattice(seed=2)
        outputs = {"b%d" % i: torch.tensor([[1.0], [0.0]])
                   for i in range(4)}
        outputs["st"] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        pack = (torch.zeros((2, 13)),
                torch.tensor([[1.0] * 4, [0.0] * 4]),
                torch.tensor([1.0, 0.0]))
        with patch.object(lat, "_drive", return_value=outputs), \
             patch("rsi_programmer.lprecision._word_exact",
                   return_value=0.0), \
             patch("rsi_programmer.lprecision._struct_f1",
                   return_value=0.0):
            report = sweep_thresholds(lat, pack)
        self.assertEqual(
            report["calibration_joint_word_emit_accuracy"], 1.0)

    def test_missing_calibration_reports_joint_metric_unavailable(
            self) -> None:
        lat = BitLattice(seed=2)
        row = [1.0, 0.0, 0.0] + [0.0] * 9
        group = [(row, 0), (row, 1)]
        rates = {"b%d" % i: 0.5 for i in range(4)}
        rates.update({"st_emit": 0.5, "st_stop": 0.5})
        ledger = type("Ledger", (), {
            "record_intent": lambda *args: None,
            "close": lambda *args: None})()
        with patch("rsi_programmer.ltrain.contrastive_groups",
                   return_value=[group, group]), \
             patch("rsi_programmer.ltrain.fit_packs"), \
             patch("rsi_programmer.ltrain._calibrate",
                   return_value={"rates": rates, "converged": True,
                                 "cycles": 1, "target": 0.5,
                                 "tolerance": 0.08,
                                 "max_deviation": 0.0}), \
             patch("rsi_programmer.ltrain._measure_rates",
                   return_value=rates), \
             patch("rsi_programmer.ltrain.mission_ledger",
                   return_value=ledger):
            report = train_lattice(lat, "synthetic", "unused", epochs=0)
        self.assertFalse(report["calibration_available"])
        self.assertIsNone(
            report["calibration_joint_word_emit_accuracy"])

    def test_module_opcode_argmax_is_labeled_as_diagnostic(self) -> None:
        class Policy:
            def head_rates(self, feats, gap=None):
                return {"op": [[0.9, 0.1, 0.0]], "lo": [[0.7]]}

        report = module_report(Policy(), [{
            "feats": [[0.0] * 12], "gap": 1.0,
            "solops": [0], "ops": [0], "lefts": [[1.0]],
            "want": [[1.0]]}])
        self.assertEqual(report["op_argmax_role"],
                         "not_directly_supervised_diagnostic")
        self.assertFalse(report["op_argmax_directly_supervised"])

    def test_threshold_sweep_keeps_current_values_on_ties(self) -> None:
        lat = BitLattice(seed=2)
        lat.thresholds.update({"b0": 0.55, "b1": 0.5,
                               "b2": 0.5, "b3": 0.5,
                               "st": 0.1})
        pack = (torch.zeros((2, 13)), torch.zeros((2, 4)),
                torch.zeros((2,)))
        with patch("rsi_programmer.lprecision._word_exact",
                   return_value=0.0), \
             patch("rsi_programmer.lprecision._struct_f1",
                   return_value=0.0):
            report = sweep_thresholds(lat, pack)
        self.assertEqual(lat.thresholds["b0"], 0.55)
        self.assertEqual(lat.thresholds["st"], 0.1)
        self.assertEqual(report["bit_thresholds"][0], 0.55)
        self.assertEqual(report["st_margin"], 0.1)

    def test_lattice_restore_keeps_recipient_adam_options_and_freezes(
            self) -> None:
        lat = BitLattice(seed=2)
        lat.set_encoder_frozen(True)
        lat._opt.param_groups[0].update(
            {"lr": 0.07, "weight_decay": 0.1, "betas": (0.8, 0.9)})
        restored = BitLattice(seed=8)
        restored._opt.param_groups[0]["lr"] = 0.03
        restored.restore(lat.snapshot())
        self.assertEqual(restored._opt.param_groups[0]["lr"], 0.03)
        self.assertEqual(restored._opt.param_groups[0]["weight_decay"], 0.0)
        self.assertEqual(restored._opt.param_groups[0]["betas"],
                         (0.9, 0.999))
        self.assertEqual(
            {name for name, param in restored._net.named_parameters()
             if not param.requires_grad},
            {name for name, param in lat._net.named_parameters()
             if not param.requires_grad})

    def test_lattice_snapshot_optimizer_is_explicit_opt_in(self) -> None:
        donor = BitLattice(seed=2)
        donor._opt.param_groups[0].update(
            {"lr": 0.07, "weight_decay": 0.1, "betas": (0.8, 0.9)})
        restored = BitLattice(seed=8)
        restored.restore(donor.snapshot(), optimizer_from_snapshot=True)
        group = restored._opt.param_groups[0]
        self.assertEqual(group["lr"], 0.07)
        self.assertEqual(group["weight_decay"], 0.1)
        self.assertEqual(group["betas"], (0.8, 0.9))

    def test_modular_restore_uses_recipient_or_explicit_donor_options(
            self) -> None:
        donor = ModularPolicy(seed=2)
        donor._opt.param_groups[0].update(
            {"lr": 0.07, "weight_decay": 0.1, "betas": (0.8, 0.9)})
        snapshot = donor.snapshot()

        recipient = ModularPolicy(seed=8, lr=0.03)
        recipient.restore(snapshot)
        group = recipient._opt.param_groups[0]
        self.assertEqual(group["lr"], 0.03)
        self.assertEqual(group["weight_decay"], 0.0)
        self.assertEqual(group["betas"], (0.9, 0.999))

        donor_options = ModularPolicy(seed=9)
        donor_options.restore(snapshot, optimizer_from_snapshot=True)
        group = donor_options._opt.param_groups[0]
        self.assertEqual(group["lr"], 0.07)
        self.assertEqual(group["weight_decay"], 0.1)
        self.assertEqual(group["betas"], (0.8, 0.9))

    def test_restore_rejects_tagged_incompatible_feature_encoding(
            self) -> None:
        policy = ModularPolicy(seed=2)
        snap = policy.snapshot()
        snap["feature_encoding"] = "old-global-budget-v0"
        with self.assertRaisesRegex(ValueError, "feature encoding"):
            ModularPolicy(seed=3).restore(snap)

        lattice = BitLattice(seed=2)
        snap = lattice.snapshot()
        snap["feature_encoding"] = "old-global-budget-v0"
        with self.assertRaisesRegex(ValueError, "feature encoding"):
            BitLattice(seed=3).restore(snap)

    def test_legacy_untagged_snapshots_remain_loadable(self) -> None:
        modular = ModularPolicy(seed=2).snapshot()
        modular.pop("feature_encoding")
        ModularPolicy(seed=3).restore(modular)

        lattice = BitLattice(seed=2).snapshot()
        lattice.pop("feature_encoding")
        BitLattice(seed=3).restore(lattice)

    def test_calibration_reports_both_struct_channels_afterwards(self) -> None:
        lat = BitLattice(seed=2, steps=4)
        mat = torch.tensor([[0.0] * 13, [1.0] * 13])
        rates = _calibrate(lat, mat)
        self.assertEqual(set(rates["rates"]), {
            "b0", "b1", "b2", "b3", "st_emit", "st_stop"})
        self.assertGreaterEqual(rates["cycles"], 1)
        self.assertLessEqual(rates["cycles"], 3)
        for name, col, key in (("st_emit", 0, "st"),
                               ("st_stop", 1, "st")):
            lat.reset()
            with torch.no_grad():
                out = lat._drive(mat)
            self.assertAlmostEqual(rates["rates"][name],
                                   out[key][:, col].mean().item(),
                                   places=6)


if __name__ == "__main__":
    unittest.main()
