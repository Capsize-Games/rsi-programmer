"""Objective and metric contracts for lattice and modular policies."""
from __future__ import annotations

import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

import torch

from rsi_programmer.ltrain import _margins, _rate_margin
from rsi_programmer.lprecision import _bit_threshold_grid, split_groups
from rsi_programmer.mgroups import (_pairwise_preference,
                                   modular_groups, module_report)
from rsi_programmer.author import solution_sigs_by_size_report
from rsi_programmer.author import contrastive_groups
from rsi_programmer.policy import OP_ORDER
from rsi_programmer.traces import TraceRecorder

ROOT = Path(__file__).resolve().parent.parent
SHADOW = str(ROOT / "fixtures" / "rsi_shadow_tasks.json")


class LatticeMarginTests(unittest.TestCase):
    def test_one_step_margin_has_interior_zero_loss_and_gradient(self) -> None:
        """A one-spike gap is enough; the objective need not saturate."""
        want = torch.tensor([1.0, 0.0])
        attainable_rates = {5: [0.6, 0.2], 8: [0.625, 0.25]}
        for steps in (5, 8):
            with self.subTest(steps=steps):
                out = torch.tensor(attainable_rates[steps],
                                   requires_grad=True)
                loss = _margins(out, want, steps)
                self.assertAlmostEqual(_rate_margin(steps), 1.0 / steps)
                self.assertEqual(loss.item(), 0.0)
                loss.backward()
                self.assertTrue(torch.equal(out.grad, torch.zeros_like(out)))

    def test_mixed_rate_hinge_pushes_reversed_pairs_toward_order(self) -> None:
        out = torch.tensor([0.2, 0.4], requires_grad=True)
        want = torch.tensor([1.0, 0.0])
        loss = _margins(out, want, steps=5)
        self.assertAlmostEqual(loss.item(), 0.4, places=6)
        loss.backward()
        self.assertLess(out.grad[0].item(), 0.0)
        self.assertGreater(out.grad[1].item(), 0.0)

    def test_nonpositive_step_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _margins(torch.tensor([0.2, 0.8]),
                     torch.tensor([0.0, 1.0]), steps=0)

    def test_constant_rate_target_keeps_squared_error(self) -> None:
        out = torch.tensor([0.2, 0.6])
        want = torch.zeros(2)
        self.assertAlmostEqual(_margins(out, want).item(), 0.2)

    def test_discrete_threshold_grid_covers_every_rate_cut(self) -> None:
        for steps in (5, 8):
            with self.subTest(steps=steps):
                grid = _bit_threshold_grid(steps)
                self.assertEqual(grid[0], 0.0)
                self.assertGreater(grid[-1], 1.0)
                patterns = {
                    tuple(rate >= threshold for threshold in grid)
                    for rate in (index / steps
                                 for index in range(steps + 1))}
                self.assertEqual(len(patterns), steps + 1)

    def test_opcode_stratification_is_disjoint_and_reported(self) -> None:
        groups = contrastive_groups(SHADOW)
        report = {}
        train, calibration, evaluation = split_groups(
            groups, report=report)
        expected = list(OP_ORDER)
        self.assertEqual(report["strategy"],
                         "stratified_candidate_opcode")
        self.assertTrue(report["stratified"])
        self.assertTrue(report["coverage_complete"])
        self.assertEqual(report["train_opcodes"], expected)
        self.assertEqual(report["calibration_opcodes"], expected)
        self.assertEqual(report["evaluation_opcodes"], expected)
        self.assertFalse(set(map(id, train)) & set(map(id, calibration)))
        self.assertFalse(set(map(id, train)) & set(map(id, evaluation)))
        self.assertFalse(set(map(id, calibration)) & set(map(id, evaluation)))

    def test_sparse_opcode_strata_do_not_claim_complete_coverage(self) -> None:
        def group(op: int) -> list[tuple[list[float], int]]:
            feat = [float(i == op) for i in range(3)] + [0.0] * 9
            return [(feat, 0)]

        groups = [group(0), group(0), group(1), group(1),
                  group(1), group(2), group(2), group(2)]
        report = {}
        train, calibration, evaluation = split_groups(
            groups, report=report)
        self.assertEqual(report["strategy"],
                         "stratified_candidate_opcode")
        self.assertTrue(report["stratified"])
        self.assertFalse(report["coverage_complete"])
        self.assertEqual(report["train_opcodes"], list(OP_ORDER))
        self.assertEqual(len(train) + len(calibration) + len(evaluation),
                         len(groups))
        self.assertEqual(len({id(group) for group in
                              train + calibration + evaluation}),
                         len(groups))


class ModularObjectiveMetricTests(unittest.TestCase):
    def test_bisect_preference_matches_pair_loop_with_ties(self) -> None:
        """The optimized statistic keeps the exact pairwise definition."""
        for scores in product((0.0, 0.5, 1.0), repeat=4):
            pos, neg = [0, 1], [2, 3]
            outcomes = [1.0 if scores[p] > scores[n] else
                        0.5 if scores[p] == scores[n] else 0.0
                        for p in pos for n in neg]
            expected = sum(outcomes) / len(outcomes)
            self.assertEqual(_pairwise_preference(
                list(scores), pos, neg), expected)
        self.assertIsNone(_pairwise_preference([0.0], [], [0]))

    def test_pairwise_metrics_match_the_five_ranked_row_scores(self) -> None:
        """Objective metrics compare positives/negatives per head."""
        class FixedPolicy:
            def head_rates(self, feats, gap):
                return {
                    "op": [[0.8, 0.2, 0.1], [0.1, 0.7, 0.2],
                           [0.3, 0.6, 0.1], [0.9, 0.4, 0.1]],
                    "lo": [[0.2], [0.3], [0.8], [0.9]],
                    "ro": [[0.9], [0.9], [0.1], [0.1]],
                    "st": [[0.9, 0.1], [0.9, 0.1],
                           [0.1, 0.1], [0.1, 0.1]],
                }

        group = {
            "feats": [[0.0], [1.0], [2.0], [3.0]],
            "gap": 1.0,
            "pos": [0, 1],
            "neg": [2, 3],
            "ops": [0, 1, 0, 1],
            "solops": [0, 1],
            "want": [[1.0], [3.0]],
            "lefts": [[1.0], [3.0], [2.0], [4.0]],
        }
        report = module_report(FixedPolicy(), [group])
        self.assertEqual(report["objective_ranking_layers"], 1)
        self.assertEqual(report["objective_ranking_chance"], 0.5)
        self.assertEqual(report["combined_pairwise_preference_accuracy"], 1.0)
        self.assertEqual(report["op_own_pairwise_preference_accuracy"], 1.0)
        self.assertEqual(
            report["left_score_pairwise_preference_accuracy"], 0.0)
        self.assertEqual(
            report["right_score_pairwise_preference_accuracy"], 1.0)
        self.assertEqual(report["emit_pairwise_preference_accuracy"], 1.0)
        self.assertEqual(report["op_argmax_role"],
                         "not_directly_supervised_diagnostic")
        self.assertFalse(report["op_argmax_directly_supervised"])
        self.assertTrue(report["legacy_metric_names_deprecated"])
        self.assertEqual(report["op_accuracy"],
                         report["op_argmax_accuracy"])

    def test_objective_metrics_survive_empty_choice_diagnostics(self) -> None:
        """Ranking output does not depend on opcode diagnostic labels."""
        class FixedPolicy:
            def head_rates(self, feats, gap):
                return {
                    "op": [[0.8, 0.2, 0.1], [0.1, 0.7, 0.2],
                           [0.3, 0.6, 0.1], [0.9, 0.4, 0.1]],
                    "lo": [[0.2], [0.3], [0.8], [0.9]],
                    "ro": [[0.9], [0.9], [0.1], [0.1]],
                    "st": [[0.9, 0.1], [0.9, 0.1],
                           [0.1, 0.1], [0.1, 0.1]],
                }

        group = {"feats": [[0.0], [1.0], [2.0], [3.0]],
                 "gap": 1.0, "pos": [0, 1], "neg": [2, 3],
                 "ops": [0, 1, 0, 1], "solops": [],
                 "want": [], "lefts": []}
        report = module_report(FixedPolicy(), [group])
        self.assertEqual(report["layers"], 0)
        self.assertEqual(report["objective_ranking_layers"], 1)
        self.assertEqual(report["combined_pairwise_preference_accuracy"], 1.0)
        self.assertNotIn("op_argmax_accuracy", report)

    def test_modular_labels_match_exact_size_reachability(self) -> None:
        reports = []
        groups = modular_groups(SHADOW, supervision_report=reports)
        tasks = {task["id"]: task
                 for task in TraceRecorder(SHADOW).load_family()["tasks"]}
        evidence = {}
        for task_id, task in tasks.items():
            rows = tuple((row[0], row[1]) for row in task["train"])
            target = tuple(row[2] for row in task["train"])
            evidence[task_id], _ = solution_sigs_by_size_report(
                rows, target, int(task.get("max_tree_ops", 4)))
        self.assertTrue(groups)
        for group in groups:
            self.assertEqual(group["label_semantics"],
                             "retained_train_signature_graph_reachability")
            self.assertEqual(set(map(tuple, group["want"])),
                             evidence[group["task"]].get(
                                 group["label_size"], set()))
        self.assertTrue(all(item["included"] for item in reports))

    def test_truncated_modular_task_is_skipped_and_reported(self) -> None:
        task = {"id": "capped", "train": [], "max_tree_ops": 2}

        class FamilyRecorder:
            def __init__(self, fixture):
                pass

            def load_family(self):
                return {"tasks": [task]}

        truncated = {
            "label_semantics": "retained_train_signature_graph_reachability",
            "candidate_signature_evaluations": 1,
            "candidate_cap": 1,
            "truncated": True,
            "retained_graph_complete": False,
            "bound": 2,
        }
        reports = []
        with patch("rsi_programmer.mgroups.TraceRecorder", FamilyRecorder), \
                patch("rsi_programmer.mgroups.solution_sigs_by_size_report",
                      return_value=({}, truncated)):
            self.assertEqual(modular_groups("ignored",
                                            supervision_report=reports), [])
        self.assertEqual(reports[0]["task"], "capped")
        self.assertFalse(reports[0]["included"])
        self.assertTrue(reports[0]["truncated"])


if __name__ == "__main__":
    unittest.main()
