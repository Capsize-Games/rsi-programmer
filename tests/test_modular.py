#!/usr/bin/env python3
"""Tests for the factored transmitting policy (#1)."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from rsi_programmer.evolve import guided_search
from rsi_programmer.mgroups import modular_groups, module_report
from rsi_programmer.modules import ModularPolicy
from rsi_programmer.policy import FEATURE_ENCODING, SearchPolicy
from rsi_programmer.traces import TraceRecorder

ROOT = Path(__file__).resolve().parent.parent
SHADOW = str(ROOT / "fixtures" / "rsi_shadow_tasks.json")


def _budget() -> int:
    """Frozen enumerator cost as the search budget."""
    rec = TraceRecorder(SHADOW)
    return sum(int(r["tried"])
               for r in rec.record_all()["records"])


class ModularPolicyTests(unittest.TestCase):
    def test_snapshot_round_trips_scores(self) -> None:
        """Restore reproduces identical combined rankings."""
        first = ModularPolicy(seed=0)
        snap = first.snapshot()
        second = ModularPolicy(seed=99)
        second.restore(snap)
        self.assertEqual(snap["feature_encoding"], FEATURE_ENCODING)
        feats = [[float(i == j) for j in range(12)]
                 for i in range(6)]
        before = first.head_rates(feats, gap=0.5)
        after = second.head_rates(feats, gap=0.5)
        self.assertEqual(before, after)

    def test_per_module_metrics_reported(self) -> None:
        """Op and operand heads measured against chance."""
        policy = ModularPolicy(seed=0)
        groups = modular_groups(SHADOW)
        self.assertTrue(groups)
        loss = policy.fit(groups, epochs=2)
        self.assertTrue(loss == loss)
        rep = module_report(policy, groups)
        self.assertGreater(rep["layers"], 0)
        for key in ("op_argmax_accuracy", "op_argmax_chance",
                    "left_operand_top1_match",
                    "left_operand_top1_chance", "op_accuracy",
                    "op_chance", "operand_top1", "operand_chance"):
            self.assertIn(key, rep)
            self.assertGreaterEqual(rep[key], 0.0)
            self.assertLessEqual(rep[key], 1.0)
        print("modular=%s" % rep)

    def test_guided_search_solves_shadow_family(self) -> None:
        """Factored ranker solves the same tasks via search."""
        policy = ModularPolicy(seed=0)
        budget = _budget()
        rec = TraceRecorder(SHADOW)
        for task in rec.load_family()["tasks"]:
            rows = tuple((c[0], c[1]) for c in task["train"])
            target = tuple(c[2] for c in task["train"])
            bound = int(task.get("max_tree_ops", 4))
            out = guided_search(policy, rows, target, bound,
                                budget, str(task["id"]))
            self.assertTrue(out["solved"], task["id"])
            self.assertLessEqual(out["tried"], budget)

    def test_explicit_continuation_carries_state(self) -> None:
        """Sequence callers can explicitly continue prior state."""
        policy = ModularPolicy(seed=0)
        drive = [[0.5] * 12 for _ in range(4)]
        fresh = policy.head_rates(drive, gap=0.9)
        policy.head_rates(drive, gap=0.1, continue_state=True)
        primed = policy.head_rates(drive, gap=0.9,
                                   continue_state=True)
        self.assertFalse(_close(fresh, primed))

    def test_delayed_feedback_changes_outputs_within_one_drive(self) -> None:
        """The feedback edge remains active in a fresh drive."""
        policy = ModularPolicy(seed=0)
        torch.manual_seed(100)
        drive = torch.rand(8, 12).tolist()
        intact = policy.head_rates(drive, gap=0.9)
        ablated = ModularPolicy(seed=0)
        with torch.no_grad():
            for name, param in ablated._net.named_parameters():
                if "fb_fc" in name:
                    param.zero_()
        cut = ablated.head_rates(drive, gap=0.9)
        self.assertFalse(_close(intact, cut))

    def test_same_rows_are_independent_of_prior_candidate_batches(
            self) -> None:
        """Scoring independent batches never inherits prior state."""
        policy = ModularPolicy(seed=2)
        feats = [[float((i + j) % 3) / 2.0 for j in range(12)]
                 for i in range(5)]
        other = [[float((i * j) % 5) / 4.0 for j in range(12)]
                 for i in range(5)]
        first = policy.scores(feats)
        policy.scores(other)
        self.assertEqual(first, policy.scores(feats))


def _close(first: dict, second: dict) -> bool:
    """True when every head rate matches exactly."""
    return all(first[k] == second[k] for k in first)


class SharedRankPathTests(unittest.TestCase):
    def test_old_policy_ignores_new_rank_args(self) -> None:
        """Extended rank signature leaves monolith behavior."""
        feats = [[float(i == j) for j in range(12)]
                 for i in range(6)]
        self.assertEqual(SearchPolicy(seed=0).rank(feats),
                         SearchPolicy(seed=0).rank(feats, None,
                                                   None))


if __name__ == "__main__":
    unittest.main()
