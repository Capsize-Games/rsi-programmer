#!/usr/bin/env python3
"""Tests for organization metrics (#11)."""

from __future__ import annotations

import unittest

from rsi_programmer.autonomy import agency_profile, autonomy_ratio
from rsi_programmer.autonomy import integration_divergence
from rsi_programmer.modules import ModularPolicy
from rsi_programmer.policy import SearchPolicy


def _feats(rows: int = 8, dims: int = 12) -> list[list[float]]:
    """Deterministic non-degenerate drive batch."""
    return [[float((i + j) % 3) / 2.0 for j in range(dims)]
            for i in range(rows)]


class AutonomyRatioTests(unittest.TestCase):
    def test_full_separation_scores_one(self) -> None:
        """Disjoint histories own all the variance."""
        rep = autonomy_ratio({"h1": [0.0, 0.0],
                              "h2": [1.0, 1.0]})
        self.assertEqual(rep["ratio"], 1.0)
        self.assertEqual(rep["histories"], 2)
        self.assertEqual(rep["samples"], 4)

    def test_shared_spread_scores_zero(self) -> None:
        """Identical spreads leave nothing to history."""
        rep = autonomy_ratio({"h1": [0.0, 1.0],
                              "h2": [0.0, 1.0]})
        self.assertEqual(rep["ratio"], 0.0)

    def test_degenerate_inputs_score_zero(self) -> None:
        """One history, empty input, or flat lines score zero."""
        self.assertEqual(autonomy_ratio({"h1": [1.0]})
                         ["ratio"], 0.0)
        self.assertEqual(autonomy_ratio({})["ratio"], 0.0)
        self.assertEqual(autonomy_ratio({"h1": [2.0, 2.0],
                                         "h2": [2.0, 2.0]})
                         ["ratio"], 0.0)

    def test_partial_variance_fraction(self) -> None:
        """Between over total on a mixed population."""
        rep = autonomy_ratio({"h1": [0.0, 0.0, 1.0],
                              "h2": [1.0, 1.0, 1.0]})
        self.assertAlmostEqual(rep["ratio"], 0.5)


class DivergenceTests(unittest.TestCase):
    def test_chain_reports_no_integration(self) -> None:
        """Feedforward monolith has nothing to sever."""
        policy = SearchPolicy(seed=0)
        rep = integration_divergence(
            policy, _feats(),
            lambda outs: outs["out"][:, 0])
        self.assertEqual(rep["divergence"], 0.0)
        self.assertEqual(rep["severed"], [])
        self.assertEqual(rep["reason"], "feedforward chain")

    def test_feedback_edge_lists_and_measures(self) -> None:
        """Modular feedback severs and reports a number."""
        policy = ModularPolicy(seed=0)
        rep = integration_divergence(
            policy, _feats(8, 13),
            lambda outs: outs["st"][:, 0])
        self.assertEqual(rep["severed"], ["fb_fc"])
        self.assertGreaterEqual(rep["divergence"], 0.0)
        again = integration_divergence(
            policy, _feats(8, 13),
            lambda outs: outs["st"][:, 0])
        self.assertEqual(rep["divergence"],
                         again["divergence"])

    def test_driven_feedback_moves_outputs(self) -> None:
        """Strong drive makes the cut bite: positive number."""
        policy = ModularPolicy(seed=0)
        drive = [[0.5] * 12 + [0.9] for _ in range(8)]
        read = lambda outs: outs["op"].mean(dim=1) + \
            outs["lo"][:, 0] + outs["ro"][:, 0] + \
            outs["st"].mean(dim=1)
        rep = integration_divergence(policy, drive, read)
        self.assertEqual(rep["severed"], ["fb_fc"])
        self.assertAlmostEqual(rep["divergence"], 0.208333,
                               places=5)


class AgencyProfileTests(unittest.TestCase):
    def _entries(self, spread: float) -> list[dict]:
        base = [0.0, 0.0]
        far = [spread, 0.0]
        return [{"generation": 0,
                 "weights": {"a": base, "b": far},
                 "fitness": {"correctness": 0.5}},
                {"generation": 0,
                 "weights": {"a": base, "b": base},
                 "fitness": {"correctness": 0.5}},
                {"generation": 1,
                 "weights": {"a": base, "b": base},
                 "fitness": {"correctness": 1.0}},
                {"generation": 1,
                 "weights": {"a": base, "b": base},
                 "fitness": {"correctness": 1.0}}]

    def test_shrinking_diversity_converges(self) -> None:
        """Collapsing spread with rising fitness converges."""
        prof = agency_profile(self._entries(4.0))
        self.assertEqual([p["generation"] for p in prof],
                         [0, 1])
        self.assertEqual(prof[0]["direction"], "start")
        self.assertEqual(prof[1]["direction"], "converging")
        self.assertGreater(prof[0]["diversity"], 0.0)
        self.assertEqual(prof[1]["diversity"], 0.0)

    def test_empty_archive_empty_profile(self) -> None:
        """No entries, no profile."""
        self.assertEqual(agency_profile([]), [])


if __name__ == "__main__":
    unittest.main()
