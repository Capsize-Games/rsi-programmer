#!/usr/bin/env python3
"""Scale-up loop acceptance on both families (#35)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import rsi_programmer.scale
from rsi_programmer.rich import fit_rich
from rsi_programmer import scale as rsi_scale
from rsi_programmer.scale import ScaleUp

SHADOW = "fixtures/rsi_shadow_tasks.json"
HARD = "fixtures/rsi_hard_tasks.json"


class ScaleUpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Run baselines once; evolution runs per test."""
        tmp = tempfile.mkdtemp(prefix="rsi-scale-")
        cls.loop = ScaleUp(SHADOW, HARD,
                           str(Path(tmp) / "scale.db"))
        cls.base = cls.loop.baselines()
        groups, extra = cls.loop.surrogate_groups()
        cls.groups = groups
        cls.loop._rich = fit_rich(extra)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.loop.close()

    def test_baselines_shape(self) -> None:
        for fam in ("w1", "hard"):
            part = self.base[fam]
            self.assertEqual(len(part["tasks"]), 4)
            self.assertGreater(part["cost"], 0)
        self.assertEqual(self.base["w1"]["solve_rate"], 1.0)
        self.assertEqual(self.base["hard"]["solve_rate"], 0.75)
        print("w1 cost=%d hard cost=%d"
              % (self.base["w1"]["cost"],
                 self.base["hard"]["cost"]))

    def test_surrogate_fidelity_recorded(self) -> None:
        self.assertTrue(all(len(g) > 1 for g in self.groups))
        self.assertEqual(len(self.loop._rich), 4)
        self.assertTrue(any(abs(w) > 0 for w in
                            self.loop._rich))

    def test_evolve_and_accept_no_regression(self) -> None:
        old_gen, old_budget = rsi_scale.GENERATIONS, \
            rsi_scale.EVOLVE_BUDGET_S
        rsi_scale.GENERATIONS, rsi_scale.EVOLVE_BUDGET_S = 1, 300
        try:
            evo = self.loop.evolve(self.groups)
            acc = self.loop._accept()
        finally:
            rsi_scale.GENERATIONS = old_gen
            rsi_scale.EVOLVE_BUDGET_S = old_budget
        self.assertGreaterEqual(len(evo["history"]), 1)
        for fam in ("w1", "hard"):
            base = self.base[fam]["solve_rate"]
            got = acc[fam]["solve_rate"]
            print("%s solve=%.2f cost=%d base_cost=%d"
                  % (fam, got, acc[fam]["cost"],
                     acc[fam]["baseline_cost"]))
            self.assertGreaterEqual(got, base)
        grown = len(self.loop.archive())
        self.assertGreater(grown, 0)
        print("archive entries=%d gens=%d"
              % (grown, evo["generations"]))


if __name__ == "__main__":
    unittest.main()
