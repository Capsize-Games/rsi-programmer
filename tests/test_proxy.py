#!/usr/bin/env python3
"""Proxy screening predicts full-eval transfer gain.

Few-epoch few-seed screening on a W1 subset ranks candidate
seeds; fidelity is the measured rank agreement against full
transfer eval. Selection stays empty until proven, and the
screening report keeps custom data under 4096 bytes.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from rsi_programmer.proxy import REPORT_BUDGET, ProxyScreen
from rsi_programmer.proxy import rank_agreement

ROOT = Path(__file__).resolve().parent.parent
W1 = str(ROOT / "fixtures" / "rsi_shadow_tasks.json")
NOVEL = str(ROOT / "fixtures" / "rsi_novel_tasks.json")
CANDS = [{"id": "c0", "seed": 0}, {"id": "c1", "seed": 1},
         {"id": "c2", "seed": 2}]
NOVEL_IDS = {"prod_plus_sum", "sum_squares", "scaled_mix"}


def _novel_ids() -> set[str]:
    """Ids frozen in the novel fixture file."""
    fam = json.loads(Path(NOVEL).read_text(encoding="utf-8"))
    return {str(t["id"]) for t in fam["tasks"]}


class ProxyScreenTests(unittest.TestCase):
    """Screening cost, fidelity math, and the proof gate."""

    @classmethod
    def setUpClass(cls) -> None:
        """One real screening run shared by the cost tests."""
        cls.screen = ProxyScreen(W1, NOVEL)
        cls.result = cls.screen.screen(CANDS)

    def test_screen_scores_every_candidate(self) -> None:
        """One finite proxy score per candidate seed."""
        scores = self.result["scores"]
        self.assertEqual(set(scores), {"c0", "c1", "c2"})
        for value in scores.values():
            self.assertIsInstance(value, float)
            self.assertTrue(value == value)

    def test_screen_touches_no_novel_task(self) -> None:
        """Train and eval slices stay inside the W1 family."""
        seen = set(self.result["train_tasks"])
        seen |= set(self.result["eval_tasks"])
        self.assertTrue(seen)
        self.assertFalse(seen & _novel_ids())
        self.assertFalse(seen & NOVEL_IDS)

    def test_screen_runs_within_budget(self) -> None:
        """Few epochs and few seeds stay far under budget."""
        self.assertEqual(self.result["epochs"], 2)
        self.assertEqual(self.result["seeds"], 2)
        self.assertLess(self.result["wall_s"], 300.0)

    def test_perfect_agreement_proven_selects(self) -> None:
        """Concordant ranks prove the proxy; top proxy wins."""
        proxy = {"c0": 0.1, "c1": 0.2, "c2": 0.3}
        full = {"c0": -0.5, "c1": 0.0, "c2": 0.4}
        pact = self.screen.fidelity(proxy, full)
        self.assertEqual(pact["pairs"], 3)
        self.assertEqual(pact["concordant"], 3)
        self.assertEqual(pact["agreement"], 1.0)
        self.assertTrue(pact["proven"])
        picked = self.screen.select(proxy, full)
        self.assertTrue(picked["proven"])
        self.assertEqual(picked["winner"], "c2")

    def test_inverted_ranking_unproven_selects_nothing(
            self) -> None:
        """Discordant ranks leave the proxy judging nothing."""
        proxy = {"c0": 0.3, "c1": 0.2, "c2": 0.1}
        full = {"c0": 0.1, "c1": 0.2, "c2": 0.3}
        pact = self.screen.fidelity(proxy, full)
        self.assertEqual(pact["discordant"], 3)
        self.assertEqual(pact["agreement"], 0.0)
        self.assertFalse(pact["proven"])
        picked = self.screen.select(proxy, full)
        self.assertFalse(picked["proven"])
        self.assertIsNone(picked["winner"])
        self.assertEqual(picked["reason"], "unproven-proxy")

    def test_ties_are_counted_not_hidden(self) -> None:
        """A flat proxy scores half agreement, never proven."""
        proxy = {"c0": 0.2, "c1": 0.2, "c2": 0.2}
        full = {"c0": 0.1, "c1": 0.2, "c2": 0.3}
        pact = rank_agreement(proxy, full)
        self.assertEqual(pact["tied"], 3)
        self.assertEqual(pact["agreement"], 0.5)
        self.assertFalse(self.screen.fidelity(proxy,
                                             full)["proven"])

    def test_small_sample_never_proven(self) -> None:
        """Perfect agreement on one pair still proves nothing."""
        proxy = {"c0": 0.1, "c1": 0.2}
        full = {"c0": 0.1, "c1": 0.9}
        pact = self.screen.fidelity(proxy, full)
        self.assertEqual(pact["agreement"], 1.0)
        self.assertEqual(pact["pairs"], 1)
        self.assertFalse(pact["proven"])
        self.assertIsNone(self.screen.select(proxy,
                                            full)["winner"])

    def test_report_within_custom_budget(self) -> None:
        """Historical gains give fidelity without full eval."""
        historical = {"c0": 0.05, "c1": -0.2, "c2": 0.1}
        rep = self.screen.report(CANDS, historical=historical)
        self.assertIn("agreement", rep["fidelity"])
        self.assertIn("winner", rep["selection"])
        payload = {k: rep[k] for k in ("screen", "full",
                                            "fidelity", "selection")}
        size = len(json.dumps(payload, sort_keys=True).encode())
        self.assertEqual(rep["report_bytes"], size)
        total = len(json.dumps(rep, sort_keys=True).encode())
        self.assertLess(total, REPORT_BUDGET)
        self.assertTrue(rep["within_budget"])

    def test_full_eval_uses_novel_tasks(self) -> None:
        """The full protocol evaluates on the novel family."""
        rep = self.screen.full_eval(CANDS[:1], epochs=1)
        self.assertEqual(set(rep["eval_tasks"]), _novel_ids())
        self.assertEqual(set(rep["gains"]), {"c0"})


if __name__ == "__main__":
    unittest.main()
