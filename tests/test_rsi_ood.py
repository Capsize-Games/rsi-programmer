#!/usr/bin/env python3
"""Structural-OOD expression benchmark tests (#12)."""

from __future__ import annotations

import json
import unittest
from typing import Any

from rsi_programmer.ood import CAVEAT, assert_frozen
from rsi_programmer.ood import benchmark, contrast_tensors
from rsi_programmer.ood import guidance_similarity
from rsi_programmer.ood import weights_hash
from rsi_programmer.families import generate_family
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.probe import SELECT, Expr
from rsi_programmer.tasks import _formula, _mirror, _vacuous
from rsi_programmer.traces import TraceRecorder

OOD = "fixtures/rsi_ood_tasks.json"
PLAIN = "fixtures/rsi_ood_plain.json"
TWIN = "fixtures/rsi_ood_twin.json"
V2 = "fixtures/rsi_ood_v2.json"
TRAIN = "fixtures/rsi_plain_train.json"
SHADOW = "fixtures/rsi_shadow_tasks.json"
BUDGET = 3000
EPOCHS = 40


def _load(path: str) -> list[dict[str, Any]]:
    """Task list from one fixture file."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["tasks"]


def _leaf(name: str) -> Expr:
    """One input leaf."""
    return Expr("input", name, None, 0)


class MirrorTests(unittest.TestCase):
    def test_inverts_comparison_and_swaps_roles(self) -> None:
        """lt(x, y) mirrors to lt(y, x)."""
        expr = Expr("lt", _leaf("x"), _leaf("y"), 1)
        self.assertEqual(_formula(_mirror(expr)), "(y < x)")

    def test_preserves_consts_and_involutes(self) -> None:
        """Consts survive; mirroring twice restores roles."""
        expr = Expr("add", _leaf("x"),
                    Expr("const", 2, None, 0), 1)
        once = _mirror(expr)
        self.assertEqual(_formula(once), "(y + 2)")
        self.assertEqual(_formula(_mirror(once)), "(x + 2)")

    def test_twin_rows_align_oracles_differ(self) -> None:
        """Same rows, flipped outputs: memorized answers fail."""
        twin, plain = _load(TWIN), _load(PLAIN)
        self.assertEqual(len(twin), len(plain))
        aligned = sum(
            1 for a, b in zip(twin, plain)
            if [c[:2] for c in a["train"]] ==
            [c[:2] for c in b["train"]] and
            [c[:2] for c in a["held_out"]] ==
            [c[:2] for c in b["held_out"]])
        flipped = sum(
            1 for a, b in zip(twin, plain)
            if [c[2] for c in a["train"]] !=
            [c[2] for c in b["train"]] and
            [c[2] for c in a["held_out"]] !=
            [c[2] for c in b["held_out"]])
        self.assertEqual(aligned, len(twin))
        self.assertEqual(flipped, len(twin))

    def test_names_degenerate_v1_patterns(self) -> None:
        """Constant and out-of-range conditions are flagged."""
        const_cond = Expr("eq", Expr("const", 3, None, 0),
                          Expr("const", 5, None, 0), 1)
        const_task = Expr(SELECT, const_cond, _leaf("y"), 2,
                          third=_leaf("x"))
        self.assertIn("const-condition",
                      _vacuous(const_task))
        oob_cond = Expr("eq", _leaf("y"),
                        Expr("const", 5, None, 0), 1)
        oob_task = Expr(SELECT, oob_cond, _leaf("x"), 2,
                        third=_leaf("y"))
        self.assertIn("vacuous-condition",
                      _vacuous(oob_task))
        healthy_cond = Expr("lt", _leaf("x"), _leaf("y"), 1)
        healthy = Expr(SELECT, healthy_cond, _leaf("x"), 2,
                       third=_leaf("y"))
        self.assertEqual(_vacuous(healthy), [])

    def test_strict_filter_bites_and_freezes(self) -> None:
        """Strict seed-35 differs, matches file, old files hold."""
        loose = generate_family(35, 8, 4, 2, "gen_ood2",
                                mirror=True)
        filed = _load(V2)
        self.assertEqual(len(loose["tasks"]), len(filed))
        self.assertNotEqual(
            [t["task"] for t in loose["tasks"]],
            [t["task"] for t in filed])
        tight = generate_family(35, 8, 4, 2, "gen_ood2",
                                mirror=True, strict=True)
        self.assertEqual(tight, {"max_tree_ops": 4,
                                 "tasks": filed, "version": 1})
        twin = generate_family(34, 8, 4, 0, "gen_ood_u",
                               mirror=False, consts=False)
        self.assertEqual(twin["tasks"], _load(TWIN))

    def test_splits_verify_and_no_leakage(self) -> None:
        """Split hashes hold; held-out never trains its task."""
        for path in (OOD, PLAIN, TWIN, V2, TRAIN):
            for task in _load(path):
                self.assertEqual(
                    TraceRecorder.split_hash(task),
                    task["split_sha256"])
                train = {tuple(c[:2]) for c in task["train"]}
                held = {tuple(c[:2]) for c in task["held_out"]}
                self.assertEqual(train & held, set())


class FrozenBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Overfit narrow plus broadly trained policies."""
        cls.groups = SearchPolicy(seed=0).trace_groups(SHADOW)
        cls.narrow = SearchPolicy(seed=0)
        cls.narrow.fit(cls.groups[:2], EPOCHS)
        cls.broad = SearchPolicy(seed=0)
        cls.broad.fit(cls.groups, EPOCHS)
        cls.twin, cls.plain = _load(TWIN), _load(PLAIN)
        cls.pairs = contrast_tensors(cls.twin, cls.plain)
        cls.extra = SearchPolicy(seed=0).trace_groups(TRAIN)
        cls.wide = SearchPolicy(seed=0)
        cls.wide.fit(cls.groups + cls.extra, EPOCHS,
                     contrast=cls.pairs)

    def test_weights_locked_through_eval(self) -> None:
        """Frozen run leaves every weight byte untouched."""
        before = weights_hash(self.broad)
        report = benchmark(self.narrow, self.broad,
                           self.plain[:2], BUDGET,
                           name="ood-test")
        self.assertEqual(weights_hash(self.broad), before)
        self.assertTrue(report["weights_unchanged"])
        self.assertEqual(report["tasks"], 2)
        self.assertEqual(report["caveat"], CAVEAT)
        self.assertAlmostEqual(
            report["expression_gap"],
            report["broad_solve"] - report["narrow_solve"])
        print("ood-test: narrow=%.3f broad=%.3f gap=%+.3f"
              % (report["narrow_solve"], report["broad_solve"],
                 report["expression_gap"]))

    def test_tamper_aborts(self) -> None:
        """A moved weight fails the frozen check."""
        before = weights_hash(self.narrow)
        moved = SearchPolicy(seed=0)
        moved._net.load_state_dict(
            self.narrow._net.state_dict())
        moved.perturb(sigma=0.5, seed=1)
        with self.assertRaises(RuntimeError):
            assert_frozen(before, moved)

    def test_contrast_training_separates_guidance(self) -> None:
        """Mirror penalty pushes broad guidance off baseline."""
        self.assertEqual(len(self.pairs), len(self.twin))
        first, second = self.pairs[0]
        self.assertEqual(first.shape, second.shape)
        self.assertGreater(first.shape[0], 0)
        shaped = SearchPolicy(seed=0)
        shaped.fit(self.groups, EPOCHS, contrast=self.pairs)
        sim = guidance_similarity(shaped, self.twin, self.plain)
        self.assertLessEqual(sim["mean"], 0.8946)
        print("contrast blindness: %.4f (baseline 0.9446)"
              % sim["mean"])

    def test_expanded_training_keeps_discipline(self) -> None:
        """Wide training runs frozen with caveat attached."""
        rep = benchmark(self.narrow, self.wide, self.plain[:2],
                        BUDGET, name="ood-wide")
        self.assertEqual(rep["tasks"], 2)
        self.assertEqual(rep["caveat"], CAVEAT)
        self.assertTrue(rep["weights_unchanged"])
        sim = guidance_similarity(self.wide, self.twin,
                                  self.plain)
        self.assertGreaterEqual(sim["mean"], -1.0)
        self.assertLessEqual(sim["mean"], 1.0)
        print("wide: narrow=%.3f wide=%.3f blindness=%.4f"
              % (rep["narrow_solve"], rep["broad_solve"],
                 sim["mean"]))

    def test_overfit_policy_blinder_to_mirror(self) -> None:
        """Narrow guidance barely notices the reversal."""
        narrow_sim = guidance_similarity(
            self.narrow, self.twin, self.plain)
        broad_sim = guidance_similarity(
            self.broad, self.twin, self.plain)
        self.assertEqual(narrow_sim["pairs"], len(self.twin))
        for sim in (narrow_sim, broad_sim):
            self.assertGreaterEqual(sim["mean"], 0.0)
            self.assertLessEqual(sim["mean"], 1.0)
        self.assertGreater(narrow_sim["mean"],
                           broad_sim["mean"])
        print("blindness: narrow=%.4f broad=%.4f"
              % (narrow_sim["mean"], broad_sim["mean"]))


if __name__ == "__main__":
    unittest.main()
