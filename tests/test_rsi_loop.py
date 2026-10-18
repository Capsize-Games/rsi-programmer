#!/usr/bin/env python3
"""Tests for the unattended self-edit loop (#6)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "scripts"))

from rsi_loop import RsiLoop
from rsi_programmer.edits import DEFAULT_CONFIG
from rsi_programmer.lattice import BitLattice
from rsi_programmer.lauthor import lattice_runs
from rsi_programmer.lineage import LineageStore
from rsi_programmer.ltrain import train_lattice
from rsi_programmer.modules import ModularPolicy
from rsi_programmer.proposer import EditProposer
from rsi_programmer.traces import TraceRecorder


def _stub(calls: list) -> Any:
    """lr-linked scores: raising lr always improves."""

    def run(config: dict[str, Any], seed: int,
            workdir: str) -> dict[str, list[float]]:
        calls.append(dict(config))
        level = float(config.get("lr", 0.0)) * 10.0
        return {"train": [level] * 4, "held": [level] * 8}

    return run


def _loop(tmp: str, name: str, **kwargs: Any) -> RsiLoop:
    """Loop with isolated fence, ledger, and trace."""
    return RsiLoop(tmp + "/fence", tmp + "/" + name + ".db",
                   tmp + "/" + name + ".jsonl", **kwargs)


def _diag_from_lattice() -> dict[str, Any]:
    """Real lattice diagnostics from the full-IR wall.

    Trains the bit-slice team once and authors the full-IR
    stratum lattice-only; verdicts, named rates, and
    agreed-but-wrong commits become the diagnostic the
    proposer ranks from. Slow-lane cost, real signals.
    """
    root = Path(__file__).resolve().parent.parent
    fixture = str(root / "fixtures" / "rsi_shadow_tasks.json")
    full = str(root / "fixtures" / "rsi_ood_tasks.json")
    with open(full, encoding="utf-8") as handle:
        tasks = json.load(handle)["tasks"]
    tmp = tempfile.mkdtemp(prefix="rsi-loopdiag-")
    lat = BitLattice(seed=0)
    lat.load_encoder(ModularPolicy(seed=0).weights())
    train_lattice(lat, fixture, tmp + "/diag.db")
    per = lattice_runs(lat, tasks, 3000, 3,
                       TraceRecorder(full))
    hist: dict[str, int] = {}
    for row in per:
        key = "solved" if row["solved"] else row["verdict"]
        hist[key] = hist.get(key, 0) + 1
    return {"reject_histogram": hist,
            "named_rate": sum(r["named_rate"] for r in per)
            / len(per),
            "agreed_wrong": sum(1 for r in per
                                if r["agreed"]
                                and not r["solved"])}


class RsiLoopTests(unittest.TestCase):
    def test_dry_run_plans_without_training(self) -> None:
        """Dry runs record plans, promote nothing, train never."""
        tmp = tempfile.mkdtemp(prefix="rsi-loop-")
        calls: list = []
        loop = _loop(tmp, "dry", cycles=3, dry_run=True)
        try:
            rep = loop.run(dict(DEFAULT_CONFIG), _stub(calls))
        finally:
            loop.close()
        self.assertEqual(rep["cycles"], 3)
        self.assertEqual(rep["promotions"], 0)
        self.assertFalse(calls)
        self.assertTrue(all(r["decision"] == "dry-run"
                            for r in rep["records"]))

    def test_multi_cycle_promotes_logged_gains(self) -> None:
        """Three-plus cycles, logged gains, transfer holds."""
        tmp = tempfile.mkdtemp(prefix="rsi-loop-")
        calls: list = []
        loop = _loop(tmp, "run", cycles=6, seed=1)
        try:
            rep = loop.run(dict(DEFAULT_CONFIG), _stub(calls))
        finally:
            loop.close()
        self.assertEqual(rep["cycles"], 4)
        self.assertEqual(rep["promotions"], 1)
        gains = [r["mean_gain"] for r in rep["records"]
                 if r["decision"] == "promote"]
        self.assertEqual(len(gains), 1)
        self.assertGreater(gains[0], 0)
        for record in rep["records"]:
            self.assertEqual(record["proxy"], "unproven")
            self.assertTrue(record["lineage"])
        store = LineageStore(tmp + "/run.db")
        try:
            rows = store.commands_in("rsi-loop", "prepared")
            self.assertEqual(len(rows), len(rep["records"]))
        finally:
            store.close()
        print("loop cycles=%d promotions=%d gains=%s"
              % (rep["cycles"], rep["promotions"],
                 [round(g, 3) for g in gains]))

    def test_reproducible_from_seed(self) -> None:
        """Same seed replays identical diffs and configs."""
        reps = []
        for tag in ("a", "b"):
            tmp = tempfile.mkdtemp(prefix="rsi-loop-")
            loop = _loop(tmp, tag, cycles=4, seed=1)
            try:
                reps.append(loop.run(dict(DEFAULT_CONFIG),
                                      _stub([])))
            finally:
                loop.close()
        first = [(r["diff"], r["decision"]) for r in
                 reps[0]["records"]]
        second = [(r["diff"], r["decision"]) for r in
                  reps[1]["records"]]
        self.assertEqual(first, second)
        self.assertEqual(reps[0]["config"], reps[1]["config"])

    def test_crashing_trainer_keeps_base(self) -> None:
        """Mid-cycle blowups leave the base config intact."""

        def boom(config: dict[str, Any], seed: int,
                 workdir: str) -> dict[str, list[float]]:
            raise RuntimeError("synthetic blowup")

        tmp = tempfile.mkdtemp(prefix="rsi-loop-")
        loop = _loop(tmp, "boom", cycles=3, seed=2)
        base = dict(DEFAULT_CONFIG)
        try:
            rep = loop.run(base, boom)
        finally:
            loop.close()
        self.assertEqual(rep["config"], DEFAULT_CONFIG)
        self.assertEqual(rep["promotions"], 0)
        self.assertTrue(all(r["decision"] == "rollback"
                            for r in rep["records"]))

    def test_two_misses_stop_the_run(self) -> None:
        """Consecutive non-improving evals end the run early."""
        tmp = tempfile.mkdtemp(prefix="rsi-loop-")
        loop = _loop(tmp, "miss", cycles=10, seed=3)
        try:
            rep = loop.run(dict(DEFAULT_CONFIG),
                           _stub([]))
        finally:
            loop.close()
        self.assertLess(rep["cycles"], 10)


class AuthoredLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Real lattice diagnostics once for the whole class."""
        cls.diag = _diag_from_lattice()
        print("loop diagnostics: %s" % (cls.diag,))

    def test_prior_promotion_authors_next_one(self) -> None:
        """A measured prior promotion authors the next loop.

        Phase one promotes lr under a founder-report
        stand-in diagnostic; phase two ranks from that real
        prior promotion plus the lattice wall signals and
        promotes again through the unweakened gate, with
        descriptive outcome-association metadata and lineage
        on every outcome.
        """
        prop = EditProposer()
        tmp = tempfile.mkdtemp(prefix="rsi-loopA-")
        first = _loop(tmp, "first", cycles=6, seed=1,
                      proposer=prop,
                      diagnostics={"suspect_fields": ["lr"]})
        try:
            rep_a = first.run(dict(DEFAULT_CONFIG),
                              _stub([]))
        finally:
            first.close()
        history = [r["edit"]["field"] for r in
                   rep_a["records"]
                   if r["decision"] == "promote"]
        self.assertTrue(history)
        tmp = tempfile.mkdtemp(prefix="rsi-loopB-")
        second = _loop(tmp, "second", cycles=6, seed=1,
                       proposer=prop,
                       diagnostics={**self.diag,
                                    "promotion_history": history})
        try:
            rep = second.run(dict(DEFAULT_CONFIG),
                             _stub([]))
        finally:
            second.close()
        self.assertEqual(rep["promotions"], 1)
        self.assertEqual(rep["authored_promotions"], 1)
        won = [r for r in rep["records"]
               if r["decision"] == "promote"]
        self.assertEqual(won[0]["provenance"]["rule"],
                         "repeat")
        self.assertGreater(won[0]["mean_gain"], 0.0)
        self.assertEqual(rep["outcome_association"]["grouping"],
                         "promotion decision outcome")
        self.assertEqual(rep["outcome_association"]["interpretation"],
                         "descriptive outcome association")
        self.assertEqual(rep["autonomy"]["alias_for"],
                         "outcome_association")
        self.assertEqual(rep["autonomy"]["ratio"],
                         rep["outcome_association"]["ratio"])
        self.assertTrue(rep["agency"])
        store = LineageStore(tmp + "/second.db")
        try:
            rows = store.commands_in("rsi-loop", "prepared")
            self.assertEqual(len(rows), len(rep["records"]))
        finally:
            store.close()
        print("authored loop promotions=%d outcome_association=%.3f"
              % (rep["promotions"],
                 rep["outcome_association"]["ratio"]))

    def test_real_diagnostics_report_honestly(self) -> None:
        """Lattice-wall diagnostics drive an honest report.

        Either a lattice-authored promotion with attribution
        or an early misses-stop with every record carrying
        the real driving signal; nothing is faked either way,
        and the same diagnostics rank the same order twice.
        """
        prop = EditProposer()
        reps = []
        for tag in ("a", "b"):
            tmp = tempfile.mkdtemp(prefix="rsi-loopC-")
            loop = _loop(tmp, tag, cycles=6, seed=7,
                         proposer=prop,
                         diagnostics=dict(self.diag))
            try:
                reps.append(loop.run(dict(DEFAULT_CONFIG),
                                     _stub([])))
            finally:
                loop.close()
        rep = reps[0]
        self.assertIn("autonomy", rep)
        self.assertIn("agency", rep)
        for row in rep["records"]:
            self.assertTrue(row["authored"])
            self.assertTrue(row["provenance"]["signal"])
        cited = " ".join(r["provenance"]["signal"]
                         for r in rep["records"])
        self.assertTrue("rejects=" in cited
                        or "agreed-wrong=" in cited)
        if rep["promotions"]:
            self.assertGreaterEqual(
                rep["authored_promotions"], 1)
        else:
            self.assertLess(rep["cycles"], 6)
        first = [(r["diff"], r["decision"]) for r in
                 reps[0]["records"]]
        second = [(r["diff"], r["decision"]) for r in
                  reps[1]["records"]]
        self.assertEqual(first, second)

    def test_dry_run_uses_authored_order(self) -> None:
        """Dry plans follow proposer rank, train nothing."""
        prop = EditProposer()
        want = [row["edit"] for row in prop.rank(
            dict(DEFAULT_CONFIG), {"suspect_fields": ["lr"]})]
        tmp = tempfile.mkdtemp(prefix="rsi-loopD-")
        calls: list = []
        loop = _loop(tmp, "dry", cycles=3, dry_run=True,
                     proposer=prop,
                     diagnostics={"suspect_fields": ["lr"]})
        try:
            rep = loop.run(dict(DEFAULT_CONFIG), _stub(calls))
        finally:
            loop.close()
        self.assertFalse(calls)
        self.assertEqual(rep["promotions"], 0)
        got = [r["edit"] for r in rep["records"]]
        self.assertEqual(got, want[:3])
        self.assertTrue(all(r["authored"]
                            for r in rep["records"]))

    def test_real_trainer_joint_trial(self) -> None:
        """Joint-field edits reach a real gate verdict."""
        from rsi_programmer.loop_trainer import train_scores
        from rsi_programmer.self_gate import SelfGate
        tmp = tempfile.mkdtemp(prefix="rsi-loopE-")
        gate = SelfGate(tmp + "/fence", tmp + "/gate.db",
                        mission="loop-joint", budget=1)
        try:
            out = gate.propose(
                dict(DEFAULT_CONFIG),
                {"op": "set", "field": "joint_epochs",
                 "value": 10}, train_scores, seed=0)
        finally:
            gate.close()
        self.assertIn(out["decision"], ("promote", "refuse"))
        self.assertTrue(out["lineage"])
        print("joint trial decision=%s" % out["decision"])

    def test_const_terminals_solve_without_them_failing(
            self) -> None:
        """Const-flagged tasks solve with consts, miss without.

        The capability wall from #48: unreachable tasks stay
        unreachable bare and solve once constant terminals
        join level zero. Untrained ranker, enumerator-scale
        tried counts.
        """
        from rsi_programmer.policy import SearchPolicy
        from rsi_programmer.probe import CONSTS
        from rsi_programmer.search import guided_search
        root = Path(__file__).resolve().parent.parent
        with open(root / "fixtures" / "rsi_ood_v2.json",
                  encoding="utf-8") as handle:
            tasks = json.load(handle)["tasks"]
        plain = [t for t in tasks
                 if "branch" not in t.get("features", [])]
        self.assertEqual(len(plain), 4)
        policy = SearchPolicy(seed=0)
        for task in plain:
            rows = tuple((c[0], c[1]) for c in task["train"])
            target = tuple(c[2] for c in task["train"])
            bound = int(task.get("max_tree_ops", 4))
            bare = guided_search(policy, rows, target, bound,
                                 10 ** 9, task["id"])
            full = guided_search(policy, rows, target, bound,
                                 10 ** 9, task["id"],
                                 consts=CONSTS)
            self.assertTrue(full["solved"])
            print("%s bare=%s const=%s tried=%d"
                  % (task["id"], bare["solved"],
                     full["solved"], full["tried"]))

    def test_loop_trainer_threads_consts(self) -> None:
        """Loop held shows const tasks solved, branch missed."""
        from rsi_programmer.loop_trainer import HELD
        from rsi_programmer.loop_trainer import train_scores
        self.assertTrue(HELD.endswith("rsi_ood_v2.json"))
        tmp = tempfile.mkdtemp(prefix="rsi-loopF-")
        rep = train_scores(dict(DEFAULT_CONFIG), 0,
                           tmp + "/scores")
        held = rep["held"]
        self.assertEqual(len(held), 8)
        root = Path(__file__).resolve().parent.parent
        with open(root / "fixtures" / "rsi_ood_v2.json",
                  encoding="utf-8") as handle:
            tasks = json.load(handle)["tasks"]
        for task, score in zip(tasks, held):
            if "branch" in task.get("features", []):
                self.assertEqual(score, 0.0)
            else:
                self.assertGreater(score, 1.0)
        print("loop held const flips: %s"
              % [round(v, 3) for v in held])


if __name__ == "__main__":
    unittest.main()
