#!/usr/bin/env python3
"""Tests for the word-gated lattice author (lattice v33)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.lattice import BitLattice
from rsi_programmer.lauthor import lattice_runs
from rsi_programmer.lauthor import lattice_search
from rsi_programmer.ltrain import train_lattice
from rsi_programmer.modules import ModularPolicy
from rsi_programmer.probe import enumerate_programs
from rsi_programmer.traces import TraceRecorder

FIXTURE = str(Path(__file__).resolve().parent.parent
              / "fixtures" / "rsi_shadow_tasks.json")
NOVEL = str(Path(__file__).resolve().parent.parent
            / "fixtures" / "rsi_novel_tasks.json")
TWIN = str(Path(__file__).resolve().parent.parent
           / "fixtures" / "rsi_ood_twin.json")
PLAIN = str(Path(__file__).resolve().parent.parent
            / "fixtures" / "rsi_ood_plain.json")
FULL = str(Path(__file__).resolve().parent.parent
           / "fixtures" / "rsi_ood_tasks.json")
BUDGET = 3000


def _load(path: str) -> list[dict[str, Any]]:
    """Task list from one fixture file."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["tasks"]


class LatticeAuthorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Train the team once; author shadow + novel."""
        tmp = tempfile.mkdtemp(prefix="rsi-lauth-")
        cls.lat = BitLattice(seed=0)
        cls.lat.load_encoder(ModularPolicy(seed=0).weights())
        train_lattice(cls.lat, FIXTURE, tmp + "/lat.db")
        cls.recorder = TraceRecorder(FIXTURE)
        cls.shadow = _load(FIXTURE)
        cls.shadow_per = lattice_runs(cls.lat, cls.shadow,
                                      BUDGET, 3, cls.recorder)
        cls.novel = _load(NOVEL)
        cls.novel_per = lattice_runs(cls.lat, cls.novel,
                                     BUDGET, 3,
                                     TraceRecorder(NOVEL))
        cls.ood_per = []
        for path in (TWIN, PLAIN):
            tasks = _load(path)
            cls.ood_per += lattice_runs(cls.lat, tasks,
                                        BUDGET, 3,
                                        TraceRecorder(path))
        full_tasks = _load(FULL)
        cls.full_per = lattice_runs(cls.lat, full_tasks,
                                    BUDGET, 3,
                                    TraceRecorder(FULL))

    def _run_one(self, task: dict[str, Any]) -> dict[str, Any]:
        """Author a single task with the lattice only."""
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        return lattice_search(self.lat, rows, target, bound,
                              BUDGET, str(task["id"]), 3, 6)

    def test_shadow_authors_cleanly(self) -> None:
        """Shadow solves come from agreement, oracle-checked."""
        solved = [r for r in self.shadow_per if r["solved"]]
        print("lattice shadow: %d/%d named-rate=%.3f" %
              (len(solved), len(self.shadow_per),
               sum(r["named_rate"] for r in self.shadow_per)
               / len(self.shadow_per)))
        self.assertGreater(len(solved), 0)
        for row in self.shadow_per:
            self.assertEqual(row["path"], "lattice")
            self.assertEqual(row["solved"], row["oracle"])
            if row["solved"]:
                self.assertTrue(row["agreed"])
                self.assertGreater(row["named_rate"], 0.0)

    def test_zero_budget_never_falls_back(self) -> None:
        """Zero budget stays unsolved; no hidden rescue path."""
        task = self.shadow[0]
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        expr, _, _ = enumerate_programs(rows, target, bound)
        self.assertIsNotNone(expr)
        out = lattice_search(self.lat, rows, target, bound, 0,
                             str(task["id"]), 3, 6)
        self.assertFalse(out["solved"])
        self.assertEqual(out.get("path"), "lattice")
        self.assertEqual(out.get("verdict"), "budget-exhausted")

    def test_novel_miss_stays_missed(self) -> None:
        """A capability miss never resolves by another path."""
        missed = [r for r in self.novel_per if not r["solved"]]
        print("lattice novel misses=%d" % len(missed))
        if not missed:
            return
        task = next(t for t in self.novel if t["id"] ==
                    missed[0]["task"])
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        expr, _, _ = enumerate_programs(rows, target, bound)
        self.assertIsNotNone(expr)
        out = self._run_one(task)
        self.assertFalse(out["solved"])
        self.assertEqual(out.get("path"), "lattice")
        self.assertEqual(out.get("verdict"),
                         "no-authored-solution")

    def test_novel_transfer_reported(self) -> None:
        """Novel-family word-exact solves with reject tally."""
        solved = [r["task"] for r in self.novel_per
                  if r["solved"]]
        verdicts: dict[str, int] = {}
        for row in self.novel_per:
            key = "solved" if row["solved"] else row["verdict"]
            verdicts[key] = verdicts.get(key, 0) + 1
        print("lattice novel: solved=%s rejects=%s"
              % (solved, verdicts))
        for row in self.novel_per:
            self.assertEqual(row["path"], "lattice")
            self.assertEqual(row["solved"], row["oracle"])

    def test_ood_headline_reported(self) -> None:
        """Author-only OOD numbers with path=lattice."""
        solved = [r["task"] for r in self.ood_per
                  if r["solved"]]
        verdicts: dict[str, int] = {}
        for row in self.ood_per:
            key = "solved" if row["solved"] else row["verdict"]
            verdicts[key] = verdicts.get(key, 0) + 1
        print("lattice ood: %d/%d solved=%s rejects=%s"
              % (len(solved), len(self.ood_per), solved,
                 verdicts))
        for row in self.ood_per:
            self.assertEqual(row["path"], "lattice")
            self.assertEqual(row["solved"], row["oracle"])

    def test_full_ir_boundary_reported(self) -> None:
        """Full-IR boundary: real misses carry real verdicts."""
        solved = [r["task"] for r in self.full_per
                  if r["solved"]]
        verdicts: dict[str, int] = {}
        for row in self.full_per:
            key = "solved" if row["solved"] else row["verdict"]
            verdicts[key] = verdicts.get(key, 0) + 1
        print("lattice full-ir: %d/%d solved=%s rejects=%s"
              % (len(solved), len(self.full_per), solved,
                 verdicts))
        for row in self.full_per:
            self.assertEqual(row["path"], "lattice")
            self.assertEqual(row["solved"], row["oracle"])
            if not row["solved"]:
                self.assertEqual(row["verdict"],
                                 "no-authored-solution")


if __name__ == "__main__":
    unittest.main()
