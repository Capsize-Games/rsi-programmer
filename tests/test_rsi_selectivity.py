#!/usr/bin/env python3
"""Tests for lattice gate selectivity reporting (#39)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.author import refine_author, train_author
from rsi_programmer.lattice import BitLattice
from rsi_programmer.ltrain import train_lattice
from rsi_programmer.modules import ModularPolicy
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.selectivity import family_report
from rsi_programmer.selectivity import selectivity_report
from rsi_programmer.traces import TraceRecorder

ROOT = Path(__file__).resolve().parent.parent
FILES = ("fixtures/rsi_shadow_tasks.json",
         "fixtures/rsi_novel_tasks.json",
         "fixtures/rsi_ood_twin.json",
         "fixtures/rsi_ood_plain.json",
         "fixtures/rsi_ood_tasks.json")
BUDGET = 3000
WIDTH = 3


def _load(name: str) -> tuple[list[dict[str, Any]],
                              TraceRecorder]:
    """Tasks plus recorder for one fixture file."""
    path = str(ROOT / name)
    with open(path, encoding="utf-8") as handle:
        tasks = json.load(handle)["tasks"]
    return tasks, TraceRecorder(path)


class SelectivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Train lattice and beam policy once, report all."""
        tmp = tempfile.mkdtemp(prefix="rsi-sel-")
        cls.lat = BitLattice(seed=0)
        cls.lat.load_encoder(ModularPolicy(seed=0).weights())
        train_lattice(cls.lat, str(ROOT / FILES[0]),
                      tmp + "/lat.db")
        cls.policy = SearchPolicy(seed=0)
        train_author(cls.policy, str(ROOT / FILES[0]),
                     epochs=100)
        tasks = _load(FILES[0])[0]
        rows = [(str(t["id"]),
                 tuple((c[0], c[1]) for c in t["train"]),
                 tuple(c[2] for c in t["train"]),
                 int(t.get("max_tree_ops", 4))) for t in tasks]
        cls.weights, _ = refine_author(cls.policy, rows,
                                       seed=0)
        cls.families = {name.split("/")[-1].split(".")[0]:
                        _load(name) for name in FILES}
        cls.report = selectivity_report(
            cls.lat, cls.policy, cls.weights, cls.families,
            BUDGET, WIDTH)

    def test_report_covers_all_families(self) -> None:
        """Every family reports solves, rates, and tallies."""
        self.assertEqual(set(self.report),
                         {"rsi_shadow_tasks", "rsi_novel_tasks",
                          "rsi_ood_twin", "rsi_ood_plain",
                          "rsi_ood_tasks"})
        for name, part in self.report.items():
            self.assertLessEqual(part["solved"], part["tasks"])
            self.assertLessEqual(part["shuffle_solves"],
                                 part["tasks"])
            for key in ("named_rate", "echo_overlap"):
                self.assertGreaterEqual(part[key], 0.0)
                self.assertLessEqual(part[key], 1.0)
            for op in part["per_op"]:
                self.assertIn(op, ("add", "sub", "mul"))

    def test_family_report_is_deterministic(self) -> None:
        """Same inputs reproduce the same report exactly."""
        tasks, recorder = self.families["rsi_shadow_tasks"]
        first = family_report(self.lat, self.policy,
                              self.weights, tasks, recorder,
                              BUDGET, WIDTH)
        second = family_report(self.lat, self.policy,
                               self.weights, tasks,
                               recorder, BUDGET, WIDTH)
        self.assertEqual(first, second)
        self.assertEqual(first["solved"], second["solved"])
        self.assertEqual(first["named_rate"],
                         second["named_rate"])
        self.assertEqual(first["echo_overlap"],
                         second["echo_overlap"])
        self.assertEqual(first["shuffle_solves"],
                         second["shuffle_solves"])


if __name__ == "__main__":
    unittest.main()
