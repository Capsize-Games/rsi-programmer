#!/usr/bin/env python3
"""Tests for the deterministic task generator (#5)."""

from __future__ import annotations

import json
import random
import unittest
from pathlib import Path

from rsi_programmer.policy import SearchPolicy
from rsi_programmer.promote import PromotionGate
from rsi_programmer.families import curriculum_params
from rsi_programmer.families import generate_family, write_family
from rsi_programmer.tasks import TRAIN_ROWS, generate_task
from rsi_programmer.traces import TraceRecorder

ROOT = Path(__file__).resolve().parent.parent
TRAIN = str(ROOT / "fixtures" / "rsi_gen_train.json")
NOVEL = str(ROOT / "fixtures" / "rsi_gen_novel.json")


def _load(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


class TaskGeneratorTests(unittest.TestCase):
    def test_same_seed_byte_identical(self) -> None:
        """Determinism: one seed always yields the same bytes."""
        first = generate_family(7, 24, 4, 2, "gen_train")
        second = generate_family(7, 24, 4, 2, "gen_train")
        self.assertEqual(first, second)
        tmp = self._tmp("det.json")
        write_family(tmp, first)
        write_family(self._tmp("det2.json"), second)
        self.assertEqual(Path(tmp).read_bytes(),
                         Path(self._tmp("det2.json")).read_bytes())

    def test_frozen_splits_and_counts(self) -> None:
        """24+12 tasks, matching hashes, disjoint rows."""
        train = _load(TRAIN)["tasks"]
        novel = _load(NOVEL)["tasks"]
        self.assertEqual(len(train), 24)
        self.assertEqual(len(novel), 12)
        rec = TraceRecorder(TRAIN)
        for task in train + novel:
            self.assertEqual(rec.split_hash(task),
                             task["split_sha256"])
            self.assertEqual(len(task["train"]), TRAIN_ROWS)
            pairs = [(c[0], c[1]) for c in task["train"]]
            held = [(c[0], c[1]) for c in task["held_out"]]
            self.assertEqual(len(set(pairs)), len(pairs))
            self.assertFalse(set(pairs) & set(held))
            self.assertTrue(task.get("features"))

    def test_branch_tasks_require_branch(self) -> None:
        """A fresh branch task fails bare and const-only search."""
        from rsi_programmer.probe import CONSTS, enumerate_programs
        rng = random.Random(3)
        task = generate_task(rng, "probe_branch", 4, True, True)
        self.assertIn("branch", task["features"])
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bare, _, _ = enumerate_programs(rows, target, 4)
        self.assertIsNone(bare)
        const_only, _, _ = enumerate_programs(rows, target, 4,
                                              consts=CONSTS)
        self.assertIsNone(const_only)
        expr, _, _ = enumerate_programs(rows, target, 4,
                                        consts=CONSTS,
                                        branch=True)
        self.assertIsNotNone(expr)

    def test_curriculum_loop_closed(self) -> None:
        """Gate failures become family parameters and fresh tasks."""
        gate = PromotionGate()
        rep = gate.curriculum([{"kind": "tie"},
                               {"kind": "tie"},
                               {"kind": "no-authored-solution"}])
        params = curriculum_params(rep)
        self.assertGreater(params["count"], 0)
        family = generate_family(99, params["count"], 4,
                                 params.get("branch_every", 2),
                                 "loop")
        self.assertEqual(len(family["tasks"]), params["count"])
        rec = TraceRecorder(TRAIN)
        for task in family["tasks"]:
            self.assertEqual(rec.split_hash(task),
                             task["split_sha256"])

    def test_safety_report_yields_no_tasks(self) -> None:
        """Safety failures are a gate issue, never a data order."""
        gate = PromotionGate()
        rep = gate.curriculum([{"kind": "forged-completion"}])
        params = curriculum_params(rep)
        self.assertEqual(params["count"], 0)

    def _tmp(self, name: str) -> str:
        path = Path("/tmp") / ("rsi-tasks-" + name)
        return str(path)


if __name__ == "__main__":
    unittest.main()
