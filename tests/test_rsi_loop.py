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
from rsi_programmer.lineage import LineageStore


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


if __name__ == "__main__":
    unittest.main()
