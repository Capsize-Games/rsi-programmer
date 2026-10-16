#!/usr/bin/env python3
"""Tests for the fenced inward promotion gate (#4)."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.edits import DEFAULT_CONFIG
from rsi_programmer.lineage import LineageStore
from rsi_programmer.self_gate import SelfGate

ROOT = Path(__file__).resolve().parent.parent


def _stub(score: float, calls: list) -> Any:
    """Trainer stub returning a flat held-out score."""

    def run(config: dict[str, Any], seed: int,
            workdir: str) -> dict[str, list[float]]:
        calls.append((dict(config), seed, workdir))
        level = float(config.get("lr", 0.0)) * 10.0 + score
        return {"train": [level] * 4, "held": [level] * 8}

    return run


_OPEN: list[SelfGate] = []


def tearDownModule() -> None:
    """Close every gate ledger opened by these tests."""
    for gate in _OPEN:
        gate.close()


def _gate(tmp: str, name: str, budget: int = 2) -> SelfGate:
    """Gate with an isolated fence and ledger."""
    gate = SelfGate(tmp + "/fence", tmp + "/" + name + ".db",
                    mission=name, budget=budget)
    _OPEN.append(gate)
    return gate


class SelfGateTests(unittest.TestCase):
    def test_beneficial_edit_promotes_with_lineage(self) -> None:
        """Uniform gain clears bootstrap and gain bars."""
        tmp = tempfile.mkdtemp(prefix="rsi-gate-")
        gate = _gate(tmp, "promote")
        base = dict(DEFAULT_CONFIG)
        base["lr"] = 0.05
        edit = {"op": "set", "field": "lr", "value": 0.1}
        out = gate.propose(base, edit, _stub(0.0, []))
        self.assertEqual(out["decision"], "promote")
        self.assertEqual(out["config"]["lr"], 0.1)
        self.assertEqual(base["lr"], 0.05)
        self.assertTrue(out["lineage"])
        store = LineageStore(tmp + "/promote.db")
        try:
            rows = store.commands_in("promote", "prepared")
            self.assertTrue(rows)
        finally:
            store.close()

    def test_harmful_edit_refused_base_intact(self) -> None:
        """Red team: uniform loss refuses with logged reason."""
        tmp = tempfile.mkdtemp(prefix="rsi-gate-")
        gate = _gate(tmp, "redteam")
        base = dict(DEFAULT_CONFIG)
        base["lr"] = 0.1
        calls: list = []
        edit = {"op": "set", "field": "lr", "value": 0.05}
        before = dict(base)
        out = gate.propose(base, edit, _stub(0.0, calls))
        self.assertEqual(out["decision"], "refuse")
        self.assertEqual(out["config"], before)
        self.assertEqual(base, before)
        self.assertTrue(out["reason"])
        self.assertTrue(calls)

    def test_budget_exactness(self) -> None:
        """Zero budget runs nothing; N attempts then exhausted."""
        tmp = tempfile.mkdtemp(prefix="rsi-gate-")
        calls: list = []
        edit = {"op": "set", "field": "lr", "value": 0.1}
        empty = _gate(tmp, "empty", budget=0)
        out = empty.propose(dict(DEFAULT_CONFIG), edit,
                            _stub(0.0, calls))
        self.assertEqual(out["decision"], "budget-exhausted")
        self.assertFalse(calls)
        one = _gate(tmp, "one", budget=1)
        one.propose(dict(DEFAULT_CONFIG), edit,
                    _stub(0.0, calls))
        out = one.propose(dict(DEFAULT_CONFIG), edit,
                          _stub(0.0, calls))
        self.assertEqual(out["decision"], "budget-exhausted")
        self.assertEqual(out["budget_left"], 0)

    def test_malformed_edit_never_trains(self) -> None:
        """Bad edits refuse before any trainer call."""
        tmp = tempfile.mkdtemp(prefix="rsi-gate-")
        calls: list = []
        gate = _gate(tmp, "malformed")
        out = gate.propose(dict(DEFAULT_CONFIG),
                           {"op": "set", "field": "nope",
                            "value": 1},
                           _stub(0.0, calls))
        self.assertEqual(out["decision"], "refuse")
        self.assertFalse(calls)

    def test_crash_rolls_back_and_records(self) -> None:
        """Trainer exceptions become recorded rollbacks."""
        tmp = tempfile.mkdtemp(prefix="rsi-gate-")
        gate = _gate(tmp, "crash")

        def boom(config: dict[str, Any], seed: int,
                 workdir: str) -> dict[str, list[float]]:
            raise RuntimeError("synthetic blowup")

        base = dict(DEFAULT_CONFIG)
        edit = {"op": "set", "field": "pop", "value": 8}
        out = gate.propose(base, edit, boom)
        self.assertEqual(out["decision"], "rollback")
        self.assertEqual(out["config"], base)
        self.assertIn("blowup", out["reason"])
        store = LineageStore(tmp + "/crash.db")
        try:
            kinds = [r["intent"] for r in
                     store.commands_in("crash", "prepared")]
            self.assertTrue(any("rollback-crash" in k
                                for k in kinds))
        finally:
            store.close()

    def test_judge_files_untouched_by_trial(self) -> None:
        """Gate and sandbox bytes identical after a full trial."""
        before = {name: hashlib.sha256(
            (ROOT / "rsi_programmer" / name).read_bytes()
        ).hexdigest() for name in ("promote.py", "sandbox.py")}
        tmp = tempfile.mkdtemp(prefix="rsi-gate-")
        gate = _gate(tmp, "hash")
        gate.propose(dict(DEFAULT_CONFIG),
                     {"op": "set", "field": "pop", "value": 8},
                     _stub(0.0, []))
        after = {name: hashlib.sha256(
            (ROOT / "rsi_programmer" / name).read_bytes()
        ).hexdigest() for name in ("promote.py", "sandbox.py")}
        self.assertEqual(before, after)

    def test_real_trainer_protocol_holds(self) -> None:
        """Smallest real training loop obeys the fence."""
        from rsi_programmer.policy import SearchPolicy
        from rsi_programmer.traces import TraceRecorder
        rec = TraceRecorder(str(ROOT / "fixtures"
                                / "rsi_shadow_tasks.json"))
        tasks = rec.load_family()["tasks"][:2]

        def trainer(config: dict[str, Any], seed: int,
                    workdir: str) -> dict[str, list[float]]:
            pol = SearchPolicy(seed=seed,
                               hidden=int(config["hidden"]))
            groups = pol.trace_groups(
                str(ROOT / "fixtures" / "rsi_shadow_tasks.json"))
            pol.fit(groups, 2)
            held = []
            for task in tasks:
                rows = tuple((c[0], c[1]) for c in task["train"])
                target = tuple(c[2] for c in task["train"])
                from rsi_programmer.evolve import guided_search
                out = guided_search(pol, rows, target,
                                    int(task.get("max_tree_ops",
                                                 4)),
                                    10 ** 9, str(task["id"]))
                held.append(1.0 if out["solved"] else 0.0)
            return {"train": held, "held": held}

        tmp = tempfile.mkdtemp(prefix="rsi-gate-")
        gate = _gate(tmp, "real", budget=2)
        base = dict(DEFAULT_CONFIG)
        left_before = gate._budget
        out = gate.propose(base,
                           {"op": "set", "field": "hidden",
                            "value": 8},
                           trainer, seed=0)
        self.assertIn(out["decision"], ("promote", "refuse"))
        self.assertEqual(base["hidden"], 16)
        self.assertEqual(gate._budget, left_before - 1)
        self.assertTrue(out["lineage"])


if __name__ == "__main__":
    unittest.main()
