#!/usr/bin/env python3
"""Fenced inward promotion gate for trainer self-edits.

One edit at a time: snapshot the base config, run base and trial
trainers in fenced workdirs, decide on paired held-out scores via
PromotionGate, then promote or roll back. Every outcome lands in
lineage; the budget caps attempts and crashes roll back too. The
proxy may only pre-screen elsewhere — this gate always full-evals.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from rsi_programmer.edits import apply_edit, describe
from rsi_programmer.edits import validate_edit
from rsi_programmer.lineage import mission_ledger
from rsi_programmer.promote import PromotionGate

Scores = dict[str, list[float]]
Trainer = Callable[[dict[str, Any], int, str], Scores]


class SelfGate:
    """Budgeted fenced trials with lineage on every outcome."""

    def close(self) -> None:
        """Flush and close the lineage handle."""
        self._log.close()

    def __init__(self, workdir: str, ledger: str,
                 mission: str = "self-gate", budget: int = 1,
                 min_gain: float = 0.05) -> None:
        """Bind fence directories, ledger, budget, and gain bar."""
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self._root = Path(workdir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._log = mission_ledger(ledger, mission)
        self._mission = mission
        self._budget = budget
        self._gate = PromotionGate(min_gain=min_gain)
        self._trials = 0

    def _record(self, kind: str, intent: dict[str, Any]) -> str:
        """Lineage entry; returns its digest for the verdict."""
        command = "%s-%04d" % (self._mission, self._trials)
        self._trials += 1
        self._log.record_intent(command, self._mission, 0, 1,
                                {"kind": kind, **intent})
        rows = self._log.commands_in(self._mission, "prepared")
        return str(rows[-1]["digest"])

    def propose(self, base: dict[str, Any],
                edit: dict[str, Any], trainer: Trainer,
                seed: int = 0) -> dict[str, Any]:
        """Fenced trial of one edit; promote or roll back."""
        try:
            validate_edit(edit)
        except ValueError as exc:
            digest = self._record("refuse-malformed",
                                  {"reason": str(exc)})
            return {"decision": "refuse", "reason": str(exc),
                    "config": dict(base), "lineage": digest,
                    "budget_left": self._budget}
        if self._budget <= 0:
            digest = self._record("budget-exhausted",
                                  {"edit": edit})
            return {"decision": "budget-exhausted",
                    "reason": "no attempts left",
                    "config": dict(base), "lineage": digest,
                    "budget_left": 0}
        self._budget -= 1
        trial = apply_edit(base, edit)
        start = time.perf_counter()
        try:
            base_dir = str(self._root / ("base-%d" % seed))
            trial_dir = str(self._root / ("trial-%d" % seed))
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            Path(trial_dir).mkdir(parents=True, exist_ok=True)
            ref = trainer(dict(base), seed, base_dir)["held"]
            got = trainer(dict(trial), seed, trial_dir)["held"]
        except Exception as exc:
            # The fence holds for any trainer failure: roll back.
            digest = self._record("rollback-crash",
                                  {"edit": edit,
                                   "reason": str(exc)})
            return {"decision": "rollback", "reason": str(exc),
                    "config": dict(base), "lineage": digest,
                    "budget_left": self._budget}
        wall = time.perf_counter() - start
        verdict = self._gate.decide(list(ref), list(got), 0, 0,
                                    wall)
        diff = describe(edit, base)
        if verdict["promote"]:
            digest = self._record("promote",
                                  {"edit": edit, "diff": diff,
                                   "mean_gain": verdict[
                                       "mean_gain"]})
            return {"decision": "promote", "reason": diff,
                    "config": trial, "lineage": digest,
                    "budget_left": self._budget,
                    "mean_gain": verdict["mean_gain"]}
        digest = self._record("refuse-no-gain",
                              {"edit": edit, "diff": diff,
                               "reasons": verdict["reasons"]})
        return {"decision": "refuse",
                "reason": "; ".join(verdict["reasons"]),
                "config": dict(base), "lineage": digest,
                "budget_left": self._budget}
