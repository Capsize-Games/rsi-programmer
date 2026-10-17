#!/usr/bin/env python3
"""Tests for the diagnostic proposer (lattice track, #41)."""
from __future__ import annotations

import random
import tempfile
import unittest
from typing import Any

from rsi_programmer.edits import DEFAULT_CONFIG, all_edits
from rsi_programmer.edits import validate_edit
from rsi_programmer.proposer import CAPACITY, COUPLING
from rsi_programmer.proposer import EditProposer
from rsi_programmer.self_gate import SelfGate


def _stub(calls: list) -> Any:
    """lr-linked scores: only raising lr promotes."""

    def run(config: dict[str, Any], seed: int,
            workdir: str) -> dict[str, list[float]]:
        calls.append(dict(config))
        level = float(config.get("lr", 0.0)) * 10.0
        return {"train": [level] * 4, "held": [level] * 8}

    return run


def _trials(winning: list[dict[str, Any]], stub: Any) -> int:
    """Trials to first promotion driving the stub by hand."""
    tmp = tempfile.mkdtemp(prefix="rsi-prop-")
    gate = SelfGate(tmp + "/fence", tmp + "/gate.db",
                    mission="proposer", budget=100)
    try:
        base = dict(DEFAULT_CONFIG)
        for num, edit in enumerate(winning, 1):
            out = gate.propose(base, edit, stub)
            if out["decision"] == "promote":
                return num
            base = out["config"]
    finally:
        gate.close()
    return -1


class ProposerTests(unittest.TestCase):
    def test_suspect_field_beats_seeded_order(self) -> None:
        """Suspect diagnostics reach promotion in fewer trials.

        The stub promotes only on raising lr; the proposer
        learns lr is suspect from the diagnostic while the
        baseline walks the alphabet in seeded order.
        """
        prop = EditProposer()
        diag = {"suspect_fields": ["lr"]}
        mine = [row["edit"] for row in
                prop.rank(dict(DEFAULT_CONFIG), diag)]
        rng = random.Random(0)
        pool = all_edits(dict(DEFAULT_CONFIG))
        rng.shuffle(pool)
        stub_a: list = []
        stub_b: list = []
        authored = _trials(mine, _stub(stub_a))
        blind = _trials(pool, _stub(stub_b))
        print("proposer trials=%d seeded trials=%d"
              % (authored, blind))
        self.assertGreater(authored, 0)
        self.assertGreater(blind, 0)
        self.assertLessEqual(authored, blind)

    def test_coupling_fields_lead_on_agreed_wrong(self) -> None:
        """Agreed-but-wrong commits rank coupling first."""
        prop = EditProposer()
        diag = {"agreed_wrong": 3}
        mine = prop.rank(dict(DEFAULT_CONFIG), diag)
        lead = {row["edit"]["field"] for row in mine[:3]}
        self.assertTrue(lead & set(COUPLING))

    def test_capacity_fields_lead_on_reject_wall(self) -> None:
        """A no-solution reject wall ranks capacity first."""
        prop = EditProposer()
        diag = {"reject_histogram":
                {"no-authored-solution": 12, "solved": 4},
                "named_rate": 0.25}
        mine = prop.rank(dict(DEFAULT_CONFIG), diag)
        lead = {row["edit"]["field"] for row in mine[:4]}
        self.assertTrue(lead & set(CAPACITY))

    def test_every_proposal_validates_with_provenance(self) -> None:
        """All proposals validate; each names its signal."""
        prop = EditProposer()
        diag = {"suspect_fields": ["lr"],
                "promotion_history": ["steps"],
                "agreed_wrong": 1,
                "reject_histogram":
                {"no-authored-solution": 5},
                "named_rate": 0.4}
        mine = prop.rank(dict(DEFAULT_CONFIG), diag)
        self.assertEqual(len(mine),
                         len(all_edits(dict(DEFAULT_CONFIG))))
        for row in mine:
            validate_edit(row["edit"])
            self.assertIn(row["provenance"]["rule"],
                          ("suspect", "repeat", "coupling",
                           "capacity", "baseline"))
            self.assertTrue(row["provenance"]["signal"])

    def test_fenced_edit_refused_before_gate(self) -> None:
        """A fenced proposal never spends budget or trains."""
        tmp = tempfile.mkdtemp(prefix="rsi-prop-")
        gate = SelfGate(tmp + "/fence", tmp + "/gate.db",
                        mission="proposer", budget=3)
        calls: list = []
        try:
            out = gate.propose(dict(DEFAULT_CONFIG),
                               {"op": "set", "field": "gate",
                                "value": 1}, _stub(calls))
        finally:
            gate.close()
        self.assertEqual(out["decision"], "refuse")
        self.assertFalse(calls)
        self.assertEqual(out["budget_left"], 3)


if __name__ == "__main__":
    unittest.main()
