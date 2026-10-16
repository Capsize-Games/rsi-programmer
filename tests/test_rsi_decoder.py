#!/usr/bin/env python3
"""Tests for the fixed spike-to-edit decoder (#10)."""

from __future__ import annotations

import random
import unittest

import torch

from rsi_programmer.actions import ActionSpace
from rsi_programmer.author import emit_edit
from rsi_programmer.decoder import CONST, INPUT, OP_NAMES
from rsi_programmer.decoder import STRUCT, SpikeDecoder
from rsi_programmer.policy import STEPS
from rsi_programmer.probe import CMP, CONSTS, OPS, SELECT
from rsi_programmer.probe import interpret

REFS: list = ["x", "y", 0, 1]


def hot(size: int, winner: int, level: float = 1.0) -> torch.Tensor:
    """STEPS-long train with one firing unit, rest silent."""
    train = torch.zeros(STEPS, size)
    train[:, winner] = level
    return train


def drive(op: str, pointers: dict[str, int] | None = None,
          value: int | None = None,
          refs: list | None = None) -> dict[str, torch.Tensor]:
    """Spike dict voting for one exact edit."""
    width = len(REFS if refs is None else refs)
    spikes = {"struct": hot(len(STRUCT), 0),
              "op": hot(len(OP_NAMES), OP_NAMES.index(op))}
    for name in ("cond", "left", "right"):
        idx = (pointers or {}).get(name, 0)
        spikes[name] = hot(width, idx)
    spikes["value"] = hot(len(CONSTS), 0 if value is None
                           else list(CONSTS).index(value))
    return spikes


class SpikeDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decoder = SpikeDecoder()

    def test_round_trip_every_op_and_terminal(self) -> None:
        """Hand-built spikes decode to each exact alphabet action."""
        for op in list(OPS) + list(CMP):
            action, rec = self.decoder.decode(
                drive(op, {"left": 0, "right": 1}), REFS)
            self.assertEqual(rec["verdict"], "")
            self.assertEqual(action, {"kind": "apply", "op": op,
                                      "left": "x", "right": "y"})
        action, rec = self.decoder.decode(
            drive(SELECT, {"cond": 2, "left": 1, "right": 0},
                  refs=["x", "y", 0]),
            ["x", "y", 0], ["Int", "Int", "Bool"])
        self.assertEqual(rec["verdict"], "")
        self.assertEqual(action, {"kind": "apply", "op": SELECT,
                                  "cond": 0, "left": "y",
                                  "right": "x"})
        action, rec = self.decoder.decode(
            drive(SELECT, {"cond": 0, "left": 1, "right": 0}),
            REFS)
        self.assertIsNone(action)
        self.assertEqual(rec["verdict"], "type-mismatch")
        action, _ = self.decoder.decode(drive(INPUT, {"left": 1}),
                                        REFS)
        self.assertEqual(action, {"kind": "input", "ref": "y"})
        action, _ = self.decoder.decode(drive(CONST, value=3),
                                        REFS)
        self.assertEqual(action, {"kind": "const", "value": 3})

    def test_tie_and_threshold_reject(self) -> None:
        """Ambiguous or quiet populations never guess."""
        tied = drive("add", {"left": 0, "right": 1})
        tied["op"][:, 0] = 1.0
        tied["op"][:, 1] = 1.0
        action, rec = self.decoder.decode(tied, REFS)
        self.assertIsNone(action)
        self.assertEqual(rec["verdict"], "tie")
        quiet = drive("add", {"left": 0, "right": 1})
        quiet["op"] = torch.zeros(STEPS, len(OP_NAMES))
        action, rec = self.decoder.decode(quiet, REFS)
        self.assertIsNone(action)
        self.assertEqual(rec["verdict"], "sub-threshold")
        stopped = drive("add", {"left": 0, "right": 1})
        stopped["struct"] = hot(len(STRUCT), 1)
        action, rec = self.decoder.decode(stopped, REFS)
        self.assertIsNone(action)
        self.assertEqual(rec["verdict"], "stop")

    def test_malformed_populations_reject(self) -> None:
        """Wrong shapes and missing populations are unknown."""
        action, rec = self.decoder.decode({}, REFS)
        self.assertIsNone(action)
        action, rec = self.decoder.decode(
            {"struct": hot(len(STRUCT), 0),
             "op": torch.zeros(STEPS, 2)}, REFS)
        self.assertIsNone(action)
        self.assertEqual(rec["verdict"], "unknown-token")

    def test_random_spikes_validate_or_reject(self) -> None:
        """Fuzz: every output typechecks or carries a reason."""
        space = ActionSpace()
        rng = random.Random(0)
        graph = {"input_types": {"x": "Int", "y": "Int"},
                 "nodes": [{"op": "const", "type": "Int",
                            "args": [], "value": 3},
                           {"op": "add", "type": "Int",
                            "args": ["x", 0]}],
                 "output": 1}
        for _ in range(200):
            sizes = {"struct": len(STRUCT), "op": len(OP_NAMES),
                     "cond": len(REFS), "left": len(REFS),
                     "right": len(REFS), "value": len(CONSTS)}
            spikes = {k: torch.tensor(
                [[rng.choice((0.0, 0.0, 1.0)) for _ in range(n)]
                 for _ in range(STEPS)]) for k, n in sizes.items()}
            action, rec = self.decoder.decode(spikes, REFS)
            if action is None:
                self.assertTrue(rec["verdict"])
            else:
                space.typecheck(graph, action)

    def test_decoded_edits_author_valid_program(self) -> None:
        """Decoded const+add commits graph for x+3 via emit_edit."""
        space = ActionSpace()
        graph: dict = {"input_types": {"x": "Int", "y": "Int"},
                       "nodes": [], "output": "x"}
        const, _ = self.decoder.decode(drive(CONST, value=3),
                                       REFS)
        graph, rec, ok = emit_edit(space, graph, const, "t",
                                   (3,) * 6)
        self.assertTrue(ok)
        refs = ["x", "y", 0]
        add, _ = self.decoder.decode(
            drive("add", {"left": 0, "right": 2}, refs=refs), refs)
        graph, rec, ok = emit_edit(space, graph, add, "t",
                                   (0, 1, 2, 3, 4, 5))
        self.assertTrue(ok)
        self.assertEqual(interpret(graph, 4, 9), 7)


if __name__ == "__main__":
    unittest.main()
