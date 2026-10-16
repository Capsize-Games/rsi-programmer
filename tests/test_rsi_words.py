#!/usr/bin/env python3
"""Tests for the bit-slice word codec and oracle (lattice v30)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.probe import CONSTS, enumerate_programs
from rsi_programmer.probe import validate_graph
from rsi_programmer.words import ISA, OPCODES, decode
from rsi_programmer.words import oracle_holds_words, pack, unpack

FIXTURE = str(Path(__file__).resolve().parent.parent
              / "fixtures" / "rsi_shadow_tasks.json")


def _encode(term: Any, words: list[int],
            index: dict[int, int], count: list[int]) -> None:
    """Append opcode + ref-words for one Expr tree (test only)."""
    if isinstance(term, str) or term.op == "input":
        return
    if term.op == "const":
        words += [OPCODES["const"], CONSTS.index(term.left)]
        index[id(term)] = count[0]
        count[0] += 1
        return
    kids = [term.left, term.right] + ([term.third]
                                      if term.op == "select" else [])
    for kid in kids:
        _encode(kid, words, index, count)
    ordered: list[int] = []
    for kid in kids:
        if isinstance(kid, str):
            ordered.append(0 if kid == "x" else 1)
        elif kid.op == "input":
            ordered.append(0 if kid.left == "x" else 1)
        else:
            ordered.append(2 + index[id(kid)])
    words.append(OPCODES[term.op])
    words += ordered
    index[id(term)] = count[0]
    count[0] += 1


def _words_for(task: dict[str, Any]) -> list[int]:
    """Word program for a fixture task via the enumerator."""
    rows = tuple((c[0], c[1]) for c in task["train"])
    target = tuple(c[2] for c in task["train"])
    bound = int(task.get("max_tree_ops", 4))
    expr, _, _ = enumerate_programs(rows, target, bound)
    assert expr is not None
    words: list[int] = []
    _encode(expr, words, {}, [0])
    return words + [OPCODES["stop"]]


class CodecTests(unittest.TestCase):
    def test_pack_unpack_round_trip_all_words(self) -> None:
        """All 16 codepoints pack and unpack losslessly."""
        for word in range(16):
            bits = unpack(word)
            self.assertEqual(len(bits), 4)
            self.assertTrue(all(b in (0, 1) for b in bits))
            self.assertEqual(pack(bits), word)

    def test_pack_rejects_bad_bits(self) -> None:
        """Wrong length, non-binary, or range fail loudly."""
        for bad in ([0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 2]):
            with self.assertRaises(ValueError):
                pack(bad)
        for bad_word in (-1, 16, "3"):
            with self.assertRaises(ValueError):
                unpack(bad_word)

    def test_isa_covers_sixteen_with_stop(self) -> None:
        """ISA v0 names every codepoint 0-8; rest reserved."""
        self.assertEqual(len(ISA), 9)
        self.assertEqual(ISA[OPCODES["stop"]], "stop")
        for word in range(9, 16):
            with self.assertRaises(ValueError):
                decode([word, OPCODES["stop"]])

    def test_decode_each_family_validates(self) -> None:
        """One word program per op family passes validation."""
        progs = [[0, 0, 1, 8], [1, 0, 1, 8], [2, 0, 1, 8],
                 [3, 0, 1, 5, 2, 0, 1, 8],
                 [4, 0, 1, 5, 2, 0, 1, 8],
                 [6, 0, 0, 0, 1, 8], [7, 3, 8]]
        for prog in progs:
            validate_graph(decode(prog))

    def test_decode_rejects_truncated_missing_or_typeless(
            self) -> None:
        """Truncated, stopless, and outputless fail loudly."""
        with self.assertRaises(ValueError):
            decode([0, 0])
        with self.assertRaises(ValueError):
            decode([0, 0, 1])
        with self.assertRaises(ValueError):
            decode([6, 0, 8])
        with self.assertRaises(ValueError):
            decode([3, 0, 1, 8])

    def test_decode_rejects_type_mismatch(self) -> None:
        """Bool node as Int operand fails validation."""
        with self.assertRaises(ValueError):
            decode([3, 0, 1, 0, 2, 0, 8])


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Load the frozen shadow family once."""
        with open(FIXTURE, encoding="utf-8") as handle:
            cls.tasks = json.load(handle)["tasks"]

    def test_word_programs_hold_on_all_shadow(self) -> None:
        """Encoded solutions verify train + held-out."""
        for task in self.tasks:
            words = _words_for(task)
            self.assertTrue(oracle_holds_words(task, words),
                            task["id"])

    def test_corrupted_words_fail_oracle(self) -> None:
        """Truncated and reserved-token edits do not verify."""
        for task in self.tasks:
            words = _words_for(task)
            self.assertFalse(oracle_holds_words(task,
                                               words[:-1]),
                             task["id"])
            bad = [9] + words[1:]
            self.assertFalse(oracle_holds_words(task, bad),
                             task["id"])

    def test_fixtures_load_unchanged(self) -> None:
        """Codec work writes no fixtures; all still parse."""
        for name in ("rsi_shadow_tasks.json",
                     "rsi_ood_plain.json",
                     "rsi_ood_twin.json"):
            path = Path(FIXTURE).parent / name
            with open(path, encoding="utf-8") as handle:
                fam = json.load(handle)
            self.assertIn("tasks", fam)


if __name__ == "__main__":
    unittest.main()
