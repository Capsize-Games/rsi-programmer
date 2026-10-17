#!/usr/bin/env python3
"""Tests for stack bytecode lowering and VM oracle (#32)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.bytecode import assemble, binary_holds
from rsi_programmer.bytecode import disassemble, run_vm
from rsi_programmer.probe import CONSTS, enumerate_programs
from rsi_programmer.probe import lower_to_dag

ROOT = Path(__file__).resolve().parent.parent
FILES = ("fixtures/rsi_shadow_tasks.json",
         "fixtures/rsi_ood_twin.json",
         "fixtures/rsi_ood_plain.json",
         "fixtures/rsi_novel_tasks.json",
         "fixtures/rsi_ood_tasks.json")
SOLVED = {"diff_squares", "twice_product", "square_plus",
          "square_minus", "gen_ood_u_00", "gen_ood_u_01",
          "gen_ood_u_02", "gen_ood_u_03", "gen_ood_u_04",
          "gen_ood_u_05", "gen_ood_u_06", "gen_ood_u_07",
          "gen_ood_p_00", "gen_ood_p_01", "gen_ood_p_02",
          "gen_ood_p_03", "gen_ood_p_04", "gen_ood_p_05",
          "gen_ood_p_06", "gen_ood_p_07", "prod_plus_sum",
          "sum_squares", "scaled_mix", "gen_ood_05",
          "gen_ood_07"}


def _reference(task: dict[str, Any]) -> bytes:
    """Assembled binary for one solved task's reference."""
    rows = tuple((c[0], c[1]) for c in task["train"] +
                task["held_out"])
    target = tuple(c[2] for c in task["train"] +
                  task["held_out"])
    feats = list(task.get("features", []))
    expr, _, _ = enumerate_programs(
        rows, target, int(task.get("max_tree_ops", 4)),
        consts=CONSTS if "const" in feats else (),
        branch="branch" in feats)
    assert expr is not None, task["id"]
    return assemble(lower_to_dag(expr))


class BytecodeTests(unittest.TestCase):
    def test_known_program_byte_exact(self) -> None:
        """x + y assembles to five exact bytes."""
        graph = {"input_types": {"x": "Int", "y": "Int"},
                 "nodes": [{"op": "add", "type": "Int",
                            "args": ["x", "y"]}],
                 "output": 0}
        code = assemble(graph)
        self.assertEqual(code, bytes([0x01, 0x02, 0x06,
                                      0x05, 0x00, 0x04,
                                      0x00, 0x00]))
        self.assertEqual(run_vm(code, 3, 4), 7)
        self.assertEqual(run_vm(code, -3, 4), 1)

    def test_disassemble_round_trips(self) -> None:
        """Decode names every instruction and operand."""
        prog = disassemble(bytes([0x01, 0x02, 0x06, 0x05,
                                  0x00, 0x04, 0x00, 0x00]))
        self.assertEqual(prog[0], ("PUSH_X",))
        self.assertEqual(prog[2], ("ADD",))
        self.assertEqual(prog[-1], ("HALT",))

    def test_decoder_rejects_junk(self) -> None:
        """Bad opcodes, truncations, and missing HALT fail."""
        with self.assertRaises(ValueError):
            disassemble(bytes([0xFF, 0x00]))
        with self.assertRaises(ValueError):
            disassemble(bytes([0x03, 0x01]))
        with self.assertRaises(ValueError):
            disassemble(bytes([0x01, 0x02, 0x06]))
        with self.assertRaises(ValueError):
            disassemble("not-bytes")  # type: ignore[arg-type]

    def test_all_solved_tasks_verify_binary(self) -> None:
        """Every solved task's binary holds train + held-out."""
        count, sizes = 0, []
        biggest = 0
        for name in FILES:
            with open(ROOT / name, encoding="utf-8") as handle:
                tasks = json.load(handle)["tasks"]
            for task in tasks:
                if task["id"] not in SOLVED:
                    continue
                code = _reference(task)
                self.assertTrue(binary_holds(task, code),
                                task["id"])
                for case in task["train"] + task["held_out"]:
                    biggest = max(biggest, abs(case[2]))
                    for value in case[:2]:
                        biggest = max(biggest, abs(value))
                count += 1
                sizes.append((task["id"], len(code)))
        print("binaries=%d biggest-value=%d" % (count,
                                                biggest))
        for tid, size in sorted(sizes):
            print("  %-14s %d bytes" % (tid, size))
        self.assertEqual(count, 25)
        self.assertLess(biggest, 2 ** 15)

    def test_corrupted_binaries_fail_oracle(self) -> None:
        """Truncated, foreign, and behavior-flipped fail."""
        with open(ROOT / FILES[0], encoding="utf-8") as handle:
            tasks = json.load(handle)["tasks"]
        task = next(t for t in tasks
                    if t["id"] == "square_plus")
        code = _reference(task)
        self.assertTrue(binary_holds(task, code))
        self.assertFalse(binary_holds(task, code[:-1]))
        bad = bytes([0xFF]) + code[1:]
        self.assertFalse(binary_holds(task, bad))
        flipped = bytes(b ^ 0x01 if b in (0x06, 0x07)
                        else b for b in code)
        self.assertNotEqual(flipped, code)
        self.assertFalse(binary_holds(task, flipped))


if __name__ == "__main__":
    unittest.main()
