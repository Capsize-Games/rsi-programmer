#!/usr/bin/env python3
"""Tests for the trainer-edit operators (#2)."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from rsi_programmer.edits import DEFAULT_CONFIG, FENCED, FIELDS
from rsi_programmer.edits import all_edits, apply_edit, describe
from rsi_programmer.edits import invert, validate_edit

SOURCE = Path(__file__).resolve().parent.parent \
    / "rsi_programmer" / "edits.py"


class EditOperatorTests(unittest.TestCase):
    def test_every_operator_round_trips(self) -> None:
        """Apply then invert restores the config exactly."""
        for field, values in FIELDS.items():
            for value in values:
                base = dict(DEFAULT_CONFIG)
                edit = {"op": "set", "field": field,
                        "value": value}
                moved = apply_edit(base, edit)
                self.assertEqual(moved[field], value)
                self.assertEqual(base[field],
                                 DEFAULT_CONFIG[field])
                back = invert(edit, base)
                self.assertEqual(apply_edit(moved, back), base)
                self.assertIn(field, describe(edit, base))

    def test_malformed_and_foreign_edits_rejected(self) -> None:
        """Unknown fields, ops, and values all raise."""
        bad = [{"op": "set", "field": "hidden"},
               {"op": "set", "field": "hidden", "value": 32,
                "extra": 1},
               {"op": "exec", "field": "hidden", "value": 32},
               {"op": "set", "field": "gate_gain", "value": 1},
               {"op": "set", "field": "hidden", "value": 7},
               {"op": "set", "field": "hidden", "value": "32"},
               "set-hidden-32"]
        for edit in bad:
            with self.subTest(edit=edit):
                with self.assertRaises(ValueError):
                    validate_edit(edit)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            apply_edit({}, {"op": "set", "field": "hidden",
                            "value": 32})

    def test_gate_modules_unnameable(self) -> None:
        """No operator, field, or import can touch the judge."""
        for field in FIELDS:
            lowered = field.lower()
            for token in FENCED:
                self.assertNotIn(token, lowered)
        tree = ast.parse(SOURCE.read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(str(node.module))
        for name in imports:
            for token in ("promote", "sandbox", "lineage",
                          "validate_graph", "oracle"):
                self.assertNotIn(token, name)

    def test_all_edits_lists_nontrivial_only(self) -> None:
        """Enumeration skips current values and stays valid."""
        edits = all_edits(dict(DEFAULT_CONFIG))
        self.assertTrue(edits)
        for edit in edits:
            validate_edit(edit)
            self.assertNotEqual(edit["value"],
                                DEFAULT_CONFIG[edit["field"]])
        fields = {e["field"] for e in edits}
        self.assertEqual(fields, set(FIELDS))


if __name__ == "__main__":
    unittest.main()
