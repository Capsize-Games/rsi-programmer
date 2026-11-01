"""Fragment scope, source provenance and a strong classical control."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_fragment_classical import synthesize
from rsi_programmer.statement_fragment_library import qualify
from rsi_programmer.statement_fragments import mine
from rsi_programmer.statement_syntax import parse

ROOT = Path(__file__).resolve().parents[1]
DONORS = ROOT / "artifacts/statement-programs-v1-execution-20261101"


class StatementFragmentTests(unittest.TestCase):
    """Promote only code supported by unchanged independent evidence."""

    def test_verified_programs_induce_contexts_with_provenance(self) -> None:
        library = qualify(DONORS)
        self.assertEqual(len(library["donors"]), 4)
        self.assertEqual(len(library["fragments"]), 3)
        counting = next(fragment for fragment in library["fragments"]
                        if fragment["donors"] == ["odd_count"])
        tokens = (*counting["prefix"], "lt", "value", "zero",
                  *counting["suffix"])
        self.assertEqual(execute(tokens, [-3, 2, 0, -1]), 2)

    def test_nested_condition_holes_preserve_scoped_syntax(self) -> None:
        tokens = tuple("for if gt value zero if eq rem value two zero "
                       "set add acc value end end end return acc stop".split())
        fragments = mine([("nested", tokens)])
        self.assertEqual(len(fragments), 2)
        for fragment in fragments:
            parse((*fragment["prefix"], "lt", "value", "zero",
                   *fragment["suffix"]))

    def test_changed_donor_source_is_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "copy"
            shutil.copytree(DONORS, root)
            path = root / "odd_count/python/clean_readings.py"
            path.write_text(path.read_text()
                            + "\n# Changed after validation.\n")
            with self.assertRaises(ValueError):
                qualify(root)

    def test_truthy_acceptance_flags_cannot_hide_a_wrong_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "copy"
            shutil.copytree(DONORS, root)
            path = root / "report.json"
            data = json.loads(path.read_text())
            row = next(row for row in data["rows"] if row["id"] == "odd_count")
            row["python_records"][0]["actual"] = 999
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                qualify(root)

    def test_classical_control_fills_the_same_typed_hole(self) -> None:
        library = qualify(DONORS)
        # Zero after a negative distinguishes state-dependent predicates.
        result = synthesize([([-1, 0, 2], 1), ([1, 2, 3], 0), ([-2, -1], 2)],
                            library["fragments"], budget_seconds=0.5)
        self.assertIsNotNone(result["tokens"])
        self.assertEqual(execute(result["tokens"], [-4, 0, 3, -2]), 2)
        with self.assertRaises(ValueError):
            synthesize([([], 0)], library["fragments"], budget_seconds=0)

    def test_no_library_returns_no_program_without_enumerating(self) -> None:
        result = synthesize([([], 0)], [], budget_seconds=0.5)
        self.assertIsNone(result["tokens"])
        self.assertEqual(result["visited"], 0)

    def test_empty_cases_cannot_qualify_a_donor(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "copy"
            shutil.copytree(DONORS, root)
            (root / "held-inputs.json").write_text("[]")
            for name, key in (("report.json", "python_records"),
                              ("curlee-execution.json", "records")):
                path = root / name
                data = json.loads(path.read_text())
                for row in data["rows"]:
                    row[key] = []
                path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                qualify(root)
