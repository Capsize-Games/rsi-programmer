"""Contract tests for the frozen bounded-list edit-value evaluation."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.check_edit_value import BANK, verify, verify_pre_fit
from scripts.measure_edit_value import (
    BASELINE, COMPILER, CURLEE_GUARD, _corpus, _fresh_signatures,
    _source_hashes,
)


class EditValueEvaluationTests(unittest.TestCase):
    def test_frozen_bank_and_checker_references_are_consistent(self) -> None:
        result = verify()
        protocol_sha = (
            "2c7dc36c284df6edfcafa44b6d4e3aeeeeaa67413c0d889fde0304beac441a1b")
        examples_sha = (
            "592593be8df66a61958a8f89afd3432902520ee3d7f65bf202e0b62cee668ffa")
        held_sha = (
            "02ebde24e29044e320aac2a25e05e64331e473d928538f2b1a6e2badf9cff1e5")
        self.assertEqual(result["fresh_tasks"], 6)
        self.assertEqual(result["fresh_signatures_distinct"], 6)
        self.assertEqual(result["held_cases_per_task"], 20)
        self.assertEqual(result["protocol_sha256"], protocol_sha)
        self.assertEqual(result["examples_sha256"], examples_sha)
        self.assertEqual(result["held_cases_sha256"], held_sha)

    def test_pre_fit_contract_does_not_open_held_rows(self) -> None:
        from unittest.mock import patch

        original = Path.read_text

        def guarded(path: Path, *args: object, **kwargs: object) -> str:
            if path.name == "held_cases.json":
                raise AssertionError("held rows were opened before choices")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", guarded):
            self.assertEqual(verify_pre_fit()["fresh_tasks"], 6)

    def test_training_pool_excludes_fresh_signatures_and_pins_sources(
            self) -> None:
        protocol = json.loads((BANK / "protocol.json").read_text())
        from rsi_programmer.array_curriculum import probes

        excluded = _fresh_signatures(protocol, probes())
        corpus = _corpus(excluded, 64)
        signatures = {str(row["signature"]) for row in corpus}
        self.assertTrue(corpus)
        self.assertFalse(signatures & excluded)
        source_hashes = _source_hashes()
        self.assertIn("rsi_programmer/edit_value_ranker.py", source_hashes)
        self.assertIn("baseline_checkpoint", source_hashes)
        self.assertIn("curlee_compiler", source_hashes)
        self.assertIn("curlee_resource_guard", source_hashes)
        self.assertTrue(BASELINE.is_file())
        self.assertTrue(COMPILER.is_file())
        self.assertTrue(CURLEE_GUARD.is_file())

    def test_fresh_structures_are_not_claimed_as_new_families(self) -> None:
        protocol = json.loads((BANK / "protocol.json").read_text())
        self.assertEqual(protocol["development_tasks"], [
            "clip_negative_readings", "cap_positive_readings",
            "negative_magnitudes"])
        self.assertEqual(protocol["baseline_recheck_tasks"], [
            "clip_negative_readings", "cap_positive_readings",
            "negative_magnitudes"])
        self.assertEqual(protocol["retention_tasks"], [])
        self.assertIn("not_measured_for_edit_value_ranker",
                      protocol["retention_status"])
        self.assertTrue(all(task["task_family"] == "bounded_integer_list"
                            for task in protocol["fresh_tasks"]))
        self.assertEqual(len({task["task_type"]
                              for task in protocol["fresh_tasks"]}), 6)
        self.assertIn("Strings", protocol["claims"][2])

    def test_solver_visible_examples_are_separate_from_held_rows(self) -> None:
        examples = json.loads(
            (BANK / "fresh_examples.json").read_text())
        held = json.loads((BANK / "held_cases.json").read_text())
        self.assertEqual({row["task_id"] for row in examples},
                         {row["task_id"] for row in held})
        self.assertTrue(all(set(row) == {
            "task_id", "task_family", "task_type", "examples"}
            for row in examples))
        self.assertTrue(all(set(row) == {
            "task_id", "held_inputs", "expected"} for row in held))


if __name__ == "__main__":
    unittest.main()
