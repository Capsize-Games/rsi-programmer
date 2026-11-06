"""Frozen v2 bank separation, seal verification and held refusal."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

from scripts.branch_transfer_v2_probes import BOUNDARY
from scripts.branch_transfer_v2_probes import canonical_probes
from scripts.branch_transfer_v2_probes import extended_probes
from scripts.branch_transfer_v2_probes import oracle_signature
from scripts.branch_transfer_v2_probes import program_signature
from scripts.branch_transfer_v2_probes import training_names
from scripts.branch_transfer_v2_probes import v1_references
from scripts.evaluate_frozen_branch_transfer_v2 import expected_reports
from scripts.evaluate_frozen_branch_transfer_v2 import main as evaluate
from scripts.run_frozen_branch_transfer_v2 import verify_bank
from scripts.seal_branch_transfer_v2 import seal_bank
from rsi_programmer.statement_curriculum import teachers
from rsi_programmer.statement_execution import execute

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "fixtures/frozen_branch_transfer_v2"
V1 = ROOT / "fixtures/frozen_branch_transfer_v1"


def _bank_tasks() -> list[dict[str, Any]]:
    return json.loads((BANK / "tasks.json").read_text())["tasks"]


def _v1_inputs() -> set[tuple[int, ...]]:
    bank = json.loads((V1 / "tasks.json").read_text())
    seen = {tuple(v) for v in bank["query_inputs"]}
    for task in bank["tasks"]:
        seen.update(tuple(v) for v, _ in task["observed"])
    held = json.loads((V1 / "held.json").read_text())
    return seen | {tuple(v) for v in held}


def _probes() -> list[list[int]]:
    return extended_probes(canonical_probes(training_names()))


class FrozenBranchTransferV2Tests(unittest.TestCase):
    """Validate v2 dataset admission without running any solver."""

    def test_hashes_and_balanced_groups_are_fixed(self) -> None:
        hashes = json.loads((BANK / "sha256.json").read_text())
        for name, expected in hashes.items():
            actual = hashlib.sha256((BANK / name).read_bytes()
                                    ).hexdigest()
            self.assertEqual(actual, expected)
        tasks = _bank_tasks()
        self.assertEqual(len(tasks), 54)
        self.assertEqual(set(Counter(task["generation_group"]
                                     for task in tasks).values()), {6})
        held = json.loads((BANK / "held.json").read_text())
        self.assertEqual(len(held), 20)

    def test_extended_suite_matches_documented_probes(self) -> None:
        probes = _probes()
        data = json.loads((BANK / "tasks.json").read_text())
        exclusion = data["exclusion"]
        self.assertEqual(len(probes), exclusion["total_probes"])
        self.assertEqual(exclusion["canonical_probes"], 24)
        v1_held = json.loads((V1 / "held.json").read_text())
        self.assertEqual(exclusion["v1_held_probes"], len(v1_held))
        canonical = canonical_probes(training_names())
        for values in canonical + v1_held:
            self.assertIn(values, probes)
        self.assertGreater(len(probes), 24)

    def test_functions_exclude_training_dev_and_v1(self) -> None:
        names = training_names()
        probes = _probes()
        training, _ = teachers(names, stateful=True,
                               return_variants=True)
        excluded = {program_signature(p, probes) for p in training}
        excluded.update(oracle_signature(n, probes) for n in names)
        excluded.update(program_signature(p, probes)
                        for p in v1_references())
        for task in _bank_tasks():
            signature = program_signature(tuple(task["reference"]),
                                          probes)
            self.assertNotIn(signature, excluded)
            excluded.add(signature)

    def test_exact_tokens_avoid_v1_and_dev_programs(self) -> None:
        report = json.loads((ROOT / "artifacts/statement-composed-route"
                             "-v1-20261101/report.json").read_text())
        forbidden = set(v1_references())
        forbidden.update(tuple(row["chosen"]["tokens"])
                         for row in report["rows"])
        for task in _bank_tasks():
            self.assertNotIn(tuple(task["reference"]), forbidden)

    def test_visible_held_disjoint_and_seeded_fresh(self) -> None:
        bank = json.loads((BANK / "tasks.json").read_text())
        held = json.loads((BANK / "held.json").read_text())
        visible = {tuple(v) for v in bank["query_inputs"]}
        for task in bank["tasks"]:
            visible.update(tuple(v) for v, _ in task["observed"])
        self.assertFalse(visible & {tuple(v) for v in held})
        boundary = {tuple(v) for v in BOUNDARY}
        seeded = (visible | {tuple(v) for v in held}) - boundary
        self.assertFalse(seeded & _v1_inputs())

    def test_reference_outputs_defined_on_all_inputs(self) -> None:
        bank = json.loads((BANK / "tasks.json").read_text())
        held = json.loads((BANK / "held.json").read_text())
        for task in bank["tasks"]:
            reference = tuple(task["reference"])
            for values, expected in task["observed"]:
                self.assertEqual(execute(reference, values), expected)
            for values in held + bank["query_inputs"]:
                self.assertIs(type(execute(reference, values)), int)

    def test_seal_verification_passes_and_detects_tamper(self) -> None:
        tasks, spec = verify_bank(BANK)
        self.assertEqual(len(tasks["tasks"]), 54)
        self.assertEqual(spec["tasks"], 54)
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "bank"
            shutil.copytree(BANK, copied, ignore=shutil.ignore_patterns(
                "held.json", "cost-accounting.json"))
            verify_bank(copied)
            path = copied / "tasks.json"
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                verify_bank(copied)

    def test_reseal_refuses_to_overwrite(self) -> None:
        with self.assertRaises(ValueError):
            seal_bank(BANK)

    def test_evaluator_refuses_before_held_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["prog", "--method", "classical", "--output", tmp,
                    "--bank", tmp]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(ValueError):
                    evaluate()
        self.assertEqual([str(p) for p in expected_reports([0, 1, 2])],
                         ["classical", "neural/seed-0", "neural/seed-1",
                          "neural/seed-2", "untrained/seed-0",
                          "untrained/seed-1", "untrained/seed-2"])


if __name__ == "__main__":
    unittest.main()
