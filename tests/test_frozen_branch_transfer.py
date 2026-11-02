"""Frozen function separation, input boundaries and classical startup."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from rsi_programmer.statement_curriculum import oracle, teachers
from rsi_programmer.statement_execution import execute

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "fixtures/frozen_branch_transfer_v1"


class FrozenBranchTransferTests(unittest.TestCase):
    """Validate dataset admission without running or selecting a solver."""

    def test_hashes_and_balanced_generation_groups_are_fixed(self) -> None:
        hashes = json.loads((BANK / "sha256.json").read_text())
        for name, expected in hashes.items():
            self.assertEqual(hashlib.sha256((BANK / name).read_bytes()
                                           ).hexdigest(), expected)
        tasks = json.loads((BANK / "tasks.json").read_text())["tasks"]
        self.assertEqual(len(tasks), 18)
        self.assertEqual(set(Counter(task["generation_group"]
                                     for task in tasks).values()), {2})

    def test_functions_exclude_both_training_recipes_and_development(self
                                                                  ) -> None:
        old = json.loads(
            (ROOT / "fixtures/statement_tasks_v1.json").read_text())
        new = json.loads(
            (ROOT / "fixtures/stateful_tasks_v1.json").read_text())
        names = [task["id"] for task in old["tasks"] + new["tasks"]]
        programs, probes = teachers(names, stateful=True, return_variants=True)
        known = {tuple(execute(program, values) for values in probes)
                 for program in programs}
        previous, previous_probes = teachers(names[:6])
        self.assertEqual(probes, previous_probes)
        self.assertTrue({tuple(execute(program, v) for v in probes)
                         for program in previous} <= known)
        known.update(tuple(oracle(name, values) for values in probes)
                     for name in names)
        tasks = json.loads((BANK / "tasks.json").read_text())["tasks"]
        for task in tasks:
            signature = tuple(execute(tuple(task["reference"]), values)
                              for values in probes)
            self.assertNotIn(signature, known)
            known.add(signature)

    def test_reference_outputs_are_defined_and_visible_inputs_are_disjoint(
            self) -> None:
        bank = json.loads((BANK / "tasks.json").read_text())
        held = json.loads((BANK / "held.json").read_text())
        visible = {tuple(v) for v in bank["query_inputs"]}
        for task in bank["tasks"]:
            reference = tuple(task["reference"])
            for values, expected in task["observed"]:
                visible.add(tuple(values))
                self.assertEqual(execute(reference, values), expected)
            for values in held + bank["query_inputs"]:
                self.assertIs(type(execute(reference, values)), int)
        self.assertFalse(visible & {tuple(v) for v in held})

    def test_classical_search_import_has_no_neural_runtime(self) -> None:
        result = subprocess.run([sys.executable, "-c",
            "import sys; import rsi_programmer.statement_branch_search; "
            "print('torch' in sys.modules)"], cwd=ROOT, capture_output=True,
            text=True, check=True)
        self.assertEqual(result.stdout.strip(), "False")
