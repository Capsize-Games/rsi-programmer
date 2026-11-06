"""Random-tree hard bank: schema, exclusion, determinism, refusal."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.construction_check import check_graph
from rsi_programmer.construction_hard_corpus import (
    STRATA,
    generate_corpus,
    load_v1_identities,
)

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "fixtures" / "construction_corpus_v1.json"
TINY = {"train": 6, "dev": 3, "eval": 3}
KILL = ("if the feedback-trained arm does not beat **both** "
        "untrained and imitation arms on solves, **or** does not "
        "beat streaming at equal total CPU (fit and rollout "
        "generation charged, amortized over the actual evaluation "
        "volume), the ANN-construction hypothesis at accessible "
        "scale is closed \u2014 not tuned further.")


def _identities(corpus: dict[str, Any]) -> set[Any]:
    found: set[Any] = set()
    for tasks in corpus["splits"].values():
        for task in tasks:
            poly = task["generator"]["polynomial"]
            found.add(tuple(tuple(term) for term in poly))
    return found


class ConstructionHardCorpusTests(unittest.TestCase):
    """Validate the strata-even random-tree bank before any fitting."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.v1 = load_v1_identities(V1)
        cls.corpus: dict[str, Any] = generate_corpus(
            seed=7, quotas=dict(TINY), v1_path=V1)

    def test_schema_and_strata_even_counts(self) -> None:
        self.assertEqual(set(self.corpus["splits"]),
                         {"train", "dev", "eval"})
        self.assertEqual(self.corpus["seed"], 7)
        for name, total in TINY.items():
            tasks = self.corpus["splits"][name]
            self.assertEqual(len(tasks), total)
            counts = {ops: 0 for ops in STRATA}
            for task in tasks:
                self.assertEqual(task["split"], name)
                self.assertEqual(task["max_tree_ops"], 7)
                counts[task["generator"]["sampled_ops"]] += 1
            for ops in STRATA:
                self.assertEqual(counts[ops], total // 3)

    def test_row_geometry_and_generator_shape(self) -> None:
        for tasks in self.corpus["splits"].values():
            for task in tasks:
                self.assertEqual(len(task["observed"]), 36)
                self.assertEqual(len(task["held"]), 24)
                seen = {(x, y) for x, y, _ in task["observed"]}
                held = {(x, y) for x, y, _ in task["held"]}
                self.assertFalse(seen & held)
                generator = task["generator"]
                self.assertIn(generator["sampled_ops"], set(STRATA))
                self.assertIn("program", generator)
                self.assertIn("polynomial", generator)

    def test_every_witness_passes_independent_acceptance(self) -> None:
        for tasks in self.corpus["splits"].values():
            for task in tasks:
                observed = tuple(tuple(row) for row in task["observed"])
                held = tuple(tuple(row) for row in task["held"])
                poly = task["generator"]["polynomial"]
                identity = tuple(tuple(term) for term in poly)
                result = check_graph(task["generator"]["program"],
                                     observed, held, identity,
                                     task["max_tree_ops"])
                self.assertTrue(result["accepted"], task["id"])

    def test_splits_are_function_and_id_disjoint(self) -> None:
        seen: set[Any] = set()
        identifiers: set[str] = set()
        for tasks in self.corpus["splits"].values():
            for task in tasks:
                poly = task["generator"]["polynomial"]
                identity = tuple(tuple(term) for term in poly)
                self.assertNotIn(identity, seen)
                self.assertNotIn(task["id"], identifiers)
                seen.add(identity)
                identifiers.add(task["id"])

    def test_planted_v1_polynomials_are_excluded(self) -> None:
        first = generate_corpus(seed=11, quotas=dict(TINY),
                                v1_path=None)
        planted = _identities(first)
        self.assertTrue(planted)
        rows = [{"generator": {"polynomial": [list(t) for t in p]}}
                for p in planted]
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake_v1.json"
            fake.write_text(json.dumps({"splits": {"train": rows,
                                                   "dev": [],
                                                   "eval": []}}))
            second = generate_corpus(seed=11, quotas=dict(TINY),
                                     v1_path=fake)
        self.assertFalse(_identities(second) & planted)
        for name, total in TINY.items():
            self.assertEqual(len(second["splits"][name]), total)

    def test_real_v1_polynomials_never_reappear(self) -> None:
        self.assertTrue(self.v1)
        self.assertFalse(_identities(self.corpus) & self.v1)

    def test_deterministic_byte_identical_regeneration(self) -> None:
        left = generate_corpus(seed=13, quotas=dict(TINY),
                               v1_path=None)
        right = generate_corpus(seed=13, quotas=dict(TINY),
                                v1_path=None)
        self.assertEqual(json.dumps(left, sort_keys=True),
                         json.dumps(right, sort_keys=True))

    def test_existing_output_is_refused(self) -> None:
        script = "scripts.generate_construction_hard_corpus"
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "corpus.json"
            target.write_text("sentinel\n")
            proc = subprocess.run(
                [sys.executable, "-m", script, "--output",
                 str(target), "--train", "6", "--dev", "3",
                 "--eval", "3", "--seed", "5"],
                cwd=ROOT, capture_output=True, text=True,
                timeout=300)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(target.read_text(), "sentinel\n")

    def test_protocol_fixture_freezes_arms_and_budgets(self) -> None:
        path = ROOT / "fixtures" / "construction_hard_v1.json"
        protocol = json.loads(path.read_text())
        self.assertEqual(protocol["quotas"],
                         {"train": 1200, "dev": 200, "eval": 300})
        self.assertEqual(protocol["strata"], [5, 6, 7])
        evaluation = protocol["evaluation"]
        self.assertEqual(evaluation["model_seeds"], [0, 1, 2])
        self.assertEqual(evaluation["beam_width"], 8)
        self.assertEqual(evaluation["operator_top_k"], 2)
        self.assertIsNone(evaluation["per_task_cpu_seconds"])
        self.assertIsNone(evaluation["frozen_at"])
        self.assertEqual(evaluation["imitation_epochs"], 100)
        self.assertEqual(evaluation["feedback"]["rounds"], 4)
        self.assertEqual(evaluation["feedback"]["epochs_per_round"], 30)
        self.assertEqual(evaluation["feedback"]["rollout_budget"], 256)
        self.assertEqual(evaluation["bound"], 7)

    def test_protocol_kill_criterion_matches_design_doc(self) -> None:
        path = ROOT / "fixtures" / "construction_hard_v1.json"
        protocol = json.loads(path.read_text())
        self.assertEqual(
            protocol["decision_rules"]["kill_criterion"], KILL)
        design = ROOT / "tracks" / "programmer"
        text = (design / "feedback-construction-design.md").read_text()
        self.assertIn(KILL, " ".join(text.split()))
