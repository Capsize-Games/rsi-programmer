"""Minimal-size hard bank: minimality, exclusion, refusal."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.benchmark_v2_corpus import (
    OPERATORS,
    Node,
    Polynomial,
    apply,
    leaf,
    polynomial,
)
from rsi_programmer.construction_check import check_graph
from rsi_programmer.construction_hard_corpus import (
    BOUND,
    EARLY_STOP_MULTIPLE,
    STRATA,
    catalog_levels,
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
    """Collect every task polynomial identity in a corpus."""
    found: set[Any] = set()
    for tasks in corpus["splits"].values():
        for task in tasks:
            poly = task["generator"]["polynomial"]
            found.add(tuple(tuple(term) for term in poly))
    return found


def _grow_reference(levels: dict[int, list[Node]],
                    seen: set[Polynomial], size: int) -> list[Node]:
    """Enumerate one reference level without the module builder."""
    current: list[Node] = []
    for left_size in range(size):
        for left in levels[left_size]:
            for right in levels[size - left_size - 1]:
                for op in OPERATORS:
                    node = apply(op, left, right)
                    identity = polynomial(node)
                    if identity not in seen:
                        seen.add(identity)
                        current.append(node)
    return current


def _reference_below() -> dict[int, set[Polynomial]]:
    """Recompute level 0..5 identities independently of the module."""
    levels: dict[int, list[Node]] = {0: [leaf("x"), leaf("y")]}
    seen: set[Polynomial] = {polynomial(node) for node in levels[0]}
    below = {0: set(seen)}
    for size in range(1, 6):
        levels[size] = _grow_reference(levels, seen, size)
        below[size] = {polynomial(node) for node in levels[size]}
    return below


class ConstructionHardCorpusTests(unittest.TestCase):
    """Validate the minimal-size bank before any fitting."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.v1 = load_v1_identities(V1)
        cls.levels, _ = catalog_levels()
        cls.reference = _reference_below()
        cls.corpus: dict[str, Any] = generate_corpus(
            seed=7, quotas=dict(TINY), v1_path=V1)

    def test_schema_and_strata_even_counts(self) -> None:
        self.assertEqual(set(self.corpus["splits"]), {"train", "dev", "eval"})
        self.assertEqual(self.corpus["seed"], 7)
        for name, total in TINY.items():
            tasks = self.corpus["splits"][name]
            self.assertEqual(len(tasks), total)
            counts = {ops: 0 for ops in STRATA}
            for task in tasks:
                self.assertEqual(task["split"], name)
                self.assertEqual(task["max_tree_ops"], BOUND)
                counts[task["generator"]["minimal_tree_ops"]] += 1
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
                self.assertIn(generator["minimal_tree_ops"], set(STRATA))
                self.assertNotIn("sampled_ops", generator)
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

    def test_minimal_ops_match_first_appearance(self) -> None:
        below = dict(self.reference)
        below[6] = {polynomial(node) for node in self.levels[6]}
        below[7] = {polynomial(node) for node in self.levels[7]}
        for tasks in self.corpus["splits"].values():
            for task in tasks:
                poly = task["generator"]["polynomial"]
                identity = tuple(tuple(term) for term in poly)
                claimed = task["generator"]["minimal_tree_ops"]
                for size in range(claimed):
                    self.assertNotIn(identity, below[size], task["id"])

    def test_cross_stratum_identities_never_repeat_at_scale(
            self) -> None:
        # The first sampled generation repeated identities across
        # strata; the level catalog keeps each identity at one size.
        corpus = generate_corpus(
            seed=20261106,
            quotas={"train": 300, "dev": 30, "eval": 30},
            v1_path=V1)
        identities = _identities(corpus)
        total = sum(len(tasks) for tasks in corpus["splits"].values())
        self.assertEqual(len(identities), total)
        self.assertFalse(identities & self.v1)
        strata = [task["generator"]["minimal_tree_ops"]
                  for tasks in corpus["splits"].values()
                  for task in tasks]
        for ops, want in ((6, 120), (7, 120), (8, 120)):
            self.assertEqual(strata.count(ops), want)
        ids = [task["id"] for tasks in corpus["splits"].values()
               for task in tasks]
        self.assertEqual(len(set(ids)), total)

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

    def test_level_8_collection_early_stops(self) -> None:
        record = self.corpus["level_8_collection"]
        sizes = self.corpus["catalog_level_sizes"]
        self.assertEqual(record["multiple"], EARLY_STOP_MULTIPLE)
        self.assertTrue(record["early_stop"])
        self.assertEqual(record["collected"],
                         EARLY_STOP_MULTIPLE * record["need"])
        self.assertGreaterEqual(record["examined"], record["collected"])
        self.assertEqual(sizes["8"], record["collected"])
        self.assertEqual(set(sizes), {str(n) for n in range(9)})
        self.assertEqual(sizes["0"], 2)
        for size in range(8):
            self.assertEqual(sizes[str(size)], len(self.levels[size]))

    def test_tiny_corpus_excludes_real_v1_identities(self) -> None:
        self.assertTrue(self.v1)
        self.assertFalse(_identities(self.corpus) & self.v1)

    def test_protocol_fixture_freezes_arms_and_budgets(self) -> None:
        path = ROOT / "fixtures" / "construction_hard_v1.json"
        protocol = json.loads(path.read_text())
        self.assertEqual(protocol["quotas"],
                         {"train": 1200, "dev": 200, "eval": 300})
        self.assertEqual(protocol["strata"], [6, 7, 8])
        self.assertEqual(protocol["grammar"]["max_tree_ops"], 8)
        evaluation = protocol["evaluation"]
        self.assertEqual(evaluation["model_seeds"], [0, 1, 2])
        self.assertEqual(evaluation["beam_width"], 8)
        self.assertEqual(evaluation["operator_top_k"], 2)
        self.assertEqual(evaluation["per_task_cpu_seconds"], 0.5)
        self.assertEqual(evaluation["frozen_at"], "2026-11-06")
        self.assertEqual(evaluation["imitation_epochs"], 100)
        self.assertEqual(evaluation["feedback"]["rounds"], 4)
        self.assertEqual(evaluation["feedback"]["epochs_per_round"], 30)
        self.assertEqual(evaluation["feedback"]["rollout_budget"], 256)
        self.assertEqual(evaluation["bound"], 8)

    def test_protocol_kill_criterion_matches_design_doc(self) -> None:
        path = ROOT / "fixtures" / "construction_hard_v1.json"
        protocol = json.loads(path.read_text())
        self.assertEqual(
            protocol["decision_rules"]["kill_criterion"], KILL)
        design = ROOT / "tracks" / "programmer"
        text = (design / "feedback-construction-design.md").read_text()
        self.assertIn(KILL, " ".join(text.split()))


if __name__ == "__main__":
    raise SystemExit(unittest.main())
