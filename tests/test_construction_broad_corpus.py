"""Shape-broad scaling bank: quotas, exclusions, shape, order."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.benchmark_v2_corpus import polynomial
from rsi_programmer.construction_broad_corpus import (
    STRATA,
    generate_corpus,
)
from rsi_programmer.construction_hard_corpus import (
    catalog_levels,
    load_v1_identities,
)

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "fixtures" / "construction_corpus_v1.json"
TINY = {"train": 12, "eval": 15}


def _split_polys(tasks: list[dict[str, Any]]) -> set[Any]:
    """Collect polynomial identities from one split task list."""
    return {tuple(tuple(term) for term in task["generator"]["polynomial"])
            for task in tasks}


def _identities(corpus: dict[str, Any]) -> set[Any]:
    """Collect every task polynomial identity in a corpus."""
    found: set[Any] = set()
    for tasks in corpus["splits"].values():
        found |= _split_polys(tasks)
    return found


def _strata(tasks: list[dict[str, Any]]) -> list[int]:
    """Read minimal-tree-op stratum labels in list order."""
    return [int(task["generator"]["minimal_tree_ops"]) for task in tasks]


class ConstructionBroadCorpusTests(unittest.TestCase):
    """Validate the shape-broad scaling bank before any fitting."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus: dict[str, Any] = generate_corpus(
            seed=11, quotas=dict(TINY), excluded=set())

    def test_split_sizes_and_per_stratum_counts(self) -> None:
        self.assertEqual(set(self.corpus["splits"]), {"train", "eval"})
        for name, total in TINY.items():
            tasks = self.corpus["splits"][name]
            self.assertEqual(len(tasks), total)
            counts = {ops: 0 for ops in STRATA}
            for task in tasks:
                self.assertEqual(task["split"], name)
                self.assertEqual(task["max_tree_ops"], 8)
                counts[task["generator"]["minimal_tree_ops"]] += 1
            for ops in STRATA:
                self.assertEqual(counts[ops], total // 3)

    def test_train_eval_polynomials_are_disjoint(self) -> None:
        train = _split_polys(self.corpus["splits"]["train"])
        eval_polys = _split_polys(self.corpus["splits"]["eval"])
        self.assertEqual(len(train), TINY["train"])
        self.assertEqual(len(eval_polys), TINY["eval"])
        self.assertFalse(train & eval_polys)

    def test_planted_excluded_identities_are_honored(self) -> None:
        planted = _identities(self.corpus)
        second = generate_corpus(seed=11, quotas=dict(TINY),
                                 excluded=planted)
        self.assertFalse(_identities(second) & planted)
        for name, total in TINY.items():
            self.assertEqual(len(second["splits"][name]), total)

    def test_task_format_and_row_geometry(self) -> None:
        for tasks in self.corpus["splits"].values():
            for task in tasks:
                self.assertEqual(len(task["observed"]), 36)
                self.assertEqual(len(task["held"]), 24)
                seen = {(x, y) for x, y, _ in task["observed"]}
                held = {(x, y) for x, y, _ in task["held"]}
                self.assertFalse(seen & held)
                generator = task["generator"]
                self.assertIn(generator["minimal_tree_ops"], set(STRATA))
                self.assertIn("program", generator)
                self.assertIn("polynomial", generator)
                self.assertEqual(len(task["id"]), 16)
                int(task["id"], 16)

    def test_minimal_ops_match_first_appearance(self) -> None:
        levels, _ = catalog_levels()
        below = {size: {polynomial(node) for node in levels[size]}
                 for size in range(8)}
        for tasks in self.corpus["splits"].values():
            for task in tasks:
                poly = task["generator"]["polynomial"]
                identity = tuple(tuple(term) for term in poly)
                claimed = task["generator"]["minimal_tree_ops"]
                for size in range(claimed):
                    self.assertNotIn(identity, below[size], task["id"])

    def test_seeded_regeneration_is_byte_identical(self) -> None:
        other = generate_corpus(seed=11, quotas=dict(TINY),
                                excluded=set())
        self.assertEqual(json.dumps(other, sort_keys=True),
                         json.dumps(self.corpus, sort_keys=True))
        first = self.corpus["splits"]["eval"][0]["id"]
        self.assertEqual(other["splits"]["eval"][0]["id"], first)

    def test_train_stratum_major_and_eval_shuffled(self) -> None:
        train = _strata(self.corpus["splits"]["train"])
        self.assertEqual(train, sorted(train))
        eval_order = _strata(self.corpus["splits"]["eval"])
        self.assertNotEqual(eval_order, sorted(eval_order))

    def test_default_exclusions_cover_v1(self) -> None:
        v1 = load_v1_identities(V1)
        self.assertTrue(v1)
        default = generate_corpus(seed=23, quotas=dict(TINY))
        self.assertFalse(_identities(default) & v1)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
