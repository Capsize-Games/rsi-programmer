"""Structural integrity checks for the frozen v2 corpus."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "rsi_programmer" / "benchmark_v2_corpus.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_v2_corpus_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CORPUS_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORPUS_MODULE)


class BenchmarkV2CorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        path = ROOT / "fixtures" / "benchmark_pre_eval_corpus_v2.json"
        self.corpus = json.loads(path.read_text(encoding="utf-8"))

    def test_corpus_shape_and_balanced_family_counts(self) -> None:
        splits = self.corpus["splits"]
        self.assertEqual(len(splits["train"]), 24)
        self.assertEqual(len(splits["eval"]), 48)
        for split, expected in (("train", 8), ("eval", 16)):
            counts = {family: 0 for family in CORPUS_MODULE.FAMILIES}
            for task in splits[split]:
                counts[task["family"]] += 1
                self.assertEqual(len(task["train"]), 12)
                self.assertEqual(len(task["held_out"]), 12)
                self.assertEqual(task["max_tree_ops"], 5)
            self.assertEqual(set(counts.values()), {expected})

    def test_canonical_functions_are_unique_and_difficult(self) -> None:
        functions = set()
        v1_path = ROOT / "fixtures" / "benchmark_pre_eval_corpus_v1.json"
        v1 = json.loads(v1_path.read_text(encoding="utf-8"))
        old_functions = {
            CORPUS_MODULE.polynomial(task["generator"]["program"])
            for tasks in v1["splits"].values() for task in tasks
        }
        observed_pairs = set()
        for tasks in self.corpus["splits"].values():
            for task in tasks:
                generator = task["generator"]
                program = generator["program"]
                poly = CORPUS_MODULE.polynomial(program)
                encoded = tuple(tuple(term) for term in
                                generator["canonical_polynomial"])
                self.assertEqual(poly, encoded)
                self.assertTrue(CORPUS_MODULE.meets_difficulty_floor(poly))
                self.assertEqual(generator["tree_ops"],
                                 CORPUS_MODULE.tree_ops(program))
                self.assertIn(generator["tree_ops"], (4, 5))
                self.assertNotIn(poly, functions)
                self.assertNotIn(poly, old_functions)
                functions.add(poly)
                for row in task["train"] + task["held_out"]:
                    pair = (row[0], row[1])
                    self.assertNotIn(pair, observed_pairs)
                    observed_pairs.add(pair)
                    self.assertEqual(row[2],
                                     self._evaluate(poly, row[0], row[1]))
        self.assertEqual(len(functions), 72)
        self.assertEqual(len(observed_pairs), 72 * 24)
        self.assertEqual(
            {task["generator"]["tree_ops"]
             for tasks in self.corpus["splits"].values()
             for task in tasks}, {4, 5})

    @staticmethod
    def _evaluate(poly: tuple[tuple[int, int, int], ...],
                  x: int, y: int) -> int:
        return sum(coefficient * x ** x_power * y ** y_power
                   for x_power, y_power, coefficient in poly)


if __name__ == "__main__":
    unittest.main()
