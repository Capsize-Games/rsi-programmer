"""Frozen function separation, replay labels and final-check isolation."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any
import unittest

from rsi_programmer.construction_check import check_graph
from rsi_programmer.construction_corpus import generate_corpus, teacher_steps

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "fixtures" / "construction_corpus_v1.json"


class ConstructionCorpusTests(unittest.TestCase):
    """Validate every frozen witness without fitting a network."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus: dict[str, Any] = json.loads(CORPUS.read_text())

    def test_regeneration_reproduces_exact_frozen_bytes(self) -> None:
        generated = (json.dumps(generate_corpus(), sort_keys=True,
                                separators=(",", ":")) + "\n").encode()
        self.assertEqual(generated, CORPUS.read_bytes())

    def test_splits_have_disjoint_functions_and_declared_quotas(self) -> None:
        seen: set[tuple[tuple[int, ...], ...]] = set()
        identifiers: set[str] = set()
        for split, tasks in self.corpus["splits"].items():
            counts = dict.fromkeys(("1", "2", "3", "4"), 0)
            for task in tasks:
                identity = tuple(tuple(term) for term in
                                 task["generator"]["polynomial"])
                self.assertNotIn(identity, seen)
                self.assertNotIn(task["id"], identifiers)
                seen.add(identity)
                identifiers.add(task["id"])
                self.assertEqual(task["split"], split)
                counts[str(task["generator"]["minimal_tree_ops"])] += 1
            for size, count in counts.items():
                self.assertEqual(count, self.corpus["quotas"][size][split])

    def test_every_witness_passes_independent_rows_and_polynomial_checks(
        self,
    ) -> None:
        for tasks in self.corpus["splits"].values():
            for task in tasks:
                observed = tuple(tuple(row) for row in task["observed"])
                held = tuple(tuple(row) for row in task["held"])
                observed_inputs = {(x, y) for x, y, _ in observed}
                self.assertEqual(len(observed_inputs), 36)
                self.assertFalse(observed_inputs & {
                    (x, y) for x, y, _ in held
                })
                identity = tuple(tuple(term) for term in
                                 task["generator"]["polynomial"])
                result = check_graph(task["generator"]["program"], observed,
                                     held, identity, task["max_tree_ops"])
                self.assertTrue(result["accepted"], task["id"])

    def test_teacher_replay_is_training_only_and_does_not_read_held_rows(
        self,
    ) -> None:
        for task in self.corpus["splits"]["train"]:
            poisoned = copy.deepcopy(task)
            poisoned["held"] = object()
            examples = teacher_steps(poisoned)
            state, stop = examples[-1]
            self.assertEqual(stop, {"op": "stop"})
            self.assertTrue(state.observed_match())
            self.assertFalse(state.stopped)
        for split in ("dev", "eval"):
            with self.assertRaises(ValueError):
                teacher_steps(self.corpus["splits"][split][0])

    def test_protocol_pins_the_corpus(self) -> None:
        path = ROOT / "fixtures" / "construction_v1.json"
        protocol = json.loads(path.read_text())
        self.assertEqual(protocol["corpus_sha256"],
                         hashlib.sha256(CORPUS.read_bytes()).hexdigest())
        self.assertFalse(protocol["evaluation"]["held_feedback"])
        self.assertFalse(protocol["learning"]["llm_teacher"])
