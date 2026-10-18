"""Grid and frozen-input checks for the v2 benchmark."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rsi_programmer.benchmark_v2_runner import (
    _grid_valid,
    load_frozen_inputs,
)


class ProvenanceAndGridTests(unittest.TestCase):
    def test_frozen_inputs_validate_as_a_complete_corpus(self) -> None:
        protocol, corpus = load_frozen_inputs()
        self.assertEqual(protocol["corpus"]["sha256"],
                         "a8fd6add90b34da732da8dc96f2c0fafcabc2f017"
                         "a17cdef4b31303eec4ba921")
        self.assertEqual(len(corpus["splits"]["train"]), 24)
        self.assertEqual(len(corpus["splits"]["eval"]), 48)

    def test_grid_rejects_duplicate_and_missing_cells(self) -> None:
        expected = {(0, "trained_snn", "task", 512),
                    (1, "trained_snn", "task", 512)}
        self.assertTrue(_grid_valid(expected, sorted(expected)))
        self.assertFalse(_grid_valid(expected, [next(iter(expected))] * 2))
        self.assertFalse(_grid_valid(expected, [next(iter(expected))]))

    def test_generator_byte_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            corpus_path = root / "corpus.json"
            generator_path = root / "generator.js"
            generator_path.write_text("frozen", encoding="utf-8")
            corpus = {"splits": {"train": [], "eval": []}}
            protocol = {
                "protocol_id": "benchmark-pre-eval-v2",
                "controls": ["streaming", "heuristic", "seeded_random",
                             "untrained_snn", "trained_snn", "mlp",
                             "recurrent"],
                "training": {"seeds": [0, 1, 2], "epochs": 40,
                             "learning_rate": 0.05, "positive_cap": 16,
                             "negative_cap": 256, "sampling_seed": 82078},
                "evaluation": {"tree_bound": 5,
                               "candidate_scan_budgets":
                                   [512, 2048, 8192, 32768]},
                "decision_rule": {
                    "primary_budgets": [2048, 8192],
                    "minimum_margin": 0.10,
                    "minimum_families_higher": 2,
                    "matched_success_targets": [0.25, 0.5, 0.75, 1.0],
                    "claim_scope": "test",
                },
                "uncertainty": {"draws": 2000, "seed": 82079,
                                "confidence_level": 0.95},
                "corpus": {
                    "sha256": "",
                    "generator_source": {"generator.js": ""},
                    "counts": {"train_tasks": 24, "eval_tasks": 48,
                               "train_per_family": 8,
                               "tasks_per_eval_family": 16},
                    "families": ["balanced", "repeated_subtree", "skewed"],
                    "rows_per_task": {"observed": 12, "held": 12},
                },
            }
            generator_hash = hashlib.sha256(b"frozen").hexdigest()
            protocol["corpus"]["generator_source"]["generator.js"] = (
                generator_hash)
            protocol["corpus"]["sha256"] = hashlib.sha256(
                json.dumps(corpus).encode("utf-8")).hexdigest()
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
            protocol_hash = hashlib.sha256(
                protocol_path.read_bytes()).hexdigest()
            corpus_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
            generator_path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "generator hash"):
                load_frozen_inputs(
                    protocol_path, corpus_path, (generator_path,),
                    protocol_hash, corpus_hash,
                )
