"""Integrity and leakage guards for the frozen pre-eval benchmark."""
from __future__ import annotations

import copy
import unittest
from pathlib import Path
from scripts import benchmark_pre_eval as benchmark

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "fixtures/benchmark_pre_eval_v1.json"
CORPUS = ROOT / "fixtures/benchmark_pre_eval_corpus_v1.json"


class FrozenInputsTests(unittest.TestCase):
    """Check immutable protocol bytes and corpus partition rules."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol_bytes = PROTOCOL.read_bytes()
        cls.corpus_bytes = CORPUS.read_bytes()
        cls.protocol, cls.corpus = benchmark.frozen_inputs(
            cls.protocol_bytes, cls.corpus_bytes,
        )

    def test_protocol_pins_complete_run_matrix(self) -> None:
        """Keep every seed, control, budget, and fit choice present."""
        self.assertEqual(
            self.protocol["training"]["seeds"], [0, 1, 2],
        )
        self.assertEqual(self.protocol["training"]["epochs"], 40)
        self.assertEqual(
            self.protocol["evaluation"]["candidate_scan_budgets"],
            [32, 128, 512, 2048],
        )
        self.assertEqual(
            set(self.protocol["controls"]) - {"shared_learning_rule"},
            set(benchmark.CONTROL_NAMES),
        )
        self.assertEqual(
            self.protocol["decision_rule"]["matched_success_targets"],
            [0.25, 0.5, 0.75, 1.0],
        )
        self.assertEqual(3 * 24 * 4 * len(benchmark.CONTROL_NAMES), 2016)

    def test_corpus_has_disjoint_fixed_partitions(self) -> None:
        """Validate fixed sizes, families, witnesses, and global rows."""
        train = self.corpus["splits"]["train"]
        evaluation = self.corpus["splits"]["eval"]
        self.assertEqual(len(train), 12)
        self.assertEqual(len(evaluation), 24)
        self.assertEqual(
            {task["family"] for task in train},
            {"balanced", "repeated_subtree", "skewed"},
        )
        self.assertEqual(
            {task["family"] for task in evaluation},
            {"balanced", "repeated_subtree", "skewed"},
        )

    def test_changed_input_bytes_are_rejected(self) -> None:
        """Reject altered protocol and corpus bytes before a run."""
        with self.assertRaisesRegex(ValueError, "protocol hash"):
            benchmark.frozen_inputs(
                self.protocol_bytes + b" ", self.corpus_bytes,
            )
        with self.assertRaisesRegex(ValueError, "corpus hash"):
            benchmark.frozen_inputs(
                self.protocol_bytes, self.corpus_bytes + b" ",
            )

    def test_witness_mutations_and_structure_leakage_are_rejected(
            self) -> None:
        """Catch changed outputs and repeated generating structures."""
        altered = copy.deepcopy(self.corpus)
        altered["splits"]["eval"][0]["held_out"][0][2] += 1
        with self.assertRaisesRegex(ValueError, "does not match witness"):
            benchmark.validate_corpus(altered, self.protocol)
        altered = copy.deepcopy(self.corpus)
        altered["splits"]["eval"][0]["generator"]["program_key"] = (
            altered["splits"]["train"][0]["generator"]["program_key"]
        )
        with self.assertRaisesRegex(ValueError, "structures must be unique"):
            benchmark.validate_corpus(altered, self.protocol)

    def test_training_labels_and_work_counts_are_deterministic(self) -> None:
        """Group setup is fixed by train rows, not elapsed-time values."""
        tasks = self.corpus["splits"]["train"]
        first_groups, first_cost = benchmark.training_groups(tasks)
        second_groups, second_cost = benchmark.training_groups(tasks)
        self.assertEqual(first_groups, second_groups)
        self.assertEqual(
            first_cost["generation_counters"],
            second_cost["generation_counters"],
        )
        self.assertEqual(
            first_cost["labeling_counters"],
            second_cost["labeling_counters"],
        )


if __name__ == "__main__":
    unittest.main()
