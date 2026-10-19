"""Cheap checks for the frozen sound comparison schedule."""

from __future__ import annotations

import unittest

from scripts.compare_sound_backends import (
    DEFAULT_PROTOCOL,
    LABEL_ORDER,
    load_protocol,
    probe_cases,
    stage_probe_sequences,
    validate_protocol,
)


class SoundComparisonProtocolTests(unittest.TestCase):
    """Check frozen seed disjointness and repeated stage query ordering."""

    def test_protocol_has_disjoint_training_and_probe_seeds(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL)
        train_seeds = {
            seed for entry in protocol["labels"].values()
            for seed in entry["training_seeds"]
        }
        queries = list(probe_cases(protocol))
        query_seeds = {case["seed"] for case in queries}
        self.assertFalse(train_seeds & query_seeds)
        self.assertTrue(all(seed >= 10000 for seed in query_seeds))
        self.assertEqual(len(queries), len(LABEL_ORDER) * 12)
        for family in LABEL_ORDER:
            self.assertEqual(
                sum(case["family"] == family for case in queries), 12
            )

    def test_same_query_sequence_is_used_at_each_stage(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL)
        stages = protocol["stages"]
        sequences = stage_probe_sequences(protocol)
        self.assertEqual([stage["id"] for stage in stages], [
            "initial_unknown", "three_labels", "four_labels", "after_reload"
        ])
        self.assertEqual(len(set(sequences.values())), 1)
        self.assertEqual(set(sequences), {stage["id"] for stage in stages})
        self.assertEqual(stages[1]["labels"], ["bell", "knock", "whistle"])
        self.assertEqual(stages[2]["labels"], list(LABEL_ORDER))
        self.assertTrue(stages[3]["reload"])

    def test_protocol_validation_rejects_seed_changes(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL)
        protocol["labels"]["bell"]["training_seeds"][0] += 1
        with self.assertRaisesRegex(ValueError, "training seeds"):
            validate_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
