"""Verify neural decisions, paired holdouts, and persistent sound memory."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_programmer.sound_comparison import SoundComparison
from rsi_programmer.sound_memory import SoundMemory
from rsi_programmer.sound_spiking import SoundSpiking
from tests.test_sound_contract import tone


class SoundComparisonTests(unittest.TestCase):
    def taught(self, path: Path | None = None) -> SoundComparison:
        memory = SoundComparison(path)
        memory.teach("low", tone(440), "wake")
        memory.teach("high", tone(1760), "sleep")
        return memory

    def test_both_synapse_stages_learn_and_holdout_never_trains(self) -> None:
        memory = self.taught()
        before = memory.summary()
        result = memory.record_probe("low", tone(452, 0.23))
        after = memory.summary()
        self.assertTrue(result["correct"])
        self.assertTrue(result["baseline_correct"])
        self.assertIsNone(result["action"])
        self.assertEqual(result["model_hash_before"],
                         result["model_hash_after"])
        self.assertEqual(before["fingerprint"], after["fingerprint"])
        self.assertEqual(before["learning"], after["learning"])
        for stage in ("input", "readout"):
            self.assertGreater(
                after["learning"][f"changed_{stage}_parameters"], 0)
        self.assertEqual(after["comparison"]["snn"],
                         {"correct": 1, "total": 1})

    def test_snn_decides_and_matcher_cannot_rescue_wrong_neural_label(
        self,
    ) -> None:
        memory = self.taught()
        query = tone(452)
        self.assertEqual(memory.recognize_baseline(query)["label"], "low")
        # Sorted labels are high then low. Force the SNN to choose high.
        with patch.object(memory._model, "scores", return_value=[9.0, -9.0]):
            result = memory.recognize(query)
        self.assertEqual(result["neural_label"], "high")
        self.assertIsNone(result["label"])
        self.assertIsNone(result["action"])
        with patch.object(memory._model, "scores", side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                memory.recognize(query)

    def test_reloads_weights_without_training_and_remembers_old_sounds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory = self.taught(path)
            old = memory.label_fingerprints()
            memory.teach("new", tone(5200), "dance")
            for label, fingerprint in old.items():
                self.assertEqual(memory.label_fingerprints()[label],
                                 fingerprint)
            before = memory.summary()["learning"]["model_hash"]
            predictions = [memory.recognize(tone(hz))["label"]
                           for hz in (452, 1770, 5210)]
            with patch.object(SoundSpiking, "fit", side_effect=AssertionError):
                restored = SoundComparison(path)
            self.assertIsNone(restored.summary()["load_error"])
            self.assertEqual(restored.summary()["learning"]["model_hash"],
                             before)
            self.assertEqual([restored.recognize(tone(hz))["label"]
                              for hz in (452, 1770, 5210)], predictions)
            self.assertEqual(predictions, ["low", "high", "new"])
            self.assertEqual(restored.path.stat().st_mode & 0o777, 0o600)

    def test_legacy_examples_migrate_without_touching_original_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            legacy = SoundMemory(path)
            legacy.teach("low", tone(440), "wake")
            original = hashlib.sha256(path.read_bytes()).hexdigest()
            memory = SoundComparison(path)
            self.assertTrue(memory.summary()["legacy_examples_migrated"])
            self.assertEqual(memory.recognize(tone(452))["label"], "low")
            memory.teach("high", tone(1760), "sleep")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             original)

    def test_failed_save_rolls_back_examples_and_learned_parameters(
        self,
    ) -> None:
        memory = self.taught()
        before = memory.summary()
        with patch.object(memory, "_save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                memory.teach("new", tone(5200), "dance")
        self.assertEqual(memory.summary(), before)

    def test_corrupt_memory_is_preserved_and_blocks_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            sidecar = path.with_name("memory-snn-v1.json")
            sidecar.write_text("{broken")
            memory = SoundComparison(path)
            self.assertTrue(memory.summary()["load_error"])
            with self.assertRaises(OSError):
                memory.teach("low", tone(440), "wake")
            self.assertEqual(sidecar.read_text(), "{broken")

    def test_changed_training_identity_and_probe_counts_are_rejected(
        self,
    ) -> None:
        memory = self.taught()
        memory.record_probe("low", tone(452))
        saved = memory._snapshot()
        invalid = copy.deepcopy(saved)
        invalid["memory"]["labels"]["renamed"] = (
            invalid["memory"]["labels"].pop("low"))
        before = memory.summary()
        with self.assertRaises(ValueError):
            memory._restore(invalid)
        self.assertEqual(memory.summary(), before)
        invalid = copy.deepcopy(saved)
        invalid["probes"][0]["correct"] = False
        with self.assertRaises(ValueError):
            memory._restore(invalid)
        self.assertEqual(memory.summary(), before)

    def test_delete_retrains_only_retained_examples(self) -> None:
        memory = self.taught()
        high = memory.label_fingerprints()["high"]
        self.assertTrue(memory.delete_label("low"))
        self.assertEqual(memory.label_fingerprints(), {"high": high})
        self.assertEqual(memory.recognize(tone(1770))["label"], "high")
        self.assertTrue(memory.delete_label("high"))
        self.assertIsNone(memory.recognize(tone(440))["label"])

    def test_summary_contains_no_private_parameter_or_example_arrays(
        self,
    ) -> None:
        text = json.dumps(self.taught().summary(), allow_nan=False)
        for name in ("input_weight", "readout_weight", '"vector"'):
            self.assertNotIn(name, text)

    def test_matcher_snapshot_validation_is_atomic_and_copied(self) -> None:
        memory = SoundMemory()
        memory.teach("low", tone(440), "wake")
        original = memory.fingerprint()
        snapshot = memory.snapshot()
        snapshot["labels"]["low"]["examples"][0][0] = 1.0
        self.assertEqual(memory.fingerprint(), original)
        snapshot["probes"] = [{"expected": "low", "correct": "bad"}]
        with self.assertRaises(ValueError):
            memory.restore(snapshot)
        self.assertEqual(memory.fingerprint(), original)


if __name__ == "__main__":
    unittest.main()
