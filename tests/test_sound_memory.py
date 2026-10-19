"""Nearest-example sound memory, persistence and held-out probes."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi_programmer.sound_features import SAMPLE_RATE, extract_sound
from rsi_programmer.sound_memory import SoundMemory


def tone(frequency: float, amplitude: float = 0.5) -> dict[str, object]:
    pcm = [amplitude * math.sin(2.0 * math.pi * frequency * index
                                / SAMPLE_RATE)
           for index in range(round(0.35 * SAMPLE_RATE))]
    return extract_sound(pcm)


def click_pattern(intervals: tuple[float, ...]) -> dict[str, object]:
    pcm = [0.0] * SAMPLE_RATE
    for start in intervals:
        offset = round(start * SAMPLE_RATE)
        for index in range(100):
            pcm[offset + index] = 0.7 if index % 2 == 0 else -0.7
    return extract_sound(pcm)


class SoundMemoryTests(unittest.TestCase):
    def test_nearest_examples_match_and_unknowns_reject(self) -> None:
        memory = SoundMemory()
        wake = tone(440.0)
        sleep = tone(1760.0)
        memory.teach("wake", wake, "wake")
        memory.teach("sleep", sleep, "sleep")
        self.assertEqual(memory.recognize(wake)["label"], "wake")
        self.assertEqual(memory.recognize(sleep)["label"], "sleep")
        quiet = extract_sound([0.0] * 400)
        self.assertIsNone(memory.recognize(quiet)["label"])
        self.assertEqual(memory.recognize(quiet)["reason"], "quiet")

    def test_single_label_rejects_far_band_and_different_rhythm(self) -> None:
        memory = SoundMemory()
        memory.teach("whistle", tone(440.0), "wake")

        far_band_probe = tone(4000.0)
        rhythm_probe = click_pattern((0.1, 0.3, 0.5, 0.7))
        self.assertEqual(memory.recognize(far_band_probe)["reason"],
                         "too_different")
        self.assertEqual(memory.recognize(rhythm_probe)["reason"],
                         "too_different")

    def test_smooth_pitch_drift_matches_known_sound_across_gains(self) -> None:
        memory = SoundMemory()
        for amplitude in (0.1, 0.4, 0.8):
            memory.teach("whistle", tone(440.0, amplitude), "wake")

        result = memory.recognize(tone(452.0, 0.3))
        self.assertEqual(result["label"], "whistle")

    def test_new_label_preserves_existing_vectors(self) -> None:
        memory = SoundMemory()
        memory.teach("wake", tone(440.0), "wake")
        before = memory.label_fingerprints()["wake"]
        memory.teach("timer", click_pattern((0.1, 0.5)), "timer")
        self.assertEqual(memory.label_fingerprints()["wake"], before)
        self.assertEqual(memory.recognize(tone(440.0))["label"], "wake")

    def test_probe_records_do_not_teach_or_enable_an_action(self) -> None:
        memory = SoundMemory()
        memory.teach("wake", tone(440.0), "wake")
        before = memory.summary()["total_examples"]
        record = memory.record_probe("wake", tone(440.0, amplitude=0.2))
        self.assertEqual(record["expected"], "wake")
        self.assertEqual(record["recognized"], "wake")
        self.assertTrue(record["correct"])
        self.assertIsNone(record["action"])
        self.assertEqual(memory.summary()["total_examples"], before)

    def test_added_examples_remain_independent_and_are_bounded(self) -> None:
        memory = SoundMemory()
        for _ in range(3):
            memory.teach("wake", tone(440.0), "wake")
        self.assertEqual(memory.summary()["labels"][0]["examples"], 3)
        for _ in range(17):
            memory.teach("wake", tone(440.0), "wake")
        with self.assertRaises(ValueError):
            memory.teach("wake", tone(440.0), "wake")

    def test_leave_one_out_report_is_explicitly_training_only(self) -> None:
        memory = SoundMemory()
        memory.teach("wake", tone(440.0), "wake")
        report = memory.leave_one_out_report()
        self.assertEqual(report["kind"], "training_only_leave_one_out")
        self.assertEqual(report["examples"], 1)

    def test_persistence_reload_and_corrupt_file_handling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory = SoundMemory(path)
            memory.teach("wake", tone(440.0), "wake")
            reloaded = SoundMemory(path)
            self.assertEqual(reloaded.recognize(tone(440.0))["label"], "wake")
            self.assertEqual(reloaded.label_fingerprints(),
                             memory.label_fingerprints())
            path.write_text("{broken", encoding="utf-8")
            corrupt = SoundMemory(path)
            self.assertEqual(corrupt.summary()["total_examples"], 0)
            self.assertIsNotNone(corrupt.summary()["load_error"])

    def test_label_and_action_validation(self) -> None:
        memory = SoundMemory()
        with self.assertRaises(ValueError):
            memory.teach("wake", tone(440.0), "run arbitrary code")
        with self.assertRaises(ValueError):
            memory.teach("", tone(440.0), "wake")

    def test_held_out_examples_are_not_added_to_memory(self) -> None:
        memory = SoundMemory()
        training_wake = tone(440.0)
        training_timer = click_pattern((0.1, 0.5))
        memory.teach("wake", training_wake, "wake")
        memory.teach("timer", training_timer, "timer")
        held_out = [tone(440.0, amplitude=0.18),
                    click_pattern((0.12, 0.52))]
        before = memory.summary()["total_examples"]
        results = [memory.recognize(item) for item in held_out]
        self.assertEqual(
            [item["label"] for item in results], ["wake", "timer"]
        )
        self.assertEqual(memory.summary()["total_examples"], before)

    def test_corrupt_file_cannot_be_overwritten_by_teaching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text("{broken", encoding="utf-8")
            memory = SoundMemory(path)
            with self.assertRaises(OSError):
                memory.teach("wake", tone(440.0), "wake")
            self.assertEqual(path.read_text(encoding="utf-8"), "{broken")
            self.assertEqual(memory.summary()["total_examples"], 0)

    def test_persist_failure_rolls_back_teaching_mutation(self) -> None:
        memory = SoundMemory()
        memory.teach("wake", tone(440.0), "wake")
        before = memory.summary()

        with patch.object(memory, "_save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                memory.teach("timer", click_pattern((0.1, 0.5)), "timer")

        after = memory.summary()
        self.assertEqual(after["revision"], before["revision"])
        self.assertEqual(after["fingerprint"], before["fingerprint"])
        self.assertEqual(after["total_examples"], before["total_examples"])


if __name__ == "__main__":
    unittest.main()
