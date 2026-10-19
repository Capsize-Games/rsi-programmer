"""End-to-end contracts for teachable sound recognition and persistence."""

from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path

from rsi_programmer.sound_features import extract_sound
from rsi_programmer.sound_memory import SoundMemory


SAMPLE_RATE = 16000


def tone(frequency: float, gain: float = 0.4) -> dict[str, object]:
    samples = [
        gain * math.sin(2 * math.pi * frequency * i / SAMPLE_RATE)
        for i in range(SAMPLE_RATE)
    ]
    return extract_sound(samples, SAMPLE_RATE)


def seeded_noise(seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    samples = [rng.gauss(0.0, 0.04) for _ in range(SAMPLE_RATE)]
    return extract_sound(samples, SAMPLE_RATE)


def rhythmic_tone(
    frequency: float,
    bursts: tuple[tuple[float, float], ...],
    gain: float = 0.4,
) -> dict[str, object]:
    samples: list[float] = []
    for index in range(SAMPLE_RATE):
        position = index / SAMPLE_RATE
        active = any(start <= position < start + duration
                     for start, duration in bursts)
        value = (gain * math.sin(2 * math.pi * frequency * position)
                 if active else 0.0)
        samples.append(value)
    return extract_sound(samples, SAMPLE_RATE)


class SoundLearningContractTests(unittest.TestCase):
    def test_held_out_examples_survive_added_class_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sounds.json"
            memory = SoundMemory(path)
            classes = (
                ("low tone", 440.0, "wake"),
                ("mid tone", 1320.0, "sleep"),
                ("high tone", 3520.0, "dance"),
            )
            held_out = (tone(452.0, 0.23), tone(1304.0, 0.71),
                        tone(3560.0, 0.31))
            for label, frequency, action in classes:
                for gain in (0.24, 0.42, 0.68):
                    memory.teach(label, tone(frequency, gain), action)

            before = memory.summary()
            outcomes = [memory.recognize(item) for item in held_out]
            self.assertEqual([item["label"] for item in outcomes],
                             [item[0] for item in classes])
            for expected, item in zip((row[0] for row in classes), held_out):
                memory.record_probe(expected, item)
            self.assertEqual(memory.summary()["total_examples"],
                             before["total_examples"])

            memory.teach("new class", tone(5200.0), "timer")
            after = memory.summary()
            retained = {
                **before["label_fingerprints"],
                "new class": after["label_fingerprints"]["new class"],
            }
            self.assertEqual(after["label_fingerprints"], retained)
            self.assertEqual([memory.recognize(item)["label"]
                              for item in held_out],
                             [item[0] for item in classes])

            restored = SoundMemory(path)
            self.assertEqual(restored.summary()["fingerprint"],
                             after["fingerprint"])
            self.assertEqual([restored.recognize(item)["label"]
                              for item in held_out],
                             [item[0] for item in classes])

    def test_silence_and_noise_do_not_trigger_actions(self) -> None:
        memory = SoundMemory()
        for label, frequency, action in (
                ("one", 440.0, "wake"), ("two", 1760.0, "sleep")):
            memory.teach(label, tone(frequency), action)
        silence = extract_sound([0.0] * SAMPLE_RATE, SAMPLE_RATE)
        for descriptor in (silence, seeded_noise(909)):
            with self.subTest(descriptor="silence" if descriptor is silence
                              else "noise"):
                result = memory.recognize(descriptor)
                self.assertIsNone(result["label"])
                self.assertIsNone(result["action"])

    def test_single_class_rejects_far_unknowns_and_different_rhythms(
        self,
    ) -> None:
        memory = SoundMemory()
        taught = rhythmic_tone(440.0, ((0.12, 0.10), (0.62, 0.10)))
        memory.teach("two-pulse sound", taught, "wake")

        # A one-label memory cannot rely on a competing label to manufacture
        # ambiguity rejection. Require rejection of far-off sounds and
        # substantially different rhythms, while allowing ordinary pitch
        # variation within a feature band.
        unrelated = (
            tone(6000.0),
            rhythmic_tone(440.0, ((0.0, 1.0),)),
            rhythmic_tone(440.0, ((0.10, 0.10), (0.30, 0.10),
                                  (0.50, 0.10), (0.70, 0.10),
                                  (0.90, 0.10))),
            seeded_noise(4102),
        )
        for descriptor in unrelated:
            with self.subTest(active=descriptor["active"]):
                result = memory.recognize(descriptor)
                self.assertIsNone(result["label"])
                self.assertIsNone(result["action"])

    def test_confusable_equal_examples_are_rejected(self) -> None:
        memory = SoundMemory()
        example = tone(880.0)
        memory.teach("first", example, "wake")
        memory.teach("second", example, "sleep")
        result = memory.recognize(example)
        self.assertIsNone(result["label"])
        self.assertIsNone(result["action"])
        self.assertEqual(result["reason"], "ambiguous")


if __name__ == "__main__":
    unittest.main()
