"""Training examples and held-out sound probes for feature extraction."""

from __future__ import annotations

import math
import unittest

from rsi_programmer.sound_features import (
    FEATURE_COUNT,
    SAMPLE_RATE,
    extract_sound,
)


def tone(frequency: float, amplitude: float = 0.5) -> list[float]:
    return [amplitude * math.sin(2.0 * math.pi * frequency * index
                                 / SAMPLE_RATE)
            for index in range(round(0.35 * SAMPLE_RATE))]


def clicks(intervals: tuple[float, ...]) -> list[float]:
    samples = [0.0] * round(1.0 * SAMPLE_RATE)
    for start in intervals:
        offset = round(start * SAMPLE_RATE)
        for index in range(80):
            if offset + index < len(samples):
                samples[offset + index] = 0.7 if index % 2 == 0 else -0.7
    return samples


class SoundFeatureTests(unittest.TestCase):
    def test_descriptor_has_fixed_shape_and_no_pcm(self) -> None:
        result = extract_sound(tone(440.0))
        self.assertEqual(result["version"], 1)
        self.assertEqual(result["sample_rate"], SAMPLE_RATE)
        self.assertTrue(result["active"])
        self.assertEqual(len(result["vector"]), FEATURE_COUNT)
        self.assertTrue(
            all(math.isfinite(value) for value in result["vector"])
        )
        self.assertNotIn("samples", result)

    def test_gain_changes_preserve_spectral_shape(self) -> None:
        quiet = extract_sound(tone(440.0, amplitude=0.08))["vector"]
        loud = extract_sound(tone(440.0, amplitude=0.7))["vector"]
        spectral_indexes = [index for index in range(FEATURE_COUNT)
                            if index % 20 < 16]
        differences = [abs(quiet[index] - loud[index])
                       for index in spectral_indexes]
        self.assertLess(max(differences), 1e-5)

    def test_tones_and_rhythm_have_distinct_descriptors(self) -> None:
        low = extract_sound(tone(220.0))["vector"]
        high = extract_sound(tone(1760.0))["vector"]
        slow = extract_sound(clicks((0.1, 0.55)))["vector"]
        fast = extract_sound(clicks((0.1, 0.3, 0.5, 0.7)))["vector"]
        self.assertGreater(_distance(low, high), 0.05)
        self.assertGreater(_distance(slow, fast), 0.01)

    def test_silence_is_marked_inactive(self) -> None:
        result = extract_sound([0.0] * 400)
        self.assertFalse(result["active"])

    def test_invalid_pcm_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_sound([])
        with self.assertRaises(ValueError):
            extract_sound([float("nan")])
        with self.assertRaises(ValueError):
            extract_sound([1.1])


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right))
                     / len(left))


if __name__ == "__main__":
    unittest.main()
