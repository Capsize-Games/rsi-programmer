"""Unit tests for audio frame measurements and fitted normalization."""

from __future__ import annotations

import math
import unittest

from rsi_programmer.audio_features import (
    BAND_EDGES_HZ,
    extract_frame,
    fit_band_normalizer,
    normalize_bands,
)


def sine_wave(frequency: float, sample_rate: int = 16000) -> list[float]:
    return [
        0.4 * math.sin(2.0 * math.pi * frequency * index / sample_rate)
        for index in range(round(0.2 * sample_rate))
    ]


class AudioFeatureTests(unittest.TestCase):
    def test_extracts_finite_metrics_and_tone_peak(self) -> None:
        result = extract_frame(sine_wave(440.0), 16000)
        self.assertEqual(len(result["band_dbfs"]), len(BAND_EDGES_HZ) - 1)
        self.assertAlmostEqual(result["peak_hz"], 440.0, delta=5.0)
        names = (
            "rms_dbfs", "peak_dbfs", "clip_fraction", "peak_hz",
            "tone_snr_db",
        )
        for name in names:
            self.assertTrue(math.isfinite(result[name]))
        self.assertTrue(
            all(math.isfinite(value) for value in result["band_dbfs"])
        )
        self.assertEqual(result["clip_fraction"], 0.0)

    def test_silence_and_clipping_have_expected_levels(self) -> None:
        silent = extract_frame([0.0] * 3200, 16000)
        clipped = extract_frame([1.0] * 3200, 16000)
        self.assertLess(silent["rms_dbfs"], -200.0)
        self.assertEqual(clipped["clip_fraction"], 1.0)
        self.assertEqual(clipped["peak_dbfs"], 0.0)

    def test_normalizer_fits_training_rows_and_is_bounded(self) -> None:
        training = [
            {"band_dbfs": [-50.0 + index, -60.0 + index,
                            -70.0 + index, -80.0 + index,
                            -90.0 + index, -100.0 + index]}
            for index in range(5)
        ]
        normalizer = fit_band_normalizer(training)
        self.assertEqual(len(normalizer["centers_db"]), 6)
        self.assertTrue(all(scale >= 3.0 for scale in normalizer["scales_db"]))
        normalized = normalize_bands([0.0] * 6, normalizer)
        self.assertTrue(all(0.0 < value < 1.0 for value in normalized))
        self.assertTrue(all(value > 0.5 for value in normalized))


if __name__ == "__main__":
    unittest.main()
