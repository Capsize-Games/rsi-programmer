"""Structural and numeric contracts for the frozen sensing data helpers."""
from __future__ import annotations

import math
import unittest

from rsi_programmer import sensing_data


class SensingDataTests(unittest.TestCase):
    """Check the pinned recording layout and fixed feature transforms."""

    def test_load_recordings_preserves_complete_source_groups(self) -> None:
        recordings = sensing_data.load_recordings()
        expected = {(config, speed)
                    for config in sensing_data.CONFIGURATIONS
                    for speed in sensing_data.SPEEDS}
        self.assertEqual(set(recordings), expected)
        self.assertEqual(len(recordings), 51)
        self.assertTrue(all(len(values) == 3000
                            for values in recordings.values()))
        self.assertTrue(all(math.isfinite(value)
                            for values in recordings.values()
                            for value in values))

    def test_rms_windows_are_disjoint_and_drop_incomplete_tail(self) -> None:
        values = [3.0] * 10 + [4.0] * 10 + [99.0]
        self.assertEqual(sensing_data.rms_windows(values), [3.0, 4.0])

    def test_rms_windows_reject_non_finite_samples(self) -> None:
        with self.assertRaises(ValueError):
            sensing_data.rms_windows([1.0] * 9 + [float("nan")])

    def test_rms_windows_reject_non_finite_derived_feature(self) -> None:
        with self.assertRaisesRegex(ValueError, "derived RMS"):
            sensing_data.rms_windows([1e308] * 10)

    def test_fit_scale_uses_median_and_floored_scaled_mad(self) -> None:
        center, scale = sensing_data.fit_scale([1.0, 2.0, 3.0])
        self.assertEqual(center, 2.0)
        self.assertAlmostEqual(scale, 1.4826)
        self.assertEqual(sensing_data.fit_scale([4.0, 4.0]),
                         (4.0, sensing_data.MIN_SCALE))

    def test_normalize_matches_frozen_bounded_formula(self) -> None:
        self.assertEqual(sensing_data.normalize(2.0, 2.0, 1.0), 0.5)
        self.assertAlmostEqual(
            sensing_data.normalize(3.0, 2.0, 1.0), 0.75)
        self.assertAlmostEqual(
            sensing_data.normalize(1.0, 2.0, 1.0), 0.25)
        self.assertGreaterEqual(
            sensing_data.normalize(1e100, 0.0, 1.0), 0.5)
        self.assertLessEqual(
            sensing_data.normalize(-1e100, 0.0, 1.0), 0.5)

    def test_fit_scale_requires_nonempty_finite_features(self) -> None:
        with self.assertRaises(ValueError):
            sensing_data.fit_scale([])
        with self.assertRaises(ValueError):
            sensing_data.fit_scale([1.0, float("inf")])

    def test_normalize_rejects_non_positive_scale(self) -> None:
        with self.assertRaises(ValueError):
            sensing_data.normalize(1.0, 0.0, 0.0)

    def test_normalize_rejects_non_finite_derived_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "derived normalization"):
            sensing_data.normalize(1e308, -1e308, 1.0)


if __name__ == "__main__":
    unittest.main()
