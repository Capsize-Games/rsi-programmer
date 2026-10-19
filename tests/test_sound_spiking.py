"""Tests for the explicitly trained spiking sound classifier."""

from __future__ import annotations

import copy
import json
import unittest

from rsi_programmer.sound_features import FEATURE_COUNT
from rsi_programmer.sound_spiking import SoundSpiking


def band_example(band: int, phase: int = 0) -> list[float]:
    vector = [0.0] * FEATURE_COUNT
    for temporal_bin in range(12):
        selected = band
        if phase and temporal_bin >= 6:
            selected = 15 - band
        vector[temporal_bin * 20 + selected] = 1.0
    return vector


class SoundSpikingTests(unittest.TestCase):
    def test_trains_both_synapse_stages_and_scores_known_classes(self) -> None:
        vectors = [band_example(0), band_example(0, 1),
                   band_example(15), band_example(15, 1)]
        targets = [0, 0, 1, 1]
        network = SoundSpiking(seed=7)

        report = network.fit(vectors, targets, class_count=2)

        self.assertTrue(report["trained"])
        self.assertGreater(report["actual_spikes"], 0)
        self.assertGreater(report["changed_input_parameters"], 0)
        self.assertGreater(report["changed_readout_parameters"], 0)
        self.assertNotEqual(report["initialized_encoder_hash"],
                            network.encoder_hash())
        self.assertNotEqual(report["initialized_readout_hash"],
                            network.readout_hash())
        self.assertEqual(len(network.scores(vectors[0])), 2)
        self.assertGreater(network.scores(vectors[0])[0],
                           network.scores(vectors[0])[1])
        self.assertGreater(network.scores(vectors[2])[1],
                           network.scores(vectors[2])[0])

    def test_inference_is_repeatable_and_preserves_parameters(self) -> None:
        network = SoundSpiking(seed=3)
        network.fit([band_example(1), band_example(14)], [0, 1], 2)
        before_hash = network.weight_hash()
        before_metrics = network.metrics()

        first = network.scores(band_example(1))
        second = network.scores(band_example(1))

        self.assertEqual(first, second)
        self.assertEqual(network.weight_hash(), before_hash)
        self.assertEqual(network.metrics(), before_metrics)

    def test_snapshot_round_trip_and_atomic_rejection(self) -> None:
        original = SoundSpiking(seed=11)
        original.fit([band_example(2), band_example(13)], [0, 1], 2)
        snapshot = original.snapshot()
        json.dumps(snapshot, allow_nan=False)
        restored = SoundSpiking()
        restored.restore(snapshot)
        self.assertEqual(restored.scores(band_example(2)),
                         original.scores(band_example(2)))
        self.assertEqual(restored.weight_hash(), original.weight_hash())

        prior_hash = restored.weight_hash()
        invalid = copy.deepcopy(snapshot)
        invalid["input_weight"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            restored.restore(invalid)
        self.assertEqual(restored.weight_hash(), prior_hash)

    def test_one_class_learns_known_versus_reject_score(self) -> None:
        network = SoundSpiking(seed=5)
        result = network.fit(
            [band_example(4), band_example(4, 1)], [0, 0], 1
        )
        self.assertTrue(result["trained"])
        self.assertGreater(result["changed_input_parameters"], 0)
        self.assertGreater(result["changed_readout_parameters"], 0)
        self.assertEqual(len(network.scores(band_example(4))), 1)

    def test_rejects_invalid_training_and_untrained_inference(self) -> None:
        network = SoundSpiking()
        with self.assertRaises(ValueError):
            network.scores([0.0] * FEATURE_COUNT)
        with self.assertRaises(ValueError):
            network.fit([[0.0]], [0], 1)
        with self.assertRaises(ValueError):
            network.fit([band_example(0)], [1], 1)

    def test_untrained_snapshot_round_trip(self) -> None:
        network = SoundSpiking(seed=9)
        restored = SoundSpiking()
        restored.restore(network.snapshot())
        self.assertEqual(restored.metrics()["trained"], False)
        self.assertEqual(restored.weight_hash(), network.weight_hash())


if __name__ == "__main__":
    unittest.main()
