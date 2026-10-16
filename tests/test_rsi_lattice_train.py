#!/usr/bin/env python3
"""Tests for the bit-slice team training loop (lattice v32)."""
from __future__ import annotations

import tempfile
import unittest

import torch

from rsi_programmer.lattice import BitLattice, train_lattice
from rsi_programmer.lineage import LineageStore
from rsi_programmer.modules import ModularPolicy
from rsi_programmer.ood import weights_hash

FIXTURE = "fixtures/rsi_shadow_tasks.json"


class LatticeTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Load encoder, train heads once, record lineage."""
        cls.tmp = tempfile.mkdtemp(prefix="rsi-lat-")
        cls.lat = BitLattice(seed=0)
        donor = ModularPolicy(seed=0)
        cls.lat.load_encoder(donor.weights())
        cls.before = weights_hash(cls.lat)
        cls.rep = train_lattice(cls.lat, FIXTURE,
                                cls.tmp + "/lat.db")
        cls.after = weights_hash(cls.lat)

    def test_heads_beat_chance_words_beat_sixteenth(self) -> None:
        """Per-head accuracy and word-exact clear chance."""
        acc = self.rep["head_accuracy"]
        print("heads=%s exact=%.3f rounds=%d"
              % ([round(a, 3) for a in acc],
                 self.rep["word_exact"], self.rep["rounds"]))
        for i, rate in enumerate(acc):
            self.assertGreater(rate, 0.5, i)
        self.assertGreater(self.rep["word_exact"], 1.0 / 16.0)

    def test_encoder_bytes_unchanged_by_training(self) -> None:
        """Frozen proof: shared-stage bytes identical."""
        self.assertNotEqual(self.before, self.after)
        mine = self.lat.weights()
        donor = ModularPolicy(seed=0).weights()
        for key, value in donor.items():
            if key.split(".")[0] in ("enc_fc", "enc", "st_fc",
                                     "st", "fb_fc"):
                self.assertTrue(torch.allclose(mine[key],
                                               value), key)
        for name, param in self.lat._net.named_parameters():
            if name.split(".")[0] in ("enc_fc", "enc", "st_fc",
                                      "st", "fb_fc"):
                self.assertFalse(param.requires_grad, name)

    def test_lineage_records_training_run(self) -> None:
        """One prepared intent names the fixture and metrics."""
        store = LineageStore(self.tmp + "/lat.db")
        rows = store.commands_in("rsi-lattice", "prepared")
        store.close()
        self.assertEqual(len(rows), 1)
        import json as _json
        intent = _json.loads(rows[0]["intent"])
        self.assertEqual(intent["fixture"], FIXTURE)
        self.assertIn("word_exact", intent)


if __name__ == "__main__":
    unittest.main()
