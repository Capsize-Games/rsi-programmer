#!/usr/bin/env python3
"""Tests for the bit-slice team training loop (lattice v32)."""
from __future__ import annotations

import tempfile
import unittest

import torch

from rsi_programmer.lattice import BitLattice
from rsi_programmer.ltrain import train_lattice
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

    def test_heads_beat_chance_words_beat_third(self) -> None:
        """Per-head accuracy and word-exact clear chance.

        Chance is one in three: training triples name only
        add/sub/mul, so word-exact must beat uniform-opcode
        guessing, and each head must beat a coin flip.
        """
        acc = self.rep["head_accuracy"]
        print("heads=%s exact=%.3f rounds=%d"
              % ([round(a, 3) for a in acc],
                 self.rep["word_exact"], self.rep["rounds"]))
        for i, rate in enumerate(acc):
            self.assertGreater(rate, 0.5, i)
        self.assertGreater(self.rep["word_exact"], 1.0 / 3.0)

    def test_encoder_coadapts_from_donor(self) -> None:
        """Joint training moves shared weights off the donor.

        Frozen shared weights leave fresh spiking heads
        outside the surrogate band (measured exactly-zero
        gradients); the encoder co-adapts instead while
        eval-time weights_hash still guards every benchmark.
        """
        self.assertNotEqual(self.before, self.after)
        mine = self.lat.weights()
        donor = ModularPolicy(seed=0).weights()
        moved = [key for key, value in donor.items()
                 if key in mine and
                 not torch.allclose(mine[key], value)]
        self.assertGreater(len(moved), 0)
        print("moved=%d/%d" % (len(moved), len(donor)))

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
