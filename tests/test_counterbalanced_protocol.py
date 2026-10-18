"""Frozen schedule integrity and policy isolation checks."""
from __future__ import annotations

import collections
import tempfile
import unittest
from pathlib import Path

from rsi_programmer.online_benchmark import load_corpus
from rsi_programmer.online_counterbalanced import SCHEDULE, load_schedule


class CounterbalancedProtocolTests(unittest.TestCase):
    def test_six_orders_balance_phase_position_and_predecessors(self) -> None:
        schedule = load_schedule()
        positions: collections.Counter = collections.Counter()
        predecessors: collections.Counter = collections.Counter()
        original = load_corpus()["tasks"]
        all_ids = {task["id"] for task in original}
        for order in schedule["orders"]:
            self.assertEqual(set(order["task_ids"]), all_ids)
            self.assertEqual(len(order["task_ids"]), 48)
            for position, phase in enumerate(order["phase_sequence"]):
                positions[phase, position] += 1
                block = [task["id"] for task in original
                         if task["phase"] == phase]
                self.assertEqual(order["task_ids"][position * 16:
                                                  (position + 1) * 16], block)
            for first, second in zip(order["phase_sequence"],
                                     order["phase_sequence"][1:]):
                predecessors[first, second] += 1
        self.assertEqual(set(positions.values()), {2})
        self.assertEqual(set(predecessors.values()), {2})

    def test_schedule_hash_rejects_even_whitespace_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_bytes(SCHEDULE.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_schedule(path)


if __name__ == "__main__":
    unittest.main()
