"""Focused checks for ablation artifact provenance guards."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts import measure_author_ablation as measurement


class MeasurementProvenanceTests(unittest.TestCase):
    @staticmethod
    def _args(fixture: Path) -> argparse.Namespace:
        return argparse.Namespace(
            train_fixture=fixture, eval_fixtures=[fixture], seeds=[],
            epochs=0, rounds=0, topk=1, budget=0, width=1)

    @staticmethod
    def _write_fixture(path: Path) -> None:
        path.write_text(json.dumps({"tasks": []}), encoding="utf-8")

    def test_measure_refuses_fixture_changed_after_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.json"
            self._write_fixture(fixture)
            calls = 0

            def load_rows(_path: Path) -> tuple[list[Any], list[Any]]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    fixture.write_text('{"tasks": [1]}', encoding="utf-8")
                return [], []

            with patch.object(measurement, "ROOT", root), \
                    patch.object(measurement, "_source_hash",
                                 return_value="fixed-source"), \
                    patch.object(measurement, "_dependency_provenance",
                                 return_value={
                                     "content_sha256": "fixed-dependency"}), \
                    patch.object(measurement, "_tasks", side_effect=load_rows):
                with self.assertRaisesRegex(
                        RuntimeError, "fixture files changed during"):
                    measurement.measure(self._args(fixture))

    def test_measure_refuses_dependency_source_changed_during_run(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.json"
            self._write_fixture(fixture)
            package_root = root / "spikeforge"
            package_root.mkdir()
            source = package_root / "builder.py"
            source.write_text("BUILD = 1\n", encoding="utf-8")
            calls = 0

            def load_rows(_path: Path) -> tuple[list[Any], list[Any]]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    source.write_text("BUILD = 2\n", encoding="utf-8")
                return [], []

            with patch.object(measurement, "ROOT", root), \
                    patch.object(measurement, "_source_hash",
                                 return_value="fixed-source"), \
                    patch.object(measurement, "_package_roots",
                                 return_value=[package_root]), \
                    patch.object(measurement, "_tasks", side_effect=load_rows):
                with self.assertRaisesRegex(
                        RuntimeError, "SpikeForge Python sources changed"):
                    measurement.measure(self._args(fixture))

    def test_train_fit_held_out_near_miss_is_not_accepted(self) -> None:
        graph = {
            "input_types": {"x": "Int", "y": "Int"},
            "nodes": [{"op": "add", "type": "Int",
                       "args": ["x", "x"]}],
            "output": 0,
        }
        result: dict[str, Any] = {
            "graph": graph, "solved": True, "verdict": "solved",
            "signature_evaluations": 1,
            "operand_signature_evaluations": 2,
            "examined": 1, "committed_nodes": 1,
            "output_nodes": 1, "budget_unit": "generated_triples",
        }
        task_row = ("near-miss", ((1, 1), (2, 2)), (2, 4), 1)
        task_data = {"held_out": [[1, 2, 3]]}

        with patch.object(measurement, "author_search", return_value=result):
            row = measurement._measure_one(
                None, "full", "author", task_row, task_data,
                [], 1, 1, 1)

        self.assertTrue(row["solved_train"])
        self.assertTrue(row["train_correct"])
        self.assertEqual(row["train_hits"], row["train_count"])
        self.assertFalse(row["held_correct"])
        self.assertFalse(row["accepted"])

    def test_raw_rows_use_portable_fixture_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.json"
            self._write_fixture(fixture)
            args = self._args(fixture)
            args.seeds = [0]
            task = ("tiny", ((1, 2),), (3,), 1)
            with patch.object(measurement, "ROOT", root), \
                    patch.object(measurement, "_source_hash",
                                 return_value="fixed-source"), \
                    patch.object(measurement, "_dependency_provenance",
                                 return_value={
                                     "content_sha256": "fixed-dependency"}), \
                    patch.object(measurement, "_tasks",
                                 return_value=([task], [{"id": "tiny"}])), \
                    patch.object(measurement, "SearchPolicy"), \
                    patch.object(measurement, "train_author",
                                 return_value=1), \
                    patch.object(measurement, "refine_author",
                                 return_value=([1.0] * len(
                                     measurement.EXTRA), [])), \
                    patch.object(measurement, "_measure_one",
                                 side_effect=lambda *a: {}):
                report = measurement.measure(args)
        self.assertTrue(report["results"])
        self.assertEqual({row["fixture"] for row in report["results"]},
                         {"fixture.json"})
