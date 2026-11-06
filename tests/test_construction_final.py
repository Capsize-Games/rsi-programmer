"""Sealed final gates without ever opening the frozen final split."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from rsi_programmer.construction_corpus import QUOTAS
from rsi_programmer.construction_final import (
    FINAL_SPLIT,
    cost_gate,
    evaluate_final,
    learning_gate,
    require_final_only,
    verify_frozen_corpus,
    verify_generator_sources,
)
from rsi_programmer.online_graph import graph_polynomial
from rsi_programmer.probe import Expr, lower_to_dag

PROGRAMS = {"add": lambda x, y: x + y, "mul": lambda x, y: x * y}


def _task(task_id: str, split: str, op: str) -> dict[str, Any]:
    """Build a tiny witness-backed task; never read the sealed corpus."""
    program = lower_to_dag(Expr(op, Expr("input", "x"),
                                Expr("input", "y")))
    poly = graph_polynomial(program)
    rows = [[x, y, PROGRAMS[op](x, y)]
            for x in range(-1, 2) for y in range(-1, 2)]
    return {"id": task_id, "split": split, "observed": rows[:6],
            "held": rows[6:], "max_tree_ops": 4,
            "generator": {"minimal_tree_ops": 1, "program": program,
                          "polynomial": [list(term) for term in poly]}}


def _bundle(directory: Path) -> tuple[Path, Path, Path]:
    """Write a synthetic corpus, protocol and receipt for tests."""
    corpus = {"splits": {"train": [_task("train-add", "train", "add")],
                         "dev": [], "eval": [_task("final-mul",
                                                   "eval", "mul")]}}
    raw = (json.dumps(corpus, sort_keys=True) + "\n").encode()
    corpus_path = directory / "corpus.json"
    corpus_path.write_bytes(raw)
    protocol = {"corpus_sha256": hashlib.sha256(raw).hexdigest(),
                "evaluation": {"model_seeds": [0],
                               "execution_caps": [64, 256]},
                "source_provenance": {"generator_sha256": {}}}
    protocol_path = directory / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    receipt = directory / "receipt.json"
    receipt.write_text(json.dumps({"recipe": "hidden=64,epochs=1"}))
    return corpus_path, protocol_path, receipt


class FinalSealTests(unittest.TestCase):
    """Hash and split guards refuse early or tampered final access."""

    def test_matching_hash_loads_without_reading_real_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            corpus_path, protocol_path, _ = _bundle(Path(name))
            bundle = verify_frozen_corpus(corpus_path, protocol_path)
            self.assertEqual(len(bundle["corpus"]["splits"]["eval"]), 1)

    def test_tampered_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            corpus_path, protocol_path, _ = _bundle(Path(name))
            corpus_path.write_bytes(corpus_path.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                verify_frozen_corpus(corpus_path, protocol_path)

    def test_changed_generator_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "gen.py").write_text("x = 1\n")
            protocol = {"source_provenance": {"generator_sha256":
                                              {"gen.py": "0" * 64}}}
            with self.assertRaises(ValueError):
                verify_generator_sources(root, protocol)

    def test_final_only_rejects_dev_empty_and_mixed(self) -> None:
        require_final_only([_task("f", "eval", "add")])
        for tasks in ([], [_task("d", "dev", "add")],
                      [_task("f", "eval", "add"),
                       _task("d", "dev", "add")]):
            with self.assertRaises(ValueError):
                require_final_only(tasks)

    def test_final_split_name_matches_generator_quotas(self) -> None:
        # The 2026-11-05 first one-shot invocation died pre-inference
        # on splits["final"]; the generator names the split "eval".
        for quotas in QUOTAS.values():
            self.assertIn(FINAL_SPLIT, quotas)
        self.assertEqual(
            sum(q[FINAL_SPLIT] for q in QUOTAS.values()), 244)

    def test_evaluate_final_refuses_dev_tasks(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_final([_task("d", "dev", "add")], "streaming",
                           None, 64)


class FinalGateTests(unittest.TestCase):
    """Gate arithmetic matches the frozen decision rules."""

    def test_learning_gate_passes_above_margin(self) -> None:
        rates = {0: {"trained": 0.5, "untrained": 0.35, "random": 0.35}}
        self.assertTrue(learning_gate(rates)["passed"])

    def test_learning_gate_fails_below_margin(self) -> None:
        rates = {0: {"trained": 0.45, "untrained": 0.4, "random": 0.1}}
        self.assertFalse(learning_gate(rates)["passed"])

    def test_cost_gate_passes_with_cheaper_full_coverage(self) -> None:
        trained = [{"cap": 64, "accepted": 2, "tasks": 2,
                    "cpu_seconds": 1.0}]
        streaming = [{"cap": 64, "accepted": 2, "tasks": 2,
                      "cpu_seconds": 2.0}]
        gate = cost_gate(trained, streaming, 0.2, 2)
        self.assertTrue(gate["passed"])
        self.assertAlmostEqual(gate["ratio"], 0.6)

    def test_cost_gate_fails_without_full_coverage(self) -> None:
        trained = [{"cap": 64, "accepted": 1, "tasks": 2,
                    "cpu_seconds": 1.0}]
        streaming = [{"cap": 64, "accepted": 2, "tasks": 2,
                      "cpu_seconds": 2.0}]
        gate = cost_gate(trained, streaming, 0.0, 2)
        self.assertFalse(gate["passed"])

    def test_cost_gate_fails_above_ratio(self) -> None:
        trained = [{"cap": 64, "accepted": 2, "tasks": 2,
                    "cpu_seconds": 2.0}]
        streaming = [{"cap": 64, "accepted": 2, "tasks": 2,
                      "cpu_seconds": 2.0}]
        self.assertFalse(cost_gate(trained, streaming, 0.0, 2)["passed"])

    def test_streaming_solves_trivial_final_task(self) -> None:
        summary, rows = evaluate_final([_task("f", "eval", "add")],
                                       "streaming", None, 64)
        self.assertEqual(summary["accepted"], 1)
        self.assertTrue(rows[0]["checker"]["accepted"])


class FinalRunnerTests(unittest.TestCase):
    """One-shot runner refusals and a tiny end-to-end run."""

    def _run(self, argv: list[str]) -> int:
        from scripts.run_construction_final import main
        with mock.patch.object(sys, "argv", argv):
            return main()

    def test_missing_receipt_refuses_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            corpus_path, protocol_path, _ = _bundle(root)
            argv = ["run", "--frozen-receipt", str(root / "absent"),
                    "--corpus", str(corpus_path),
                    "--protocol", str(protocol_path),
                    "--output", str(root / "out"), "--epochs", "1"]
            with self.assertRaises(ValueError):
                self._run(argv)
            self.assertFalse((root / "out").exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            corpus_path, protocol_path, receipt = _bundle(root)
            output = root / "out"
            output.mkdir()
            argv = ["run", "--frozen-receipt", str(receipt),
                    "--corpus", str(corpus_path),
                    "--protocol", str(protocol_path),
                    "--output", str(output), "--epochs", "1"]
            with self.assertRaises(FileExistsError):
                self._run(argv)

    def test_tiny_end_to_end_run_writes_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            corpus_path, protocol_path, receipt = _bundle(root)
            output = root / "out"
            argv = ["run", "--frozen-receipt", str(receipt),
                    "--corpus", str(corpus_path),
                    "--protocol", str(protocol_path),
                    "--output", str(output), "--epochs", "1"]
            self.assertEqual(self._run(argv), 0)
            report = json.loads((output / "report.json").read_text())
            self.assertIn(report["verdict"], (True, False))
            self.assertEqual(report["corpus_sha256"],
                             json.loads(
                                 protocol_path.read_text())["corpus_sha256"])
            self.assertEqual(report["receipt_sha256"],
                             hashlib.sha256(
                                 receipt.read_bytes()).hexdigest())
            self.assertEqual(len(report["model_seeds"]), 1)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
