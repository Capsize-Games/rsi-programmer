"""Sealed data-scaling final: units, refusals, fixture, tiny e2e."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.construction_scaling_final import (
    REQUIRED_KEYS, credited_solves, decide_scaling, require_scaling_budget,
    run_scaling_final, verify_scaling_bundle)
from rsi_programmer.online_graph import graph_polynomial
from rsi_programmer.probe import Expr, lower_to_dag

PROGRAMS = {"add": lambda x, y: x + y, "mul": lambda x, y: x * y}
KILL = ("if the data-scaled learned arm does not beat streaming on "
        "solves at equal total CPU (fit charged, truncated at the "
        "common budget, declared volume 12,000) on the sealed "
        "shape-broad bank, the data-scaling route closes at "
        "accessible scale \u2014 no retest at larger volume, wider "
        "beam, or larger train pool.")


def _task(task_id: str, split: str, op: str, minimal: int
          ) -> dict[str, Any]:
    """Build a tiny witness-backed task with a stratum label."""
    program = lower_to_dag(Expr(op, Expr("input", "x"), Expr("input", "y")))
    poly = graph_polynomial(program)
    rows = [[x, y, PROGRAMS[op](x, y)] for x in range(-1, 2)
            for y in range(-1, 2)]
    return {"id": task_id, "split": split, "observed": rows[:6],
            "held": rows[6:], "max_tree_ops": 4,
            "generator": {"program": program, "minimal_tree_ops": minimal,
                          "polynomial": [list(t) for t in poly]}}


def _evaluation(**over: Any) -> dict[str, Any]:
    """Frozen-shaped evaluation knobs with test overrides."""
    evaluation: dict[str, Any] = {
        "model_seeds": [0], "per_task_cpu_seconds": 5.0, "beam_width": 8,
        "bound": 4, "budget": 64, "imitation_epochs": 1,
        "declared_volume": 2, "frozen_at": "2026-11-06"}
    evaluation.update(over)
    return evaluation


def _bundle(directory: Path) -> tuple[Path, Path]:
    """Write a synthetic corpus and protocol with a matching hash."""
    train = [_task("train-add", "train", "add", 6)]
    final = [_task("eval-mul", "eval", "mul", 6),
             _task("eval-add", "eval", "add", 7)]
    corpus = {"splits": {"train": train, "eval": final}}
    raw = (json.dumps(corpus, sort_keys=True) + "\n").encode()
    corpus_path = directory / "corpus.json"
    corpus_path.write_bytes(raw)
    protocol = {"corpus_sha256": hashlib.sha256(raw).hexdigest(),
                "quotas": {"train": 1, "eval": 2},
                "source_provenance": {"generator_sha256": {}},
                "evaluation": _evaluation(),
                "decision_rules": {"kill_criterion": KILL}}
    protocol_path = directory / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    return corpus_path, protocol_path


def _tweak(protocol: Path, *path: str, value: Any) -> None:
    """Rewrite one nested protocol value for refusal tests."""
    data = json.loads(protocol.read_text())
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    protocol.write_text(json.dumps(data))


def _tiny_report() -> dict[str, Any]:
    """Run one seed with tiny knobs; return the verdict report."""
    train = [_task("train-add", "train", "add", 6)]
    final = [_task("eval-add", "eval", "add", 6),
             _task("eval-mul", "eval", "mul", 7)]
    corpus = {"splits": {"train": train, "eval": final}}
    protocol = {"corpus_sha256": "synthetic",
                "quotas": {"train": 1, "eval": 2},
                "source_provenance": {"generator_sha256": {}},
                "evaluation": _evaluation(),
                "decision_rules": {"kill_criterion": KILL}}
    return run_scaling_final(corpus, protocol)


class ScalingFinalTests(unittest.TestCase):
    """Units, refusals, fixture contract, and a tiny end-to-end."""

    def test_credited_clear_pass(self) -> None:
        """Learned beating both arms at B* passes with the kill text."""
        rows = [{"accepted": True, "cpu_seconds": 0.1,
                 "stratum": 6}]
        self.assertEqual(credited_solves(rows, 0.0, 1.0), 1)
        verdict = decide_scaling(3.0, 2.0, 1.0, 1.0, KILL)
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["kill_criterion"], KILL)

    def test_credited_kill_below_streaming(self) -> None:
        """Learned below streaming fails even above untrained."""
        verdict = decide_scaling(1.0, 2.0, 0.5, 1.0, KILL)
        self.assertFalse(verdict["passed"])

    def test_sanity_gate_below_untrained(self) -> None:
        """Learned at or below untrained fails the sanity gate."""
        self.assertFalse(decide_scaling(2.0, 1.0, 2.0, 1.0,
                                        KILL)["passed"])
        self.assertFalse(decide_scaling(1.0, 0.5, 2.0, 1.0,
                                        KILL)["passed"])

    def test_truncation_boundary(self) -> None:
        """A row crossing B* (and later rows) earns no credit."""
        rows = [{"accepted": True, "cpu_seconds": 0.6, "stratum": 6},
                {"accepted": True, "cpu_seconds": 0.6, "stratum": 7}]
        self.assertEqual(credited_solves(rows, 0.0, 1.0), 1)
        self.assertEqual(credited_solves(rows, 0.0, 1.5), 2)

    def test_fit_charged_first(self) -> None:
        """Fit CPU pushes the first eval row past a tight budget."""
        rows = [{"accepted": True, "cpu_seconds": 0.2, "stratum": 6}]
        self.assertEqual(credited_solves(rows, 0.9, 1.0), 0)
        self.assertEqual(credited_solves(rows, 0.9, 1.2), 1)

    def test_matching_hash_loads(self) -> None:
        """A frozen synthetic bundle verifies with its eval split."""
        with tempfile.TemporaryDirectory() as name:
            corpus, protocol = _bundle(Path(name))
            bundle = verify_scaling_bundle(corpus, protocol)
            self.assertEqual(len(bundle["corpus"]["splits"]["eval"]), 2)
            self.assertEqual(require_scaling_budget(bundle["protocol"]),
                             5.0)

    def test_tampered_bytes_rejected(self) -> None:
        """Edited corpus bytes fail the frozen hash check."""
        with tempfile.TemporaryDirectory() as name:
            corpus, protocol = _bundle(Path(name))
            corpus.write_bytes(corpus.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                verify_scaling_bundle(corpus, protocol)

    def test_null_hash_refuses(self) -> None:
        """A null corpus hash refuses before any inference."""
        with tempfile.TemporaryDirectory() as name:
            corpus, protocol = _bundle(Path(name))
            _tweak(protocol, "corpus_sha256", value=None)
            with self.assertRaises(ValueError):
                verify_scaling_bundle(corpus, protocol)

    def test_missing_frozen_at_refuses(self) -> None:
        """A missing frozen_at timestamp refuses the bundle."""
        with tempfile.TemporaryDirectory() as name:
            corpus, protocol = _bundle(Path(name))
            _tweak(protocol, "evaluation", "frozen_at", value=None)
            with self.assertRaises(ValueError):
                verify_scaling_bundle(corpus, protocol)

    def test_volume_and_key_refusals(self) -> None:
        """Volume, quota, and key drift each refuse the bundle."""
        with tempfile.TemporaryDirectory() as name:
            corpus, protocol = _bundle(Path(name))
            _tweak(protocol, "evaluation", "declared_volume", value=3)
            with self.assertRaises(ValueError):
                verify_scaling_bundle(corpus, protocol)
            _tweak(protocol, "evaluation", "declared_volume", value=2)
            _tweak(protocol, "quotas", "eval", value=3)
            with self.assertRaises(ValueError):
                verify_scaling_bundle(corpus, protocol)
            data = json.loads(protocol.read_text())
            del data["evaluation"]["beam_width"]
            protocol.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                verify_scaling_bundle(corpus, protocol)

    def test_generator_source_refusals(self) -> None:
        """Null or mismatched generator hashes refuse at run time."""
        with tempfile.TemporaryDirectory() as name:
            corpus, protocol = _bundle(Path(name))
            bad = {"rsi_programmer/construction_scaling_final.py": None}
            _tweak(protocol, "source_provenance", "generator_sha256",
                   value=dict(bad))
            with self.assertRaises(ValueError):
                verify_scaling_bundle(corpus, protocol)
            bad = {"rsi_programmer/construction_scaling_final.py": "0" * 64}
            _tweak(protocol, "source_provenance", "generator_sha256",
                   value=dict(bad))
            with self.assertRaises(ValueError):
                verify_scaling_bundle(corpus, protocol)

    def test_null_budget_refuses(self) -> None:
        """An unfrozen per-task budget refuses before inference."""
        self.assertRaises(ValueError, require_scaling_budget,
                          {"evaluation": _evaluation(
                              per_task_cpu_seconds=None)})

    def test_fixture_contract_and_refusal(self) -> None:
        """The shipped fixture matches the contract; runner refuses."""
        root = Path(__file__).resolve().parents[1]
        fixture = root / "fixtures/construction_scaling_v1.json"
        protocol = json.loads(fixture.read_text())
        for key in REQUIRED_KEYS:
            self.assertIn(key, protocol["evaluation"])
        self.assertEqual(protocol["decision_rules"]["kill_criterion"],
                         KILL)
        self.assertEqual(protocol["quotas"], {"train": 4800,
                                             "eval": 12000})
        self.assertEqual(protocol["evaluation"]["declared_volume"], 12000)
        self.assertIsNone(protocol["corpus_sha256"])
        missing = root / "fixtures/construction_scaling_corpus_v1.json"
        with self.assertRaises(ValueError):
            verify_scaling_bundle(missing, fixture)

    def test_tiny_end_to_end(self) -> None:
        """One seed, tiny knobs: verdict shape and field types."""
        report = _tiny_report()
        self.assertIsInstance(report["verdict"], bool)
        self.assertIsInstance(report["decision"]["passed"], bool)
        self.assertEqual(report["decision"]["kill_criterion"], KILL)
        self.assertGreater(report["decision"]["common_budget_cpu_seconds"],
                            0.0)
        self.assertEqual(set(report["means"]),
                         {"learned", "untrained", "streaming"})
        self.assertEqual(set(report["strata"]),
                         {"learned", "untrained", "streaming"})
        self.assertEqual(len(report["seeds"]), 1)
        self.assertEqual(set(report["seeds"][0]["arms"]),
                         {"learned", "untrained"})
        self.assertIsInstance(report["worker_cpu_seconds"], float)

    def test_tiny_rows_shape(self) -> None:
        """Eval rows keep the hard report's record fields."""
        report = _tiny_report()
        for arm in ("learned", "untrained"):
            rows = report["seeds"][0]["arms"][arm]["rows"]
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(set(row), {"id", "accepted",
                                            "cpu_seconds", "executions",
                                            "error", "stratum"})
        self.assertEqual(len(report["streaming"]["rows"]), 2)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
