"""Sealed hard-bank final: e2e, refusals, and verdict units.

Synthetic corpora only, never real eval data. The absent W3 loop is
stubbed through a `sys.modules` fake for
`rsi_programmer.construction_feedback` running one honest epoch.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import torch

from rsi_programmer.construction_hard_final import (
    FEEDBACK_KEYS, REQUIRED_KEYS, decide_hard, evaluate_arm,
    require_hard_budget, run_hard_seed, summarize_hard,
    verify_hard_bundle)
from rsi_programmer.construction_policy import ConstructionPolicy
from rsi_programmer.online_graph import graph_polynomial
from rsi_programmer.probe import Expr, lower_to_dag

PROGRAMS = {"add": lambda x, y: x + y, "mul": lambda x, y: x * y}
KILL = "feedback must beat both arms and survive equal CPU"


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


def _bundle(directory: Path, cpus: Any = 5.0) -> tuple[Path, Path]:
    """Write a synthetic corpus and protocol for tests."""
    train = [_task("train-add", "train", "add", 5)]
    final = [_task("eval-mul", "eval", "mul", 5),
             _task("eval-add", "eval", "add", 6)]
    corpus = {"splits": {"train": train, "dev": [], "eval": final}}
    raw = (json.dumps(corpus, sort_keys=True) + "\n").encode()
    corpus_path = directory / "corpus.json"
    corpus_path.write_bytes(raw)
    evaluation = {"model_seeds": [0, 1, 2],
                  "per_task_cpu_seconds": cpus, "beam_width": 2,
                  "bound": 4, "imitation_epochs": 1,
                  "feedback": {"rounds": 1, "epochs_per_round": 1,
                               "rollout_budget": 4}}
    protocol = {"corpus_sha256": hashlib.sha256(raw).hexdigest(),
                "source_provenance": {"generator_sha256": {}},
                "evaluation": evaluation,
                "decision_rules": {"kill_criterion": KILL}}
    (directory / "protocol.json").write_text(json.dumps(protocol))
    return corpus_path, directory / "protocol.json"


def _install_feedback_fake() -> None:
    """Stub the absent W3 loop with one honest imitation epoch."""
    from rsi_programmer.construction_training import fit_constructor
    fake = types.ModuleType("rsi_programmer.construction_feedback")

    def fit_with_feedback(policy: ConstructionPolicy,
                          tasks: list[dict[str, Any]], *args: Any,
                          **kwargs: Any) -> dict[str, Any]:
        _ = args, kwargs
        fit = fit_constructor(policy, tasks, 1)
        return {"rounds": [{"round": 0, "accepted": 1}],
                "total_rollout_cpu_seconds": 0.001,
                "total_fit_cpu_seconds": fit["complete_cpu_seconds"],
                "stopped_zero_acceptance": False}

    fake.fit_with_feedback = fit_with_feedback  # type: ignore
    sys.modules["rsi_programmer.construction_feedback"] = fake


class RealFixtureTests(unittest.TestCase):
    """The shipped W1 fixture must satisfy this runner's contract."""

    def test_real_fixture_keys_and_frozen_budget(self) -> None:
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads(
            (root / "fixtures/construction_hard_v1.json").read_text())
        evaluation = protocol["evaluation"]
        for key in REQUIRED_KEYS:
            self.assertIn(key, evaluation)
        for key in FEEDBACK_KEYS:
            self.assertIn(key, evaluation["feedback"])
        self.assertIn("kill_criterion", protocol["decision_rules"])
        self.assertEqual(require_hard_budget(protocol), 0.5)
        unfrozen = json.loads(json.dumps(protocol))
        unfrozen["evaluation"]["per_task_cpu_seconds"] = None
        with self.assertRaises(ValueError):
            require_hard_budget(unfrozen)


class HardSealTests(unittest.TestCase):
    """Hash, budget, split, and weight guards refuse bad final runs."""

    def test_matching_hash_loads_synthetic_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            corpus, protocol = _bundle(Path(name))
            bundle = verify_hard_bundle(corpus, protocol)
            self.assertEqual(len(bundle["corpus"]["splits"]["eval"]), 2)

    def test_tampered_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            corpus, protocol = _bundle(Path(name))
            corpus.write_bytes(corpus.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                verify_hard_bundle(corpus, protocol)

    def test_train_tasks_never_enter_inference(self) -> None:
        task = _task("t", "train", "add", 5)
        with self.assertRaises(ValueError):
            evaluate_arm([task], "streaming", None, 5.0, 4, 2, 0)

    def test_weight_change_during_eval_is_rejected(self) -> None:
        import rsi_programmer.construction_hard_final as hard
        real = hard.beam_construct

        def mutating(policy: Any, observed: Any, **kwargs: Any) -> Any:
            with torch.no_grad():
                next(policy.parameters()).add_(1.0)
            return real(policy, observed, **kwargs)

        patched = mock.patch.object(hard, "beam_construct",
                                    side_effect=mutating)
        with patched, self.assertRaises(ValueError):
            policy = ConstructionPolicy(seed=0, backend="ann")
            evaluate_arm([_task("f", "eval", "add", 5)], "imitation",
                         policy, 5.0, 4, 2, 0)


class HardVerdictTests(unittest.TestCase):
    """The frozen kill rule passes once and kills in both shapes."""

    def test_pass_beats_both_arms_and_survives_cpu(self) -> None:
        verdict = decide_hard(3.0, 1.0, 2.0, {"kill": False}, KILL)
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["kill_criterion"], KILL)

    def test_kill_when_feedback_ties_a_baseline(self) -> None:
        verdict = decide_hard(2.0, 1.0, 2.0, {"kill": False}, KILL)
        self.assertFalse(verdict["passed"])

    def test_kill_when_equal_cpu_truncation_loses(self) -> None:
        from rsi_programmer.cpu_matched_verdict import compare_pair
        learned = [[(True, 1.0), (False, 1.0)]]
        base = [[(True, 0.5), (True, 0.5)]]
        comparison = compare_pair(learned, base)
        self.assertTrue(comparison["kill"])
        verdict = decide_hard(3.0, 1.0, 2.0, comparison, KILL)
        self.assertFalse(verdict["passed"])
        self.assertFalse(verdict["survives_equal_cpu"])


class HardRunnerTests(unittest.TestCase):
    """Runner refusals and a smoke-scale all-arms end-to-end run."""

    def setUp(self) -> None:
        _install_feedback_fake()
        key = "rsi_programmer.construction_feedback"
        self.addCleanup(sys.modules.pop, key, None)

    def _run(self, argv: list[str]) -> int:
        from scripts.run_construction_hard_final import main
        with mock.patch.object(sys, "argv", argv):
            return main()

    def _bundle_argv(self, root: Path, cpus: Any = 5.0
                     ) -> tuple[list[str], Path, Path, Path]:
        """Bundle synthetics with argv pointing at a fresh output."""
        corpus, protocol = _bundle(root, cpus)
        receipt = root / "receipt.json"
        receipt.write_text(json.dumps({"recipe": "smoke"}))
        argv = ["run", "--frozen-receipt", str(receipt), "--corpus",
                str(corpus), "--protocol", str(protocol), "--output",
                str(root / "out")]
        return argv, root / "out", protocol, receipt

    def _check_report(self, output: Path, protocol: Path, receipt: Path
                      ) -> None:
        """Assert the frozen report shape and digests."""
        report = json.loads((output / "report.json").read_text())
        arms = ("streaming", "untrained", "imitation", "feedback")
        self.assertIn(report["verdict"], (True, False))
        self.assertEqual(set(report["means"]), set(arms))
        self.assertTrue(report["strata"]["feedback"])
        self.assertIn("kill_criterion", report["decision"])
        digest = json.loads(protocol.read_text())["corpus_sha256"]
        self.assertEqual(report["corpus_sha256"], digest)
        self.assertEqual(report["receipt_sha256"],
                         hashlib.sha256(receipt.read_bytes()).hexdigest())
        self.assertEqual(len(report["seeds"]), 3)

    def test_missing_receipt_refuses_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            argv, output, _, _ = self._bundle_argv(root)
            argv[2] = str(root / "absent")
            with self.assertRaises(ValueError):
                self._run(argv)
            self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            argv, output, _, _ = self._bundle_argv(root)
            output.mkdir()
            with self.assertRaises(FileExistsError):
                self._run(argv)

    def test_null_budget_refuses_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            argv, output, _, _ = self._bundle_argv(root, cpus=None)
            with self.assertRaises(ValueError):
                self._run(argv)
            self.assertFalse(output.exists())

    def test_tiny_all_arms_run_writes_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            argv, output, protocol, receipt = self._bundle_argv(root)
            self.assertEqual(self._run(argv), 0)
            self._check_report(output, protocol, receipt)

    def test_seed_run_reports_all_arms_and_w3_keys(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            corpus, protocol = _bundle(Path(name))
            bundle = verify_hard_bundle(corpus, protocol)
            splits = bundle["corpus"]["splits"]
            row = run_hard_seed(splits["train"], splits["eval"],
                                bundle["protocol"], 0)
            self.assertEqual(set(row["arms"]),
                             {"streaming", "untrained", "imitation",
                              "feedback"})
            for key in ("rounds", "total_rollout_cpu_seconds",
                        "total_fit_cpu_seconds",
                        "stopped_zero_acceptance"):
                self.assertIn(key, row["feedback_loop"])

    def test_summarize_stratifies_by_minimal_tree_ops(self) -> None:
        rows = [{"accepted": True, "cpu_seconds": 0.1, "stratum": 7},
                {"accepted": False, "cpu_seconds": 0.1, "stratum": 8}]
        arms = ("streaming", "untrained", "imitation", "feedback")
        arms_rows = {arm: {0: [dict(row) for row in rows]}
                     for arm in arms}
        costs = {arm: {0: {"fit_cpu": 0.0, "rollout_cpu": 0.0}}
                 for arm in arms}
        summary = summarize_hard(arms_rows, costs, KILL)
        self.assertEqual(summary["strata"]["feedback"], {"7": 1.0})
        self.assertIn("verdict", summary)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
