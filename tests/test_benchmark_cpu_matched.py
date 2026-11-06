"""Frozen-contract and CPU-budget checks for the Pursue 2 benchmark."""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer import benchmark_cpu_matched as runner
from rsi_programmer import cpu_matched_frozen as frozen
from rsi_programmer.cpu_budget import CpuBudget, CpuBudgetExceeded
from rsi_programmer.cpu_matched_verdict import (
    KILL_CRITERION,
    compare_pair,
    decide,
)

ROOT = Path(__file__).resolve().parents[1]


def _burn(seconds: float) -> None:
    """Consume CPU (not wall) time so the virtual timer can fire."""
    deadline = time.process_time() + seconds
    while time.process_time() < deadline:
        pass


class BenchmarkCpuMatchedTests(unittest.TestCase):
    def test_frozen_bytes_load_and_pin_each_other(self) -> None:
        protocol, corpus = frozen.load_frozen_inputs()
        self.assertEqual(protocol["corpus"]["sha256"],
                         frozen.CORPUS_SHA256)
        self.assertEqual(len(corpus["splits"]["train"]), 24)
        self.assertEqual(len(corpus["splits"]["eval"]), 48)

    def test_kill_criterion_verbatim_and_no_snn_arm(self) -> None:
        raw = (ROOT / "fixtures"
               / "benchmark_cpu_matched_v1.json").read_text()
        protocol = json.loads(raw)
        self.assertEqual(protocol["decision_rule"]["verbatim_kill"],
                         KILL_CRITERION)
        self.assertNotIn("snn", " ".join(protocol["arms"]))
        self.assertEqual(protocol["cost"]["cell_budget_cpu_seconds"],
                         10.0)

    def test_corpus_disjoint_from_prior_banks(self) -> None:
        _protocol, corpus = frozen.load_frozen_inputs()
        protocol = json.loads(
            (ROOT / "fixtures"
             / "benchmark_cpu_matched_v1.json").read_text())
        report = frozen.check_disjoint(corpus, protocol)
        self.assertEqual(report["overlap"], 0)
        self.assertEqual(report["new_functions"], 72)

    def test_protocol_rejects_tampered_kill_text(self) -> None:
        protocol = json.loads(
            (ROOT / "fixtures"
             / "benchmark_cpu_matched_v1.json").read_text())
        protocol["decision_rule"]["verbatim_kill"] += " (edited)"
        with self.assertRaises(ValueError):
            frozen.validate_protocol(protocol)

    def test_cpu_budget_fires_on_cpu_burn(self) -> None:
        with self.assertRaises(CpuBudgetExceeded):
            with CpuBudget(0.05):
                _burn(5.0)

    def test_cpu_budget_passes_without_burn(self) -> None:
        with CpuBudget(10.0):
            time.sleep(0.01)

    def test_run_cell_reports_budget_exceeded(self) -> None:
        def slow_search(policy: Any, rows: Any, expected: Any,
                        bound: int, budget: int,
                        task_id: str = "", **kwargs: Any
                        ) -> dict[str, Any]:
            del policy, rows, expected, bound, budget, task_id
            del kwargs
            _burn(5.0)
            return {"solved": False, "graph": None}

        task = {"id": "probe", "max_tree_ops": 5, "train": []}
        result, cpu, error = runner.run_cell(
            slow_search, None, task, 0.05, 131072)
        self.assertFalse(result.get("solved"))
        self.assertTrue(error.startswith("cpu-budget-exceeded"))
        self.assertGreaterEqual(cpu, 0.05)

    def test_equal_cpu_comparison_kills_tied_learner(self) -> None:
        learned = [[(True, 1.0), (False, 1.0)]]
        base = [[(True, 1.0), (False, 1.0)]]
        report = compare_pair(learned, base)
        self.assertTrue(report["kill"])
        self.assertEqual(report["learned_truncated_mean"],
                         report["baseline_truncated_mean"])

    def test_slow_win_over_2x_closes_line(self) -> None:
        arms = {"streaming": {"solves_mean": 1.0,
                              "charged_cpu_mean": 1.0},
                "heuristic": {"solves_mean": 0.0,
                              "charged_cpu_mean": 1.0},
                "trained_mlp": {"solves_mean": 2.0,
                                "charged_cpu_mean": 3.0}}
        comparisons = {"trained_mlp": {
            "streaming": {"kill": False},
            "heuristic": {"kill": False}}}
        verdict = decide(arms, comparisons)
        self.assertTrue(verdict["close_line"])
        self.assertEqual(verdict["line"], "CLOSE")
        self.assertIn("2x streaming's time",
                      verdict["kill_criterion"])
