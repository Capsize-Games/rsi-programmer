"""Check timing, acceptance and source contracts for total-work reporting."""
from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from rsi_programmer import benchmark_total_work as benchmark


class TotalWorkContractTests(unittest.TestCase):
    def test_published_inputs_pass_every_guard_before_group_generation(
            self) -> None:
        path = benchmark.ROOT / "fixtures/benchmark_total_work_v1.json"
        protocol = json.loads(path.read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preflight"
            with patch.object(benchmark, "build_training_groups",
                              side_effect=RuntimeError("preflight passed")):
                with self.assertRaisesRegex(RuntimeError,
                                            "preflight passed"):
                    benchmark.run_total_work(output, protocol)
            self.assertTrue((output / "raw.jsonl").exists())
            self.assertEqual((output / "raw.jsonl").read_bytes(), b"")

    def test_attempt_timers_include_initial_and_final_hashes(self) -> None:
        events: list[str] = []
        tick = 0

        def cpu() -> int:
            nonlocal tick
            tick += 100
            events.append("cpu")
            return tick

        def state(_model: Any) -> str:
            events.append("state")
            return "fixed"

        def weights(_model: Any) -> str:
            events.append("weights")
            return "fixed"

        def search(*args: Any, **kwargs: Any) -> dict[str, Any]:
            events.append("search")
            return {"solved": False, "graph": None,
                    "signature_evaluations": 0, "budget_exhausted": True}

        task = {"train": [[1, 2, 3]], "max_tree_ops": 1}
        with patch.object(benchmark.time, "process_time_ns", cpu), \
                patch.object(benchmark.v1, "_state_hash", state), \
                patch.object(benchmark.v1, "_weights_hash", weights):
            result = benchmark._attempt("baseline", search, object(),
                                        task, 1, "synthetic")
        self.assertEqual(events[0], "cpu")
        self.assertLess(events.index("state"), events.index("search"))
        self.assertEqual(events[-1], "cpu")
        self.assertGreater(
            result["attempt_cpu_ns_including_hashes_and_validation"],
            result["search_cpu_ns"])
        self.assertTrue(result["state_frozen_during_eval"])

    def test_wrong_function_cannot_be_accepted_by_matching_rows(self) -> None:
        task = {
            "train": [[1, 0, 1], [2, 0, 2]],
            "held_out": [[3, 0, 3]], "max_tree_ops": 1,
            "generator": {"program": {"op": "input", "name": "x"}},
        }

        def search(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"solved": True, "signature_evaluations": 1,
                    "graph": {"input_types": {"x": "Int", "y": "Int"},
                              "nodes": [{"op": "add", "type": "Int",
                                         "args": ["x", "y"]}],
                              "output": 0}}

        result = benchmark._attempt("baseline", search, None, task, 1,
                                    "synthetic")
        self.assertTrue(result["validation"]["train_valid"])
        self.assertTrue(result["validation"]["held_out_valid"])
        self.assertFalse(result["validation"]["exact_function_match"])
        self.assertFalse(result["accepted"])

    def test_baseline_source_drift_is_rejected_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.py"
            path.write_text("raise AssertionError('must not execute')\n")
            with self.assertRaisesRegex(ValueError, "source hash"):
                benchmark.load_baseline_search(path)

    def test_source_manifest_covers_feature_and_model_dependencies(
            self) -> None:
        hashes = benchmark._source_hashes(benchmark.BASELINE_PATH)
        paths = hashes["source_files_sha256"]
        for relative in (
                "rsi_programmer/pre_eval_features.py",
                "rsi_programmer/policy.py", "rsi_programmer/probe.py",
                "rsi_programmer/benchmark_mlp.py",
                "rsi_programmer/benchmark_recurrent.py",
                "benchmarks/reference/pre_eval_code_v65.py"):
            self.assertIn(relative, paths)

    def test_amortization_averages_seeds_and_charges_full_setup(self) -> None:
        records = []
        training = []
        for seed in (0, 1, 2):
            training.append({
                "seed": seed,
                "training_group_cpu_ns": 480 if seed == 0 else 0,
                "training_group_wall_ns": 960 if seed == 0 else 0,
                "control_construction": {
                    name: {"cpu_ns": 10 * (seed + 1),
                           "wall_ns": 20 * (seed + 1)}
                    for name in benchmark.CONTROL_NAMES},
                "controls": {
                    name: {"fit_cpu_ns": 100 * (seed + 1),
                           "fit_wall_ns": 200 * (seed + 1)}
                    for name in benchmark.CONTROL_NAMES},
            })
            for engine in benchmark.ENGINE_NAMES:
                for name in benchmark.CONTROL_NAMES:
                    for repeat in (0, 1):
                        records.append({
                            "seed": seed, "task_id": "synthetic",
                            "engine": engine, "control": name,
                            "budget": 8192, "repeat": repeat,
                            "accepted": True, "censored": False,
                            "work": {"signature_evaluations": 7},
                            "attempt_cpu_ns_including_hashes_and_validation":
                                100,
                            "attempt_wall_ns_including_hashes_and_validation":
                                200,
                        })
        protocol = {
            "evaluation": {"candidate_scan_budgets": [8192]},
            "aggregation": {"matched_success_targets": [0.5]},
            "amortization": {"task_counts": [48]},
        }
        summary = benchmark._summarize(records, protocol, training)
        costs = {item["control"]: item for item in summary["amortization"]
                 if item["engine"] == "optimized"}
        for name in ("trained_snn", "mlp", "recurrent"):
            self.assertAlmostEqual(costs[name]["amortized_cpu_ns_per_task"],
                                   100 + (480 + 220) / 48)
        self.assertEqual(costs["streaming"]["amortized_cpu_ns_per_task"], 100)


if __name__ == "__main__":
    unittest.main()
