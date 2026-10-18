"""Frozen inputs, asymmetric accounting and isolated memory contracts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_programmer import benchmark_bounded as run
from rsi_programmer.benchmark_bounded_summary import (
    configurations,
    summarize,
    training_charge,
)
from rsi_programmer.benchmark_v2_runner import load_frozen_inputs
from scripts.benchmark_bounded_candidates import main


class BoundedBenchmarkTests(unittest.TestCase):
    def test_real_frozen_preflight_and_complete_asymmetric_grid(self) -> None:
        protocol = run.load_protocol()
        _, corpus = load_frozen_inputs()
        keys = run.expected_keys(protocol, corpus)
        self.assertEqual(len(keys), 10656)
        self.assertEqual(len([k for k in keys if k[2] == "streaming"]), 1440)
        self.assertTrue(any(k[4] == 524288 for k in keys))
        self.assertFalse(
            any(k[2] == "bounded" and k[3] == "streaming" for k in keys)
        )
        self.assertEqual(
            run.baseline_search(protocol).__module__, "_cached_v71"
        )

    def test_protocol_corruption_rejected_before_loading_training_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            path.write_text(run.PROTOCOL_PATH.read_text() + " ")
            with patch.object(run, "load_frozen_inputs") as load:
                with self.assertRaisesRegex(ValueError, "protocol hash"):
                    run.load_protocol(path)
                load.assert_not_called()

    def test_cli_preflight_does_not_create_measurement_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "unused"
            self.assertEqual(
                main(["--check-only", "--output", str(output)]), 0
            )
            self.assertFalse(output.exists())

    def test_engine_pair_order_is_counterbalanced(self) -> None:
        for seed in range(3):
            first = run.engine_order(seed, "task", 8192, "mlp", 0)
            second = run.engine_order(seed, "task", 8192, "mlp", 1)
            self.assertEqual(first, second[::-1])
            self.assertEqual(set(first), {"full_cached", "bounded"})

    def test_existing_output_is_preserved_on_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            sentinel = output / "raw.jsonl"
            sentinel.write_text("retained evidence\n")
            self.assertEqual(main(["--output", str(output)]), 2)
            self.assertEqual(sentinel.read_text(), "retained evidence\n")
            self.assertFalse((output / "failure.json").exists())

    def test_learned_policy_can_score_in_isolated_memory_child(self) -> None:
        from rsi_programmer.benchmark_mlp import MLPControl

        _, corpus = load_frozen_inputs()
        task = corpus["splits"]["eval"][0]
        policy = MLPControl(seed=0, lr=0.05)
        before = run.v1._state_hash(policy)
        for mode in ("rss", "tracemalloc"):
            row = run.measure_memory(
                "bounded",
                run.bounded_pre_evaluation_search,
                policy,
                task,
                1,
                mode,
            )
            self.assertEqual(row["root_calls"], 1)
            self.assertTrue(row["state_frozen_during_eval"])
            self.assertEqual(row["work"]["feature_builds"], 12)
        self.assertEqual(before, run.v1._state_hash(policy))

    def test_full_setup_charge_and_censoring_are_not_erased(self) -> None:
        protocol = run.load_protocol()
        controls = protocol["evaluation"]["ranked_controls"]
        training = {
            "training_group_cpu_ns": 1200,
            "training_group_wall_ns": 1500,
            "seeds": [
                {
                    "seed": seed,
                    "control_construction": {
                        c: {"cpu_ns": 30, "wall_ns": 40} for c in controls
                    },
                    "controls": {
                        c: {"fit_cpu_ns": 90, "fit_wall_ns": 110}
                        for c in controls
                    },
                }
                for seed in range(3)
            ],
        }
        rows = []
        for engine, control, budgets in configurations(protocol):
            for budget in budgets:
                for repeat in range(2):
                    rows.append(
                        {
                            "engine": engine,
                            "control": control,
                            "budget": budget,
                            "repeat": repeat,
                            "accepted": engine != "bounded",
                            "censored": True,
                            "work": {
                                "scored_descriptors": 40,
                                "signature_evaluations": 10,
                            },
                            ("attempt_cpu_ns_including_hashes_"
                             "and_validation"): 100,
                            ("attempt_wall_ns_including_hashes_"
                             "and_validation"): 200,
                        }
                    )
        summary = summarize(rows, training, protocol)
        self.assertEqual(training_charge(training, "mlp", "cpu"), 1320)
        self.assertEqual(training_charge(training, "streaming", "cpu"), 0)
        bounded = [
            p
            for p in summary["matched_success_cpu_cost"]
            if p["engine"] == "bounded"
        ]
        self.assertTrue(all(p["status"] == "censored" for p in bounded))
        self.assertTrue(
            all(p["censored_at_budget"] == 131072 for p in bounded)
        )
        self.assertFalse(summary["decision_supported_on_cost"])
        self.assertFalse(summary["snn_efficiency_supported_on_cost"])
        mlp = next(
            p
            for p in summary["amortization"]
            if p["engine"] == "full_cached"
            and p["control"] == "mlp"
            and p["independent_task_count"] == 48
        )
        self.assertEqual(mlp["amortized_cpu_ns_per_task"], 127.5)
        self.assertEqual(summary["points"][0]["scan_amplification"], 4)

    def test_fresh_memory_child_keeps_instrumented_results_separate(
        self,
    ) -> None:
        protocol = run.load_protocol()
        _, corpus = load_frozen_inputs()
        task = corpus["splits"]["eval"][0]
        baseline = run.baseline_search(protocol)
        for mode in ("rss", "tracemalloc"):
            row = run.measure_memory(
                "streaming", baseline, None, task, 1, mode
            )
            self.assertTrue(row["instrumented_timing_not_used"])
            self.assertGreaterEqual(row["peak_rss_delta_kib"], 0)
            self.assertGreaterEqual(
                row["peak_rss_kib"], row["initial_rss_kib"]
            )
            self.assertEqual(row["root_calls"], 1)
            if mode == "rss":
                self.assertIsNone(row["traced_peak_bytes"])
            else:
                self.assertGreater(row["traced_peak_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
