"""Frozen inputs, asymmetric accounting and isolated memory contracts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any
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
    def test_real_training_startup_stops_before_any_evaluation(self) -> None:
        class StartupComplete(Exception):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "startup"
            protocol = run.load_protocol()
            _, corpus = load_frozen_inputs()
            frozen_tasks = corpus["splits"]["train"]
            self.assertEqual(len(frozen_tasks), 24)
            original_group_builder = run.build_training_groups
            group_calls = []

            def build_groups_spy(
                tasks: list[dict[str, Any]], **kwargs: Any
            ) -> tuple[list[Any], dict[str, Any]]:
                built_groups = original_group_builder(tasks, **kwargs)
                group_calls.append((tasks, kwargs, built_groups))
                return built_groups

            original_fit = run.work._make_models_with_cpu
            fit_calls = []

            def fit_spy(
                seed: int,
                groups: list[list[tuple[list[float], int]]],
                epochs: int,
                expected_weights: dict[str, str | None],
            ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
                pinned = protocol["model_reference"][
                    "expected_weights_after_fit"
                ][str(seed)]
                fit_calls.append({
                    "seed": seed,
                    "groups": groups,
                    "epochs": epochs,
                    "expected_weights": dict(expected_weights),
                })
                self.assertEqual(expected_weights, pinned)
                portable_expected = dict(expected_weights)
                for name in ("trained_snn", "untrained_snn", "mlp",
                             "recurrent"):
                    portable_expected[name] = None
                return original_fit(seed, groups, epochs, portable_expected)

            with patch.object(
                run, "build_training_groups", side_effect=build_groups_spy
            ), patch.object(
                run.work, "_make_models_with_cpu", side_effect=fit_spy
            ), patch.object(
                run, "measure_memory", side_effect=StartupComplete
            ) as memory:
                with self.assertRaises(StartupComplete):
                    run.run(output)
            self.assertEqual(len(group_calls), 1)
            actual_tasks, caps, built_groups = group_calls[0]
            self.assertEqual(actual_tasks, frozen_tasks)
            self.assertEqual(
                [task["id"] for task in actual_tasks],
                [task["id"] for task in frozen_tasks],
            )
            self.assertEqual(caps, {
                "positive_cap": protocol["training"]["positive_cap"],
                "negative_cap": protocol["training"]["negative_cap"],
                "sampling_seed": protocol["training"]["sampling_seed"],
            })
            memory.assert_called_once()
            checkpoint = json.loads((output / "training.json").read_text())
            self.assertEqual(len(checkpoint["training_groups"]["groups"]), 24)
            seed = checkpoint["seeds"][0]
            self.assertEqual(seed["seed"], 0)
            self.assertEqual(len(fit_calls), 1)
            self.assertEqual(fit_calls[0]["seed"], 0)
            self.assertIs(fit_calls[0]["groups"], built_groups[0])
            self.assertEqual(fit_calls[0]["epochs"], 40)
            for control in ("trained_snn", "mlp", "recurrent"):
                record = seed["controls"][control]
                self.assertTrue(record["fit_performed"])
                self.assertTrue(record["fit_changed_weights"])
                self.assertEqual(record["fit_work"]["optimizer_updates"], 40)
            self.assertEqual(seed["controls"]["untrained_snn"][
                "weights_before_fit_sha256"],
                seed["controls"]["untrained_snn"][
                    "weights_after_fit_sha256"])
            self.assertFalse((output / "memory.jsonl").exists())
            self.assertFalse((output / "raw.jsonl").exists())

    def test_model_builder_rejects_wrong_post_fit_hash(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "trained_snn post-fit weights differ"
        ):
            run.work._make_models_with_cpu(
                seed=0,
                groups=[[
                    ([0.0] * 12, 1),
                    ([1.0] * 12, 0),
                ]],
                epochs=1,
                expected_weights={"trained_snn": "deliberately-wrong"},
            )

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

    def test_unavailable_peak_counters_remain_null_in_summary(self) -> None:
        task = {"id": "sample"}
        # Match _attempt's actual shape: search metrics plus work-vector axes.
        attempt_row = {
            "work": {"scored_descriptors": 0},
            "search": {"scored_descriptors": 0},
            "root_calls": 0,
        }
        for engine in ("full_cached", "streaming"):
            with patch.object(run.work, "_attempt", return_value=attempt_row):
                attempted = run.attempt(engine, lambda: None, None, task, 8)
            self.assertIsNone(attempted["work"]["peak_live_descriptors"])
            self.assertIsNone(attempted["work"]["peak_live_features"])
            self.assertEqual(attempted["work"]["skipped_descriptors"], 0)
            self.assertEqual(
                attempted["work"]["scored_unexecuted_descriptors"], 0
            )

        protocol = run.load_protocol()
        controls = protocol["evaluation"]["ranked_controls"]
        training = {
            "training_group_cpu_ns": 0,
            "training_group_wall_ns": 0,
            "seeds": [{
                "control_construction": {
                    name: {"cpu_ns": 0, "wall_ns": 0} for name in controls
                },
                "controls": {
                    name: {"fit_cpu_ns": 0, "fit_wall_ns": 0}
                    for name in controls
                },
            }],
        }
        streaming_budget = protocol["evaluation"]["streaming_budgets"][0]
        ranked_budget = protocol["evaluation"]["ranked_budgets"][0]
        records = [
            {
                "engine": "streaming", "control": "streaming",
                "budget": streaming_budget, "repeat": repeat,
                "accepted": False, "censored": True,
                "work": {
                    "signature_evaluations": 0,
                    "scored_descriptors": 0,
                    "skipped_descriptors": 0,
                    "scored_unexecuted_descriptors": 0,
                    "peak_live_descriptors": None,
                    "peak_live_features": None,
                },
                "attempt_cpu_ns_including_hashes_and_validation": 1,
                "attempt_wall_ns_including_hashes_and_validation": 1,
            }
            for repeat in (0, 1)
        ]
        records.extend([
            {
                "engine": "bounded", "control": controls[0],
                "budget": ranked_budget, "repeat": repeat,
                "accepted": False, "censored": True,
                "work": {
                    "signature_evaluations": 1,
                    "scored_descriptors": 3,
                    "skipped_descriptors": 2,
                    "scored_unexecuted_descriptors": 2,
                    "peak_live_descriptors": peak,
                    "peak_live_features": peak - 1,
                },
                "attempt_cpu_ns_including_hashes_and_validation": 1,
                "attempt_wall_ns_including_hashes_and_validation": 1,
            }
            for repeat, peak in ((0, 7), (1, 5))
        ])
        summary = summarize(records, training, protocol)
        stream_point = next(
            point for point in summary["points"]
            if point["engine"] == "streaming"
            and point["budget"] == streaming_budget
            and point["repeat"] is None
        )
        bounded_point = next(
            point for point in summary["points"]
            if point["engine"] == "bounded"
            and point["budget"] == ranked_budget
            and point["repeat"] is None
        )
        self.assertIsNone(
            stream_point["work_totals_by_axis"]["peak_live_descriptors"]
        )
        self.assertIsNone(
            stream_point["work_totals_by_axis"]["peak_live_features"]
        )
        self.assertEqual(
            stream_point["work_totals_by_axis"]["scored_descriptors"], 0
        )
        self.assertEqual(
            stream_point["work_totals_by_axis"]["skipped_descriptors"], 0
        )
        self.assertEqual(
            stream_point["work_totals_by_axis"][
                "scored_unexecuted_descriptors"
            ], 0
        )
        self.assertEqual(
            bounded_point["work_totals_by_axis"]["peak_live_descriptors"], 12
        )
        self.assertEqual(
            bounded_point["work_totals_by_axis"]["peak_live_features"], 10
        )

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
