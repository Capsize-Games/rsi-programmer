"""Contracts for the frozen complete-fallback shortlist benchmark."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

from rsi_programmer import benchmark_shortlist as benchmark
from rsi_programmer import benchmark_shortlist_summary as summary
from rsi_programmer import benchmark_total_work as work
from rsi_programmer.benchmark_v2_runner import load_frozen_inputs
from rsi_programmer.shortlist_pre_eval import (
    shortlist_pre_evaluation_search,
)


class BenchmarkShortlistTests(unittest.TestCase):
    def test_final_protocol_preflight_has_complete_declared_matrix(
            self) -> None:
        protocol = benchmark.load_protocol()
        _, corpus = load_frozen_inputs()
        self.assertEqual(len(benchmark.expected_keys(protocol, corpus)), 11808)
        self.assertEqual(
            len(protocol["memory"]["modes"])
            * len(corpus["splits"]["eval"])
            * len(benchmark.configurations(protocol)),
            960,
        )
        selection = benchmark.load_selection_evidence(protocol)
        self.assertEqual(selection["total_cpu_ns"], 56792751000)
        self.assertEqual(selection["total_wall_ns"], 55770510319)
        self.assertEqual(selection["internal_total_cpu_ns"], 54729696495)
        self.assertEqual(selection["internal_total_wall_ns"], 54755978901)

    def test_protocol_rejects_corrupt_selection_receipt_and_bundle(self
                                                                  ) -> None:
        protocol = benchmark.load_protocol()
        bad_receipt = copy.deepcopy(protocol)
        bad_receipt["selection_evidence"]["execution_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            benchmark.load_selection_evidence(bad_receipt)

        bad_bundle = copy.deepcopy(protocol)
        bad_bundle["selection_evidence"]["files_sha256"][
            "training.json"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash map"):
            benchmark.load_selection_evidence(bad_bundle)

        bad_method = copy.deepcopy(protocol)
        bad_method["method"]["shortlist_size"] = 8
        with self.assertRaisesRegex(ValueError, "selected method"):
            benchmark.load_selection_evidence(bad_method)

    def test_actual_startup_builds_exact_groups_and_fits_seed_zero(self
                                                                  ) -> None:
        class StopBeforeEvaluation(Exception):
            """Stop after fitting, before the first evaluation search."""

        protocol = benchmark.load_protocol()
        _, corpus = load_frozen_inputs()
        expected_tasks = corpus["splits"]["train"]
        original_groups = benchmark.build_training_groups
        original_fit = work._make_models_with_cpu
        observed: dict[str, Any] = {}

        def record_groups(tasks: list[dict[str, Any]], **kwargs: Any
                          ) -> tuple[list[Any], dict[str, Any]]:
            observed["tasks"] = copy.deepcopy(tasks)
            observed["group_kwargs"] = kwargs
            return original_groups(tasks, **kwargs)

        def portable_fit(seed: int, groups: list[Any], epochs: int,
                         expected_weights: dict[str, str]
                         ) -> tuple[dict[str, Any], dict[str, Any],
                                    dict[str, Any]]:
            pinned = protocol["model_reference"][
                "expected_weights_after_fit"][str(seed)]
            observed["expected_weights_passed"] = copy.deepcopy(
                expected_weights)
            if expected_weights != pinned:
                raise AssertionError("runner changed pinned model map")
            relaxed = copy.deepcopy(expected_weights)
            for name in ("trained_snn", "untrained_snn", "mlp", "recurrent"):
                relaxed[name] = None
            models, fitted, construction = original_fit(
                seed, groups, epochs, relaxed)
            observed["trained_weight_hashes"] = {
                name: work.v1._weights_hash(model)
                for name, model in models.items()
            }
            return models, fitted, construction

        with tempfile.TemporaryDirectory(
                prefix="shortlist-startup-") as directory:
            output = Path(directory) / "run"
            with patch.object(benchmark, "build_training_groups",
                              side_effect=record_groups), patch.object(
                                  work, "_make_models_with_cpu",
                                  side_effect=portable_fit), patch.object(
                                      benchmark, "measure_memory",
                                      side_effect=StopBeforeEvaluation):
                with self.assertRaises(StopBeforeEvaluation):
                    benchmark.run(output)
            saved = json.loads((output / "training.json").read_text())
            self.assertFalse((output / "raw.jsonl").exists())
            self.assertFalse((output / "memory.jsonl").exists())

        self.assertEqual(observed["tasks"], expected_tasks)
        self.assertEqual(observed["group_kwargs"], {
            "positive_cap": protocol["training"]["positive_cap"],
            "negative_cap": protocol["training"]["negative_cap"],
            "sampling_seed": protocol["training"]["sampling_seed"],
        })
        self.assertEqual(len(saved["training_groups"]["groups"]), 24)
        self.assertEqual(saved["selection"]["total_cpu_ns"], 56792751000)
        for name in ("trained_snn", "mlp", "recurrent"):
            fitted = saved["seeds"][0]["controls"][name]
            self.assertEqual(
                observed["trained_weight_hashes"][name],
                fitted["weights_after_fit_sha256"],
            )
        for name in ("trained_snn", "mlp", "recurrent"):
            fitted = saved["seeds"][0]["controls"][name]
            self.assertEqual(fitted["fit_work"]["optimizer_updates"], 40)
            self.assertTrue(fitted["fit_changed_weights"])
            self.assertNotEqual(
                fitted["weights_before_fit_sha256"],
                fitted["weights_after_fit_sha256"],
            )

    def test_real_cheap_only_attempt_passes_metric_guard_and_memory_fork(
            self) -> None:
        protocol = benchmark.load_protocol()
        _, corpus = load_frozen_inputs()
        task = corpus["splits"]["train"][0]
        method = {"window_size": 4, "shortlist_size": 2}
        fn = partial(shortlist_pre_evaluation_search,
                     window_size=4, shortlist_size=2, cheap_only=True)
        row = benchmark.attempt("cheap_only", fn, None, task, 1)
        self.assertTrue(benchmark._shortlist_metrics_valid(row, method))

        memory = benchmark.measure_memory(
            "cheap_only", fn, None, task, 1, "rss")
        self.assertEqual(memory["memory_mode"], "rss")
        self.assertTrue(memory["instrumented_timing_not_used"])
        self.assertTrue(benchmark._shortlist_metrics_valid(memory, method))

    def test_metric_guard_rejects_bad_counter_partitions(self) -> None:
        protocol = benchmark.load_protocol()
        _, corpus = load_frozen_inputs()
        task = corpus["splits"]["train"][0]

        method = {"window_size": 4, "shortlist_size": 2}
        fn = partial(shortlist_pre_evaluation_search,
                     window_size=4, shortlist_size=2)
        ranked = benchmark.attempt(
            "complete_shortlist", fn, work.ClassicalHeuristicPolicy(),
            task, 1)
        self.assertTrue(benchmark._shortlist_metrics_valid(ranked, method))
        for key, value in (
                ("cheap_operand_value_lookups", 999),
                ("fallback_executions", 999),
                ("permanently_pruned_descriptors", 1),
                ("peak_live_cheap_scores", 999)):
            bad = copy.deepcopy(ranked)
            bad["search"][key] = value
            self.assertFalse(benchmark._shortlist_metrics_valid(bad, method))

    def test_summary_charges_selection_only_to_new_methods(self) -> None:
        protocol = {
            "evaluation": {
                "streaming_budgets": [1], "ranked_budgets": [1],
                "ranked_controls": ["heuristic", "trained_snn", "mlp",
                                    "recurrent"],
                "ranked_engines": ["full_cached", "complete_shortlist"],
                "task_count": 1,
            },
            "training": {"seeds": [0]},
            "amortization": {"task_counts": [48]},
            "aggregation": {"matched_success_targets": [1.0]},
        }
        training = {
            "training_group_cpu_ns": 100,
            "training_group_wall_ns": 200,
            "selection": {"total_cpu_ns": 500, "total_wall_ns": 700},
            "seeds": [{
                "control_construction": {
                    name: {"cpu_ns": 30, "wall_ns": 50}
                    for name in ("trained_snn", "mlp", "recurrent")
                },
                "controls": {
                    name: {"fit_cpu_ns": 70, "fit_wall_ns": 100}
                    for name in ("trained_snn", "mlp", "recurrent")
                },
            }],
        }
        records = _summary_records(protocol)
        result = summary.summarize(records, training, protocol)
        charges = {
            (row["engine"], row["control"]): row
            for row in result["amortization"]
            if row["independent_task_count"] == 48
        }
        self.assertEqual(charges["streaming", "streaming"][
            "selection_cpu_ns"], 0)
        self.assertEqual(charges["full_cached", "heuristic"][
            "selection_cpu_ns"], 0)
        self.assertEqual(charges["complete_shortlist", "heuristic"][
            "selection_cpu_ns"], 500)
        self.assertEqual(charges["cheap_only", "cheap_only"][
            "selection_cpu_ns"], 500)
        self.assertEqual(charges["complete_shortlist", "trained_snn"][
            "training_cpu_ns_mean_seed_including_full_setup"], 200)
        self.assertEqual(charges["complete_shortlist", "trained_snn"][
            "selection_cpu_ns"], 500)
        point = next(item for item in result["points"]
                     if item["engine"] == "complete_shortlist"
                     and item["control"] == "heuristic"
                     and item["repeat"] is None)
        self.assertIsNone(point["work_totals_by_axis"][
            "peak_live_cheap_scores"])

    def test_summary_sparse_or_censored_cells_cannot_support_decisions(
            self) -> None:
        protocol = {
            "evaluation": {
                "streaming_budgets": [1], "ranked_budgets": [1],
                "ranked_controls": ["heuristic", "trained_snn", "mlp",
                                    "recurrent"],
                "ranked_engines": ["full_cached", "complete_shortlist"],
                "task_count": 1,
            },
            "training": {"seeds": [0]},
            "amortization": {"task_counts": [48]},
            "aggregation": {"matched_success_targets": [1.0]},
        }
        training = {
            "training_group_cpu_ns": 0, "training_group_wall_ns": 0,
            "selection": {"total_cpu_ns": 0, "total_wall_ns": 0},
            "seeds": [{
                "control_construction": {
                    name: {"cpu_ns": 0, "wall_ns": 0}
                    for name in ("trained_snn", "mlp", "recurrent")
                },
                "controls": {
                    name: {"fit_cpu_ns": 0, "fit_wall_ns": 0}
                    for name in ("trained_snn", "mlp", "recurrent")
                },
            }],
        }
        rows = _summary_records(protocol)
        removed = next(index for index, row in enumerate(rows)
                       if row["engine"] == "complete_shortlist"
                       and row["control"] == "heuristic"
                       and row["repeat"] == 0)
        sparse = rows[:removed] + rows[removed + 1:]
        sparse_result = summary.summarize(sparse, training, protocol)
        sparse_match = next(item for item in sparse_result[
            "matched_success_cpu_cost"]
            if item["engine"] == "complete_shortlist"
            and item["control"] == "heuristic" and item["repeat"] == 0)
        self.assertEqual(sparse_match["status"], "censored")
        self.assertFalse(sparse_result["decision_supported_on_cost"])

        censored = copy.deepcopy(rows)
        censored[removed]["censored"] = True
        censored_result = summary.summarize(censored, training, protocol)
        censored_match = next(item for item in censored_result[
            "matched_success_cpu_cost"]
            if item["engine"] == "complete_shortlist"
            and item["control"] == "heuristic" and item["repeat"] == 0)
        self.assertEqual(censored_match["status"], "censored")
        self.assertFalse(censored_result["decision_supported_on_cost"])


def _summary_records(protocol: dict[str, Any]
                     ) -> list[dict[str, Any]]:
    """Return a complete toy matrix with explicit null peak metrics."""
    records: list[dict[str, Any]] = []
    for engine, control, budgets in summary.configurations(protocol):
        for budget in budgets:
            for repeat in (0, 1):
                records.append({
                    "seed": 0, "task_id": "tiny", "engine": engine,
                    "control": control, "budget": budget,
                    "repeat": repeat, "accepted": True, "censored": False,
                    "attempt_cpu_ns_including_hashes_and_validation": 1000,
                    "attempt_wall_ns_including_hashes_and_validation": 2000,
                    "work": {
                        "signature_evaluations": 1,
                        "scored_descriptors": 0,
                        "peak_live_descriptors": None,
                        "peak_live_features": None,
                        "peak_live_cheap_scores": None,
                    },
                })
    return records


if __name__ == "__main__":
    unittest.main()
