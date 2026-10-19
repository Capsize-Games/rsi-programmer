"""Independently verify continual-audio report and chart aggregates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ARMS = (
    "adaptive_snn", "readout_snn", "memory_snn", "recent_memory_snn",
    "fixed_snn", "adaptive_ann", "memory_ann", "ema", "persistence",
)
NEURAL_ARMS = ARMS[:7]
MEMORY_ARMS = ("memory_snn", "recent_memory_snn", "memory_ann")


class AggregateVerifier:
    """Check pooled arithmetic and expected model work independently."""

    def __init__(self) -> None:
        self.numeric_checks = 0

    def close(self, actual: Any, expected: float, label: str) -> None:
        value = self.finite(actual, label)
        if not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"{label} mismatch: {value} != {expected}")
        self.numeric_checks += 1

    @staticmethod
    def finite(value: Any, label: str) -> float:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise ValueError(f"{label} must be finite")
        return float(value)

    @staticmethod
    def require(condition: bool, label: str) -> None:
        if not condition:
            raise ValueError(f"verification failed: {label}")

    @staticmethod
    def object(value: Any, label: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an object")
        return value

    def verify_report(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        protocol = self.object(report.get("protocol"), "protocol")
        runs_value = report.get("seed_runs")
        self.require(isinstance(runs_value, list), "seed_runs is an array")
        runs = runs_value
        self.require(protocol.get("arms") == list(ARMS), "frozen arm order")
        self.require(protocol.get("seeds") == [0, 1, 2], "frozen seeds")
        self.require(len(runs) == 3, "three seed runs")

        for seed_index, raw_run in enumerate(runs):
            run = self.object(raw_run, f"seed_runs[{seed_index}]")
            self.require(run.get("seed") == seed_index,
                         f"seed {seed_index} ordering")
            self.require(run.get("monitor_frames") == 9000,
                         f"seed {seed_index} monitor length")
            self.require(run.get("scored_frames") == 8975,
                         f"seed {seed_index} scored frame count")
            self.require(run.get("feature_target_count") == 53850,
                         f"seed {seed_index} feature target count")
            blocks = run.get("blocks")
            checkpoints = run.get("sealed_probe_checkpoints")
            self.require(isinstance(blocks, list) and len(blocks) == 30,
                         f"seed {seed_index} complete blocks")
            self.require(
                isinstance(checkpoints, list) and len(checkpoints) == 31,
                f"seed {seed_index} sealed probe checkpoints")
            cumulative = self.object(run.get("cumulative"), "cumulative")
            for arm in ARMS:
                block_records = [
                    self.object(block, "block")["arms"][arm]
                    for block in blocks
                ]
                targets = sum(item["feature_target_count"]
                              for item in block_records)
                self.require(targets == 53850,
                             f"seed {seed_index} {arm} block targets")
                sse = sum(self.finite(item["sse"], f"{arm} block SSE")
                          for item in block_records)
                record = self.object(cumulative.get(arm),
                                     f"cumulative.{arm}")
                self.close(sse, float(record["sse"]),
                           f"seed {seed_index} {arm} cumulative SSE")
                self.close(sse / 53850, float(record["mse"]),
                           f"seed {seed_index} {arm} cumulative MSE")
                for checkpoint in checkpoints:
                    checkpoint_value = self.object(checkpoint, "checkpoint")
                    probe_arms = self.object(checkpoint_value.get("arms"),
                                             "checkpoint arms")
                    probe = self.object(probe_arms.get(arm),
                                        f"checkpoint.{arm}")
                    self.require(probe.get("frame_count") == 75,
                                 f"seed {seed_index} {arm} probe frames")
                    self.require(probe.get("feature_target_count") == 450,
                                 f"seed {seed_index} {arm} probe targets")
                    self.close(float(probe["mse"]),
                               float(probe["sse"]) / 450,
                               f"seed {seed_index} {arm} probe MSE")

            self._verify_model_work(run, seed_index)
        return runs

    def _verify_model_work(self, run: dict[str, Any],
                           seed_index: int) -> None:
        model_work = self.object(run.get("model_work"), "model_work")
        work_accounting = self.object(
            run.get("work_accounting"), "work_accounting")
        for arm in NEURAL_ARMS:
            work = self.object(model_work.get(arm), f"model_work.{arm}")
            is_memory = arm in MEMORY_ARMS
            is_fixed = arm == "fixed_snn"
            replay_targets = 29 * 149 * 6 if is_memory else 0
            expected_predictions = (54000 if is_fixed else 54900)
            if is_memory:
                expected_predictions += 29 * 150 * 6
            self.require(
                work.get("predictions") == expected_predictions,
                f"seed {seed_index} {arm} prediction work")
            expected_scored = 53994 if is_fixed else 54888
            self.require(
                work.get("scored_targets") == expected_scored + replay_targets,
                f"seed {seed_index} {arm} scored target work")
            expected_updates = 0 if is_fixed else 54888 + replay_targets
            self.require(work.get("updates") == expected_updates,
                         f"seed {seed_index} {arm} update work")
            if is_memory:
                self.require(work.get("replay_passes") == 174,
                             f"seed {seed_index} {arm} replay passes")
                self.require(work.get("replay_targets") == 25926,
                             f"seed {seed_index} {arm} replay targets")
                self.require(work.get("online_training_targets") == 8999,
                             f"seed {seed_index} {arm} online targets")
            calibration = self.object(
                work_accounting.get("calibration_copies"),
                "calibration_copies")
            probe_copies = self.object(
                work_accounting.get("sealed_probe_copies"),
                "sealed_probe_copies")
            cal_work = self.object(calibration.get(arm),
                                   f"calibration_copies.{arm}")
            probe_work = self.object(probe_copies.get(arm),
                                     f"sealed_probe_copies.{arm}")
            self.require(cal_work.get("predictions") == 900,
                         f"seed {seed_index} {arm} calibration predictions")
            self.require(probe_work.get("predictions") == 18600,
                         f"seed {seed_index} {arm} probe predictions")
            self.require(probe_work.get("updates") == 0,
                         f"seed {seed_index} {arm} probe updates")
            self.numeric_checks += 9 if is_memory else 6

    def verify_charts(self, chart_data: dict[str, Any],
                      report_bytes: bytes, runs: list[dict[str, Any]]) -> None:
        metadata = self.object(chart_data.get("metadata"), "chart metadata")
        expected_hash = hashlib.sha256(report_bytes).hexdigest()
        self.require(metadata.get("report_sha256") == expected_hash,
                     "chart report provenance hash")
        live = self.object(chart_data.get("live_cumulative"),
                           "live_cumulative")
        live_series = self.object(live.get("series"), "live series")
        probe_data = self.object(chart_data.get("sealed_probe_change"),
                                 "sealed_probe_change")
        probe_series = self.object(probe_data.get("series"), "probe series")
        for arm in ARMS:
            points = live_series.get(arm)
            self.require(isinstance(points, list) and len(points) == 30,
                         f"chart live points for {arm}")
            for block_index, point_value in enumerate(points):
                point = self.object(point_value, f"live.{arm}[{block_index}]")
                records = [
                    run["blocks"][index]["arms"][arm]
                    for run in runs for index in range(block_index + 1)
                ]
                expected_sse = sum(item["sse"] for item in records)
                expected_targets = sum(
                    item["feature_target_count"] for item in records)
                self.require(
                    point.get("pooled_feature_target_count_cumulative")
                    == expected_targets,
                    f"live pooled target count {arm}[{block_index}]")
                self.close(float(point["pooled_sse_cumulative"]),
                           expected_sse,
                           f"live pooled SSE {arm}[{block_index}]")
                self.close(float(point["pooled_mse_cumulative"]),
                           expected_sse / expected_targets,
                           f"live pooled MSE {arm}[{block_index}]")
                contributions = point.get("per_seed_cumulative")
                self.require(isinstance(contributions, list)
                             and len(contributions) == 3,
                             f"live per-seed values {arm}[{block_index}]")
                for contribution, run in zip(contributions, runs):
                    item = self.object(contribution, "live contribution")
                    records_for_seed = [
                        block["arms"][arm]
                        for block in run["blocks"][:block_index + 1]
                    ]
                    seed_sse = sum(item["sse"] for item in records_for_seed)
                    seed_targets = sum(
                        item["feature_target_count"]
                        for item in records_for_seed)
                    self.require(
                        item.get("feature_target_count") == seed_targets,
                        f"live seed target count {arm}[{block_index}]")
                    self.close(float(item["sse"]), seed_sse,
                               f"live seed SSE {arm}[{block_index}]")
                    self.close(float(item["mse"]), seed_sse / seed_targets,
                               f"live seed MSE {arm}[{block_index}]")

            initial_sse = sum(
                run["sealed_probe_checkpoints"][0]["arms"][arm]["sse"]
                for run in runs)
            initial_mse = initial_sse / 1350
            probe_points = probe_series.get(arm)
            self.require(isinstance(probe_points, list)
                         and len(probe_points) == 31,
                         f"chart probe points for {arm}")
            for checkpoint_index, point_value in enumerate(probe_points):
                point = self.object(
                    point_value, f"probe.{arm}[{checkpoint_index}]")
                records = [run["sealed_probe_checkpoints"][
                    checkpoint_index]["arms"][arm] for run in runs]
                pooled_sse = sum(item["sse"] for item in records)
                pooled_mse = pooled_sse / 1350
                self.require(point.get("pooled_feature_target_count") == 1350,
                             f"probe target count {arm}[{checkpoint_index}]")
                self.close(float(point["pooled_sse"]), pooled_sse,
                           f"probe pooled SSE {arm}[{checkpoint_index}]")
                self.close(float(point["pooled_mse"]), pooled_mse,
                           f"probe pooled MSE {arm}[{checkpoint_index}]")
                self.close(float(point["checkpoint_0_pooled_mse"]),
                           initial_mse,
                           f"probe baseline MSE {arm}[{checkpoint_index}]")
                self.close(
                    float(point["mse_reduction_percent_vs_checkpoint_0"]),
                    100.0 * (initial_mse - pooled_mse) / initial_mse,
                    f"probe reduction {arm}[{checkpoint_index}]")
                contributions = point.get("per_seed")
                self.require(isinstance(contributions, list)
                             and len(contributions) == 3,
                             f"probe per-seed values {arm}"
                             f"[{checkpoint_index}]")
                for contribution, run in zip(contributions, runs):
                    item = self.object(contribution, "probe contribution")
                    expected = run["sealed_probe_checkpoints"][
                        checkpoint_index]["arms"][arm]
                    self.require(item.get("feature_target_count") == 450,
                                 f"probe seed target count {arm}")
                    self.close(float(item["sse"]), expected["sse"],
                               f"probe seed SSE {arm}[{checkpoint_index}]")
                    self.close(float(item["mse"]), expected["mse"],
                               f"probe seed MSE {arm}[{checkpoint_index}]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("chart_data", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report_bytes = args.report.read_bytes()
        report = json.loads(report_bytes)
        if not isinstance(report, dict):
            raise ValueError("report must be a JSON object")
        verifier = AggregateVerifier()
        runs = verifier.verify_report(report)
        if args.chart_data is not None:
            chart_data_value = json.loads(
                args.chart_data.read_text(encoding="utf-8"))
            if not isinstance(chart_data_value, dict):
                raise ValueError("chart data must be a JSON object")
            verifier.verify_charts(chart_data_value, report_bytes, runs)
        result = {
            "valid": True,
            "aggregate_numeric_checks": verifier.numeric_checks,
            "seeds": [run["seed"] for run in runs],
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "work_counts_checked": (
                "fit, 8,999 online targets, 29 replay passes of 149 targets "
                "per memory band, calibration copies, and 31 sealed probes"),
            "chart_checked": args.chart_data is not None,
        }
        output_path = args.output or (
            args.report.parent / "independent-aggregate-check.json")
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, TypeError, KeyError,
            ZeroDivisionError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
