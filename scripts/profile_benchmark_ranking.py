#!/usr/bin/env python3
"""Profile full-layer ranking costs on the frozen v2 evaluation corpus."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts import benchmark_pre_eval as v1
from rsi_programmer import benchmark_total_work as total_work
from rsi_programmer.benchmark_v2_runner import (
    _control_for,
    load_frozen_inputs,
    sha256_bytes,
)
from rsi_programmer.benchmark_v2_training import build_training_groups

CONTROLS = ("streaming", "heuristic", "trained_snn", "mlp", "recurrent")
CAP = 32768
SEED = 0


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.stdout.strip()

    return {"revision": run("rev-parse", "HEAD"),
            "status_porcelain": run("status", "--porcelain")}


def _source_manifest() -> dict[str, Any]:
    candidates = set((ROOT / "rsi_programmer").rglob("*.py"))
    candidates.update((ROOT / "scripts").glob("benchmark*.py"))
    candidates.update({
        ROOT / "scripts/profile_benchmark_ranking.py",
        ROOT / "fixtures/benchmark_pre_eval_v2.json",
        ROOT / "fixtures/benchmark_pre_eval_corpus_v2.json",
        ROOT / "fixtures/benchmark_total_work_v1.json",
        ROOT / "artifacts/benchmark-pre-eval-v2-20261018/training.json",
        ROOT / "requirements.txt", ROOT / "uv.lock",
    })
    import spikeforge

    spikeforge_root = Path(spikeforge.__file__).resolve().parent
    candidates.update(spikeforge_root.rglob("*.py"))
    files = {
        (str(path.relative_to(ROOT)) if ROOT in path.parents else str(path)):
        _hash(path)
        for path in sorted(candidates) if path.is_file()
    }
    return {"files_sha256": files,
            "corpus_sha256": files[
                "fixtures/benchmark_pre_eval_corpus_v2.json"],
            "protocol_sha256": files["fixtures/benchmark_pre_eval_v2.json"],
            "total_work_protocol_sha256": files[
                "fixtures/benchmark_total_work_v1.json"],
            "model_reference_sha256": files[
                "artifacts/benchmark-pre-eval-v2-20261018/training.json"]}


def _profile_stats(profile: cProfile.Profile) -> dict[str, Any]:
    return _stats_summary(pstats.Stats(profile))


def _profile_file_stats(path: Path) -> dict[str, Any]:
    """Summarize a saved cProfile file without rerunning the search."""
    return _stats_summary(pstats.Stats(str(path)))


def _stats_summary(stats: pstats.Stats) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for (filename, line, function), values in stats.stats.items():
        primitive_calls, total_calls, own_s, cumulative_s, _callers = values
        entries.append({
            "file": filename, "line": line, "function": function,
            "primitive_calls": primitive_calls, "calls": total_calls,
            "own_cpu_s": own_s, "cumulative_cpu_s": cumulative_s,
        })
    entries.sort(key=lambda item: item["cumulative_cpu_s"], reverse=True)
    # Cumulative times overlap by design. Keep own time and cumulative time
    # separate so consumers do not sum nested call trees as disjoint costs.
    categories: dict[str, dict[str, float]] = {}
    for item in entries:
        key = item["file"] + ":" + item["function"]
        category = _cost_category(item["file"], item["function"])
        if category is None:
            continue
        record = categories.setdefault(category, {"own_cpu_s": 0.0,
                                                   "cumulative_cpu_s": 0.0})
        record["own_cpu_s"] += item["own_cpu_s"]
        record["cumulative_cpu_s"] += item["cumulative_cpu_s"]
    return {"total_calls": stats.total_calls,
            "profiled_cpu_s": stats.total_tt,
            "top_functions": entries[:40],
            "category_totals": categories,
            "category_boundary_note": (
                "Categories aggregate matching function records. Cumulative "
                "values overlap across caller/callee boundaries; own CPU "
                "values are the disjoint profiler accounting and should be "
                "used for additive ranking. Sorting/retention functions may "
                "also execute inside descriptor-generation callers.")}


def _cost_category(filename: str, function: str) -> str | None:
    name = function.lower()
    path = filename.replace("\\", "/").lower()
    if "pre_eval.py" in path and function in {"_iter_triples", "<genexpr>"}:
        return "descriptor_generation"
    if path.endswith("/policy.py") and function == "layer_triples":
        return "descriptor_generation"
    if "pre_eval_features.py" in path or function in {
            "_feature", "encode", "_features_for_layer"}:
        return "feature_assembly"
    if function in {"rank", "forward", "_rank", "scores", "_forward",
                    "_bonus"}:
        return "model_inference"
    if "sort" in name or "bisect" in name or function in {
            "sorted", "_sample_indices"}:
        return "sorting"
    if any(word in name for word in ("retain", "commit", "append")):
        return "retention"
    return None


def _run_one(policy: Any, task: dict[str, Any]) -> dict[str, Any]:
    result = v1._search_once(policy, task, CAP, str(task["id"]),
                             total_work.pre_evaluation_search)
    return {key: value for key, value in result.items() if key != "graph"}


def _run_profile_pass(models: dict[str, Any], tasks: list[dict[str, Any]],
                      out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profile_dir = out / "cprofile"
    profile_dir.mkdir()
    for control in CONTROLS:
        for ordinal, task in enumerate(tasks):
            policy = _control_for(control, models, SEED, str(task["id"]))
            profiler = cProfile.Profile()
            cpu_start = time.process_time_ns()
            wall_start = time.perf_counter_ns()
            result = profiler.runcall(_run_one, policy, task)
            cpu_ns = time.process_time_ns() - cpu_start
            wall_ns = time.perf_counter_ns() - wall_start
            profile_path = profile_dir / f"{control}-{ordinal:02d}.prof"
            profiler.dump_stats(profile_path)
            rows.append({
                "pass": "cprofile", "control": control,
                "task_id": task["id"], "family": task["family"],
                "cap": CAP, "search_cpu_ns_instrumented": cpu_ns,
                "search_wall_ns_instrumented": wall_ns,
                "profile_file": str(profile_path.relative_to(out)),
                "profile_file_sha256": _hash(profile_path),
                "search": result,
                "profile": _profile_stats(profiler),
            })
    return rows


def _run_memory_pass(models: dict[str, Any], tasks: list[dict[str, Any]]) \
        -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for control in CONTROLS:
        for ordinal, task in enumerate(tasks):
            policy = _control_for(control, models, SEED, str(task["id"]))
            tracemalloc.start()
            cpu_start = time.process_time_ns()
            wall_start = time.perf_counter_ns()
            result = _run_one(policy, task)
            cpu_ns = time.process_time_ns() - cpu_start
            wall_ns = time.perf_counter_ns() - wall_start
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append({
                "pass": "tracemalloc", "control": control,
                "task_id": task["id"], "family": task["family"],
                "cap": CAP, "search_cpu_ns_instrumented": cpu_ns,
                "search_wall_ns_instrumented": wall_ns,
                "traced_current_bytes": current_bytes,
                "traced_peak_bytes": peak_bytes,
                "search": result,
            })
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def run(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    protocol, corpus = load_frozen_inputs()
    tasks = corpus["splits"]["eval"]
    torch.set_num_threads(1)
    groups_start_cpu = time.process_time_ns()
    groups_start_wall = time.perf_counter_ns()
    groups, group_setup = build_training_groups(
        corpus["splits"]["train"], positive_cap=16, negative_cap=256,
        sampling_seed=82078,
    )
    groups_cpu_ns = time.process_time_ns() - groups_start_cpu
    groups_wall_ns = time.perf_counter_ns() - groups_start_wall
    reference_path = ROOT / (
        "artifacts/benchmark-pre-eval-v2-20261018/training.json")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected_weights = reference["seeds"][0]["controls"]
    expected = {name: expected_weights[name]["weights_after_fit_sha256"]
                for name in ("heuristic", "trained_snn", "mlp", "recurrent",
                             "seeded_random")}
    model_start_cpu = time.process_time_ns()
    model_start_wall = time.perf_counter_ns()
    models, training, construction = total_work._make_models_with_cpu(
        SEED, groups, 40, expected,
    )
    fit_cpu_ns = time.process_time_ns() - model_start_cpu
    fit_wall_ns = time.perf_counter_ns() - model_start_wall
    trained_expected = protocol["training"]
    if trained_expected["epochs"] != 40:
        raise ValueError("v2 training epoch count changed")
    manifest: dict[str, Any] = {
        "status": "profiling_in_progress", "seed": SEED,
        "controls_in_order": list(CONTROLS), "candidate_cap": CAP,
        "evaluation_tasks": len(tasks), "profiled_searches_per_pass": (
            len(tasks) * len(CONTROLS)),
        "training_group_setup": group_setup,
        "training_group_setup_cpu_ns": groups_cpu_ns,
        "training_group_setup_wall_ns": groups_wall_ns,
        "model_construction_and_fit_cpu_ns": fit_cpu_ns,
        "model_construction_and_fit_wall_ns": fit_wall_ns,
        "construction": construction, "training": training,
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "spikeforge": str(Path(
                __import__("spikeforge").__file__).resolve()),
        },
        "source_and_input": _source_manifest(), "git": _git_state(),
        "timing_note": (
            "cProfile and tracemalloc pass timings are instrumented and are "
            "not performance measurements. Use profiler own CPU stats for "
            "cost attribution; no uninstrumented timing is reported here."),
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "training.json", {
        "training_groups": group_setup, "training": training,
        "construction": construction,
        "training_group_setup_cpu_ns": groups_cpu_ns,
        "training_group_setup_wall_ns": groups_wall_ns,
        "model_construction_and_fit_cpu_ns": fit_cpu_ns,
        "model_construction_and_fit_wall_ns": fit_wall_ns,
    })
    try:
        profiled = _run_profile_pass(models, tasks, output)
        memory = _run_memory_pass(models, tasks)
        raw_path = output / "raw.jsonl"
        with raw_path.open("w", encoding="utf-8") as handle:
            for row in profiled + memory:
                handle.write(json.dumps(row, sort_keys=True,
                                        allow_nan=False) + "\n")
        manifest.update({
            "status": "complete", "profile_rows": len(profiled),
            "memory_rows": len(memory),
            "raw_jsonl_sha256": _hash(raw_path),
            "profile_file_count": len(list(
                (output / "cprofile").glob("*.prof"))),
        })
        _write_json(output / "manifest.json", manifest)
        _write_json(
            output / "profile_summary.json", _summarize(profiled, memory))
        return 0
    except Exception as exc:
        manifest.update({"status": "failed",
                         "error": type(exc).__name__ + ": " + str(exc)})
        _write_json(output / "manifest.json", manifest)
        raise


def _summarize(profiled: list[dict[str, Any]],
               memory: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"controls": {}}
    for control in CONTROLS:
        control_profiles = [row for row in profiled
                            if row["control"] == control]
        control_memory = [row for row in memory
                          if row["control"] == control]
        categories: dict[str, dict[str, float]] = {}
        for row in control_profiles:
            for name, values in row["profile"]["category_totals"].items():
                bucket = categories.setdefault(
                    name, {"own_cpu_s_sum": 0.0,
                           "cumulative_cpu_s_sum_overlapping": 0.0})
                bucket["own_cpu_s_sum"] += values["own_cpu_s"]
                bucket["cumulative_cpu_s_sum_overlapping"] += (
                    values["cumulative_cpu_s"])
        peaks = [row["traced_peak_bytes"] for row in control_memory]
        summary["controls"][control] = {
            "task_count": len(control_profiles), "cost_categories": categories,
            "tracemalloc_peak_bytes_median": sorted(peaks)[len(peaks) // 2],
            "tracemalloc_peak_bytes_max": max(peaks),
            "profiled_cpu_s_sum": sum(
                row["profile"]["profiled_cpu_s"] for row in control_profiles),
            "profiled_cpu_s_median": sorted(
                row["profile"]["profiled_cpu_s"] for row in control_profiles)[
                    len(control_profiles) // 2],
        }
    summary["stage_entrypoints"] = {}
    for control in CONTROLS:
        control_profiles = [row for row in profiled
                            if row["control"] == control]
        rank_times: list[float] = []
        encode_times: list[float] = []
        for row in control_profiles:
            functions = row["profile"]["top_functions"]
            rank_times.append(sum(
                item["cumulative_cpu_s"] for item in functions
                if item["file"].replace("\\", "/").endswith(
                    "/rsi_programmer/pre_eval.py")
                and item["function"] == "_rank"))
            encode_times.append(sum(
                item["cumulative_cpu_s"] for item in functions
                if item["file"].replace("\\", "/").endswith(
                    "/rsi_programmer/pre_eval_features.py")
                and item["function"] == "encode"))
        summary["stage_entrypoints"][control] = {
            "rank_entrypoint_function": "pre_eval._rank",
            "rank_cumulative_cpu_s_sum_inclusive": sum(rank_times),
            "rank_cumulative_cpu_s_median_inclusive": sorted(rank_times)[
                len(rank_times) // 2],
            "feature_encode_function": "FeatureStatistics.encode",
            "feature_encode_cumulative_cpu_s_sum_inclusive": sum(
                encode_times),
            "feature_encode_cumulative_cpu_s_median_inclusive": sorted(
                encode_times)[len(encode_times) // 2],
            "interpretation": (
                "Each entrypoint is counted once per task. Rank includes its "
                "policy call; encode is reported as a separate feature stage. "
                "These inclusive entrypoint totals overlap nested calls and "
                "are not additive with each other or with category totals."),
        }
    summary["interpretation"] = (
        "Descriptive profiling only. Cumulative category totals can overlap "
        "and must not be added. Own CPU is the additive cProfile attribution. "
        "Memory peaks come from separate tracemalloc runs; traced timings are "
        "not compared with cProfile timings or treated as performance.")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
