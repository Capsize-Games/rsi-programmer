#!/usr/bin/env python3
"""Select a cheap shortlist window using only frozen v2 train tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import spikeforge
import torch

from scripts import benchmark_pre_eval as pre_eval_script
from rsi_programmer import pre_eval
from rsi_programmer.benchmark_v2_runner import (
    _control_for,
    _expression_polynomial,
    graph_polynomial,
    sha256_bytes,
)
from rsi_programmer.benchmark_v2_training import build_training_groups
from rsi_programmer.probe import interpret, validate_graph
from rsi_programmer.shortlist_pre_eval import (
    shortlist_pre_evaluation_search,
)

PROTOCOL_PATH = ROOT / "fixtures/benchmark_shortlist_selection_v1.json"
V2_PROTOCOL_PATH = ROOT / "fixtures/benchmark_pre_eval_v2.json"
CORPUS_PATH = ROOT / "fixtures/benchmark_pre_eval_corpus_v2.json"
MODEL_REFERENCE_PATH = (
    ROOT / "artifacts/benchmark-pre-eval-v2-20261018/training.json"
)
METHOD_CONTROLS = (
    "cheap_only", "heuristic", "trained_snn", "mlp", "recurrent",
)
CANDIDATE_PARAMETERS = ((64, 4), (256, 16))
REPEATS = 2
ROOT_CAP = 131072
TASK_COUNT = 24


def file_sha256(path: Path) -> str:
    """Return a file's SHA256 digest."""
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def validate_protocol() -> tuple[dict[str, Any], dict[str, Any],
                                 dict[str, Any]]:
    """Verify preregistration inputs and return only the train split."""
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("status") != "frozen-before-training-or-selection":
        raise ValueError("selection protocol is not frozen")
    if file_sha256(Path(__file__)) != protocol["selection_script"]["sha256"]:
        raise ValueError("selection script hash differs from protocol")
    method = protocol["method"]
    if method["engine_api"] != (
            "rsi_programmer.shortlist_pre_eval."
            "shortlist_pre_evaluation_search"):
        raise ValueError("shortlist engine API differs from protocol")
    declared_parameters = tuple(
        (int(item["window_size"]), int(item["shortlist_size"]))
        for item in method["candidate_parameters"]
    )
    if declared_parameters != CANDIDATE_PARAMETERS:
        raise ValueError("shortlist candidate settings differ from protocol")
    if tuple(method["controls"]) != METHOD_CONTROLS:
        raise ValueError("shortlist control order differs from protocol")
    if (method["root_signature_cap"] != ROOT_CAP
            or method["tree_bound"] != 5
            or method["cheap_row_index"] != 0):
        raise ValueError("shortlist root bound or cheap row differs")
    if (protocol["runs"]["repeats"] != REPEATS
            or protocol["runs"]["expected_search_attempts"]
            != TASK_COUNT * REPEATS * (
                1 + len(CANDIDATE_PARAMETERS) * len(METHOD_CONTROLS))):
        raise ValueError("selection repeat or attempt count differs")
    v2_bytes = V2_PROTOCOL_PATH.read_bytes()
    corpus_bytes = CORPUS_PATH.read_bytes()
    model_bytes = MODEL_REFERENCE_PATH.read_bytes()
    expected_inputs = protocol["inputs"]
    for name, data in (
            ("v2_protocol", v2_bytes), ("corpus", corpus_bytes),
            ("model_reference", model_bytes)):
        if sha256_bytes(data) != expected_inputs[name]["sha256"]:
            raise ValueError(f"frozen {name} hash mismatch")
    v2_protocol = json.loads(v2_bytes)
    corpus = json.loads(corpus_bytes)
    model_reference = json.loads(model_bytes)
    if v2_protocol.get("protocol_id") != "benchmark-pre-eval-v2":
        raise ValueError("unexpected source benchmark protocol")
    if corpus.get("corpus_id") != "benchmark-pre-eval-v2":
        raise ValueError("unexpected source corpus")
    if protocol["inputs"]["corpus"]["sha256"] != (
            v2_protocol["corpus"]["sha256"]):
        raise ValueError("selection protocol and v2 protocol disagree")
    training = v2_protocol["training"]
    expected_training = protocol["training"]
    for name in ("positive_cap", "negative_cap", "sampling_seed"):
        if training[name] != expected_training["groups"][name]:
            raise ValueError(f"training group recipe mismatch: {name}")
    if (training["seeds"] != [0, 1, 2]
            or training["epochs"] != expected_training["models"]["epochs"]
            or training["learning_rate"]
            != expected_training["models"]["learning_rate"]):
        raise ValueError("model training recipe differs from v2")
    train_tasks = corpus["splits"]["train"]
    if len(train_tasks) != TASK_COUNT:
        raise ValueError("frozen corpus must contain 24 train tasks")
    if any(len(task["train"]) != 12 for task in train_tasks):
        raise ValueError("each train task must have 12 observed rows")
    return protocol, v2_protocol, {
        "train_tasks": train_tasks,
        "model_reference": model_reference,
    }


def source_manifest() -> dict[str, str]:
    """Hash tracked runtime and explicit selection dependencies."""
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "rsi_programmer"], cwd=ROOT,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.split(b"\0")
    candidates = set()
    for relative_path in tracked:
        if not relative_path.endswith(b".py") or not relative_path:
            continue
        name = Path(relative_path.decode("utf-8")).name
        if name.startswith("benchmark_shortlist"):
            continue
        candidates.add(ROOT / relative_path.decode("utf-8"))
    candidates.update({
        ROOT / "rsi_programmer/shortlist_pre_eval.py",
        ROOT / "scripts/benchmark_pre_eval.py",
        ROOT / "scripts/generate_benchmark_pre_eval_v2.py",
        ROOT / "scripts/select_shortlist_method.py",
        PROTOCOL_PATH, V2_PROTOCOL_PATH, CORPUS_PATH, MODEL_REFERENCE_PATH,
        ROOT / "fixtures/benchmark_pre_eval_v2.json",
        ROOT / "fixtures/benchmark_pre_eval_corpus_v2.json",
        ROOT / "justfile", ROOT / "requirements.txt", ROOT / "uv.lock",
    })
    dependency_root = Path(spikeforge.__file__).resolve().parent
    candidates.update(dependency_root.rglob("*.py"))
    return {
        (str(path.relative_to(ROOT)) if ROOT in path.parents
         else str(path)): file_sha256(path)
        for path in sorted(candidates) if path.is_file()
    }


def source_revision() -> str:
    """Return the Git commit used for this selection process."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def environment_manifest() -> dict[str, Any]:
    """Record the interpreter and model runtime used for selection."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "spikeforge_path": str(Path(spikeforge.__file__).resolve()),
    }


def group_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Keep costs and index hashes without duplicating feature vectors."""
    groups = []
    for group in report["groups"]:
        groups.append({
            key: group[key] for key in (
                "task_id", "status", "solving_layer", "full_layer_width",
                "generated_descriptors_through_solving_layer",
                "signature_evaluations", "candidate_row_operations",
                "selected_vector_row_operations", "operand_vector_lookups",
                "feature_builds", "feature_row_statistic_observations",
                "target_range_row_reads", "label_counts", "selected_counts",
                "selected_indices_sha256", "selected_indices_sha256_by_class",
                "sampling_recipe", "sampling_seed", "positive_cap",
                "negative_cap", "setup_seconds", "error",
            )
        })
    return {
        "label_semantics": report["label_semantics"],
        "totals": report["totals"],
        "setup_seconds": report["setup_seconds"],
        "groups": groups,
    }


def validate_training_solution(
        graph: dict[str, Any] | None,
        task: dict[str, Any],
        ) -> dict[str, Any]:
    """Validate only observed training rows and the exact train polynomial."""
    result = {
        "graph_valid": False,
        "observed_rows_valid": False,
        "exact_polynomial_valid": False,
        "validation_error": "",
        "expected_polynomial_sha256": "",
        "actual_polynomial_sha256": "",
    }
    if graph is None:
        result["validation_error"] = "no-graph"
        return result
    try:
        validate_graph(graph)
        result["graph_valid"] = True
        observed = task["train"]
        result["observed_rows_valid"] = all(
            interpret(graph, int(row[0]), int(row[1])) == int(row[2])
            for row in observed
        )
        expected_poly = _expression_polynomial(
            task["generator"]["program"])
        actual_poly = graph_polynomial(graph)
        expected_bytes = json.dumps(
            expected_poly, separators=(",", ":"),
        ).encode("utf-8")
        actual_bytes = json.dumps(
            actual_poly, separators=(",", ":"),
        ).encode("utf-8")
        result["expected_polynomial_sha256"] = sha256_bytes(expected_bytes)
        result["actual_polynomial_sha256"] = sha256_bytes(actual_bytes)
        result["exact_polynomial_valid"] = actual_poly == expected_poly
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        result["validation_error"] = type(exc).__name__ + ": " + str(exc)
    return result


def behavior_record(
        search: dict[str, Any], validation: dict[str, Any],
        ) -> dict[str, Any]:
    """Build repeat-comparison data without wall or CPU timing fields."""
    dynamic_keys = {
        key for key in search
        if key.endswith("_seconds") or key.endswith("_ns")
    }
    stable_search = {
        key: value for key, value in search.items()
        if key not in dynamic_keys and key != "graph"
    }
    return {
        "search": stable_search,
        "graph": search.get("graph"),
        "validation": validation,
    }


def behavior_sha256(
        search: dict[str, Any], validation: dict[str, Any],
        ) -> str:
    """Hash deterministic repeat outcomes and work counters."""
    data = json.dumps(
        behavior_record(search, validation), sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(data)


def schedule(task_id: str, repeat: int
             ) -> list[tuple[str, str, int, int]]:
    """Return paired method order; repeat one reverses repeat zero."""
    entries = [("streaming", "streaming", 0, 0)]
    for window, quota in CANDIDATE_PARAMETERS:
        for control in METHOD_CONTROLS:
            entries.append(("complete_shortlist", control, window, quota))
    ordered = sorted(entries, key=lambda entry: hashlib.sha256(
        f"82091:{task_id}:{entry[0]}:{entry[1]}:{entry[2]}:{entry[3]}"
        .encode("utf-8")).digest())
    return ordered if repeat == 0 else list(reversed(ordered))


def run_search(
        engine: str,
        control: str,
        window: int,
        quota: int,
        task: dict[str, Any],
        models: dict[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    """Run one uninstrumented search and its independent train-only check."""
    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    rows = tuple((int(row[0]), int(row[1])) for row in task["train"])
    expected = tuple(int(row[2]) for row in task["train"])
    policy = (None if control in {"streaming", "cheap_only"}
              else _control_for(control, models, 0, str(task["id"])))
    if engine == "streaming":
        search = pre_eval.pre_evaluation_search(
            None, rows, expected, int(task["max_tree_ops"]), ROOT_CAP,
            task_id=str(task["id"]), oracle=None, consts=(),
            alternatives_per_signature=1, signature_budget=None,
        )
    else:
        search = shortlist_pre_evaluation_search(
            policy, rows, expected, int(task["max_tree_ops"]), ROOT_CAP,
            task_id=str(task["id"]), oracle=None, consts=(),
            alternatives_per_signature=1, signature_budget=None,
            window_size=window, shortlist_size=quota,
            cheap_row_index=0, cheap_only=(control == "cheap_only"),
        )
    validation = validate_training_solution(search.get("graph"), task)
    cpu_ns = time.process_time_ns() - cpu_start
    wall_ns = time.perf_counter_ns() - wall_start
    return search, validation, cpu_ns, wall_ns


def _atomic_json(path: Path, value: Any) -> None:
    """Replace a JSON file atomically."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def _attempt_key(record: dict[str, Any]
                 ) -> tuple[str, str, int, int, str, int]:
    """Return the unique method/control/task/repeat identity."""
    return (str(record["engine"]), str(record["control"]),
            int(record["window_size"]), int(record["shortlist_size"]),
            str(record["task_id"]), int(record["repeat"]))


def summarize_attempts(
        rows: list[dict[str, Any]],
        ) -> tuple[dict[str, Any], bool]:
    """Apply the fixed coverage/CPU selection rule to the full matrix."""
    methods: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    streams: list[dict[str, Any]] = []
    for row in rows:
        if row["engine"] == "streaming":
            streams.append(row)
        else:
            methods.setdefault((int(row["window_size"]),
                                int(row["shortlist_size"]),
                                str(row["control"])), []).append(row)
    stream_coverage = sum(bool(row["accepted"]) for row in streams)
    stream_complete = len(streams) == TASK_COUNT * REPEATS
    stream_by_task: dict[str, list[dict[str, Any]]] = {}
    for item in streams:
        stream_by_task.setdefault(str(item["task_id"]), []).append(item)
    stream_repeats_equal = all(
        len(items) == REPEATS
        and len({item["behavior_sha256"] for item in items}) == 1
        for items in stream_by_task.values()
    )
    method_summaries: list[dict[str, Any]] = []
    behavior_equal = stream_repeats_equal
    for window, quota in CANDIDATE_PARAMETERS:
        control_results: dict[str, Any] = {}
        cpu_values: list[int] = []
        eligible = True
        complete = True
        for control in METHOD_CONTROLS:
            group = methods.get((window, quota, control), [])
            complete = complete and len(group) == TASK_COUNT * REPEATS
            accepted = sum(bool(row["accepted"]) for row in group)
            cpu_values.extend(int(row["attempt_cpu_ns"]) for row in group)
            by_task: dict[str, list[dict[str, Any]]] = {}
            for item in group:
                by_task.setdefault(str(item["task_id"]), []).append(item)
            group_repeats_equal = all(
                len(items) == REPEATS
                and len({item["behavior_sha256"] for item in items}) == 1
                for items in by_task.values()
            )
            behavior_equal = behavior_equal and group_repeats_equal
            enough_coverage = accepted >= stream_coverage
            eligible = eligible and enough_coverage
            control_results[control] = {
                "attempts": len(group),
                "accepted": accepted,
                "coverage_fraction": (
                    accepted / len(group) if group else None),
                "streaming_accepted": stream_coverage,
                "coverage_at_least_streaming": enough_coverage,
                "mean_attempt_cpu_ns": (
                    sum(int(item["attempt_cpu_ns"]) for item in group)
                    / len(group) if group else None),
                "mean_attempt_wall_ns": (
                    sum(int(item["attempt_wall_ns"]) for item in group)
                    / len(group) if group else None),
                "repeat_behavior_identical": group_repeats_equal,
                "errors": sum(bool(item.get("error")) for item in group),
            }
        mean_cpu = (sum(cpu_values) / len(cpu_values)
                    if cpu_values else None)
        method_summaries.append({
            "window_size": window,
            "shortlist_size": quota,
            "complete": complete and stream_complete,
            "eligible": eligible and complete and stream_complete,
            "mean_attempt_cpu_ns_across_five_controls": mean_cpu,
            "attempts_across_five_controls": len(cpu_values),
            "controls": control_results,
        })
    candidates = [item for item in method_summaries if item["eligible"]]
    candidates.sort(key=lambda item: (
        float(item["mean_attempt_cpu_ns_across_five_controls"]),
        int(item["window_size"]),
    ))
    selected = None
    if candidates and behavior_equal:
        selected = {
            "window_size": candidates[0]["window_size"],
            "shortlist_size": candidates[0]["shortlist_size"],
        }
    complete_matrix = (
        stream_complete
        and all(item["complete"] for item in method_summaries)
        and len({_attempt_key(row) for row in rows}) == len(rows)
        and len(rows) == TASK_COUNT * REPEATS * (
            1 + len(CANDIDATE_PARAMETERS) * len(METHOD_CONTROLS))
    )
    summary = {
        "status": ("selected" if selected is not None else
                   "no_eligible_pair" if complete_matrix else "incomplete"),
        "streaming": {
            "attempts": len(streams),
            "accepted": stream_coverage,
            "coverage_fraction": (
                stream_coverage / len(streams) if streams else None),
            "mean_attempt_cpu_ns": (
                sum(int(item["attempt_cpu_ns"]) for item in streams)
                / len(streams) if streams else None),
            "mean_attempt_wall_ns": (
                sum(int(item["attempt_wall_ns"]) for item in streams)
                / len(streams) if streams else None),
            "repeat_behavior_identical": stream_repeats_equal,
        },
        "methods": method_summaries,
        "selected": selected,
        "complete_matrix": complete_matrix,
        "repeats_behavior_identical": behavior_equal,
        "selection_rule_applied_without_changes": True,
    }
    return summary, behavior_equal


def run(output: Path) -> int:
    """Train seed 0, run the preregistered train-only matrix, and report."""
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    raw_path = output / "raw.jsonl"
    protocol, v2_protocol, inputs = validate_protocol()
    tasks = inputs["train_tasks"]
    reference = inputs["model_reference"]
    source_before = {
        "revision": source_revision(),
        "files_sha256": source_manifest(),
    }
    torch.set_num_threads(1)
    environment = environment_manifest()
    groups, group_report = build_training_groups(
        tasks,
        positive_cap=protocol["training"]["groups"]["positive_cap"],
        negative_cap=protocol["training"]["groups"]["negative_cap"],
        sampling_seed=protocol["training"]["groups"]["sampling_seed"],
    )
    if (group_report["totals"]["successful_groups"] != TASK_COUNT
            or group_report["totals"]["failed_groups"] != 0):
        raise ValueError("training group generation did not complete")
    models, model_training, initialization_seconds = (
        pre_eval_script._new_models(
            0, groups, protocol["training"]["models"]["epochs"])
    )
    seed_zero = next(
        item for item in reference["seeds"] if item["seed"] == 0)
    expected_weights = {
        name: seed_zero["controls"][name]["weights_after_fit_sha256"]
        for name in ("heuristic", "trained_snn", "mlp", "recurrent")
    }
    for name in ("heuristic", "trained_snn", "mlp", "recurrent"):
        observed = model_training[name]["weights_after_fit_sha256"]
        if observed != expected_weights[name]:
            raise ValueError(f"seed-0 model weight hash mismatch: {name}")
    frozen_fingerprints = {
        name: pre_eval_script._weights_hash(models[name])
        for name in ("heuristic", "trained_snn", "mlp", "recurrent")
    }
    frozen_state_fingerprints = {
        name: pre_eval_script._state_hash(models[name])
        for name in frozen_fingerprints
    }
    training_record = {
        "status": "complete",
        "training_task_ids": [str(task["id"]) for task in tasks],
        "training_groups": group_summary(group_report),
        "model_initialization_seconds": initialization_seconds,
        "models": model_training,
        "frozen_model_fingerprints": frozen_fingerprints,
        "frozen_model_state_fingerprints": frozen_state_fingerprints,
        "expected_seed0_weight_hashes": expected_weights,
        "model_weight_hashes_match_v2_reference": True,
        "seed": 0,
        "epochs": protocol["training"]["models"]["epochs"],
        "evaluation_fits": 0,
    }
    _atomic_json(output / "training.json", training_record)

    records: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for task in tasks:
            for repeat in range(REPEATS):
                for engine, control, window, quota in schedule(
                        str(task["id"]), repeat):
                    started_cpu = time.process_time_ns()
                    started_wall = time.perf_counter_ns()
                    record: dict[str, Any] = {
                        "engine": engine,
                        "control": control,
                        "window_size": window,
                        "shortlist_size": quota,
                        "task_id": str(task["id"]),
                        "family": str(task["family"]),
                        "repeat": repeat,
                        "seed": 0,
                        "root_signature_cap": ROOT_CAP,
                        "error": "",
                    }
                    try:
                        search, validation, cpu_ns, wall_ns = run_search(
                            engine, control, window, quota, task, models)
                        graph = search.get("graph")
                        accepted = all((
                            validation["graph_valid"],
                            validation["observed_rows_valid"],
                            validation["exact_polynomial_valid"],
                        ))
                        stable_search = {
                            key: value for key, value in search.items()
                            if key not in {"graph"}
                        }
                        record.update({
                            "search": stable_search,
                            "graph": graph,
                            "validation": validation,
                            "accepted": accepted,
                            "attempt_cpu_ns": cpu_ns,
                            "attempt_wall_ns": wall_ns,
                            "behavior_sha256": behavior_sha256(
                                search, validation),
                        })
                    except Exception as exc:
                        record.update({
                            "accepted": False,
                            "attempt_cpu_ns": (
                                time.process_time_ns() - started_cpu),
                            "attempt_wall_ns": (
                                time.perf_counter_ns() - started_wall),
                            "error": type(exc).__name__ + ": " + str(exc),
                            "behavior_sha256": "",
                        })
                    records.append(record)
                    raw_handle.write(json.dumps(
                        record, sort_keys=True, separators=(",", ":"))
                        + "\n")
                    raw_handle.flush()

    final_fingerprints = {
        name: pre_eval_script._weights_hash(models[name])
        for name in frozen_fingerprints
    }
    final_state_fingerprints = {
        name: pre_eval_script._state_hash(models[name])
        for name in frozen_state_fingerprints
    }
    models_frozen = (
        final_fingerprints == frozen_fingerprints
        and final_state_fingerprints == frozen_state_fingerprints
    )
    summary, behavior_equal = summarize_attempts(records)
    errors = sum(bool(record.get("error")) for record in records)
    source_after = {
        "revision": source_revision(),
        "files_sha256": source_manifest(),
    }
    source_unchanged = source_before == source_after
    expected_rows = TASK_COUNT * REPEATS * (
        1 + len(CANDIDATE_PARAMETERS) * len(METHOD_CONTROLS))
    matrix_valid = (
        len(records) == expected_rows
        and all(not record.get("error") for record in records)
        and summary["complete_matrix"]
        and behavior_equal and models_frozen and source_unchanged
    )
    selection_valid = matrix_valid and summary["selected"] is not None
    summary["status"] = (
        "complete" if selection_valid else
        "no_eligible_pair" if matrix_valid else "failed")
    summary["matrix_valid"] = matrix_valid
    summary["models_frozen"] = models_frozen
    summary["source_unchanged"] = source_unchanged
    summary["final_model_fingerprints"] = final_fingerprints
    summary["final_model_state_fingerprints"] = final_state_fingerprints
    summary["errors"] = errors
    summary["expected_attempts"] = expected_rows
    summary["actual_attempts"] = len(records)
    summary["training_fits"] = 1

    total_cpu_ns = time.process_time_ns() - cpu_start
    total_wall_ns = time.perf_counter_ns() - wall_start
    selection_report = {
        **summary,
        "total_cpu_ns": total_cpu_ns,
        "total_wall_ns": total_wall_ns,
        "total_cost_scope": (
            "Process time in run() through training group/model setup, all "
            "528 "
            "search and independent validation attempts, and per-attempt "
            "raw JSONL writes. Excludes final selection/manifest JSON writes; "
            "the command-level execution record must retain full process "
            "CPU and wall time including those final writes."
        ),
        "source_before": source_before,
        "source_after": source_after,
        "environment": environment,
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "selection_script_sha256": file_sha256(Path(__file__)),
        "v2_protocol_sha256": file_sha256(V2_PROTOCOL_PATH),
        "corpus_sha256": file_sha256(CORPUS_PATH),
        "model_reference_sha256": file_sha256(MODEL_REFERENCE_PATH),
        "training_json_sha256": file_sha256(output / "training.json"),
        "raw_jsonl_sha256": file_sha256(raw_path),
    }
    _atomic_json(output / "selection.json", selection_report)
    files = {
        name: file_sha256(output / name)
        for name in ("raw.jsonl", "training.json", "selection.json")
    }
    manifest = {
        "status": summary["status"],
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "source_head": source_after["revision"],
        "selected": summary["selected"] if selection_valid else None,
        "total_cpu_ns": total_cpu_ns,
        "total_wall_ns": total_wall_ns,
        "source_before": source_before,
        "source_after": source_after,
        "source_unchanged": source_unchanged,
        "files_sha256": files,
        "expected_attempts": expected_rows,
        "actual_attempts": len(records),
        "errors": errors,
        "models_frozen": models_frozen,
        "repeats_behavior_identical": behavior_equal,
        "matrix_valid": matrix_valid,
        "selection_json_sha256": file_sha256(output / "selection.json"),
    }
    _atomic_json(output / "manifest.json", manifest)
    return 0 if selection_valid else 2


def main(argv: list[str] | None = None) -> int:
    """Run the preregistered selection exactly once into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/benchmark-shortlist-selection-20261018",
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    existed_before_run = output.exists()
    try:
        return run(output)
    except Exception as exc:
        failure = {
            "status": "failed",
            "error": type(exc).__name__ + ": " + str(exc),
        }
        if not existed_before_run and output.is_dir():
            _atomic_json(output / "failure.json", failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
