"""Frozen complete-fallback shortlist versus full-layer and streaming search.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import resource
import time
import tracemalloc
import math
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch

from scripts import benchmark_pre_eval as v1
from rsi_programmer import benchmark_total_work as work
from rsi_programmer.benchmark_shortlist_summary import (
    configurations,
    summarize,
)
from rsi_programmer.benchmark_v2_runner import (
    ROOT,
    _atomic_json,
    _control_for,
    load_frozen_inputs,
    sha256_bytes,
)
from rsi_programmer.benchmark_v2_training import build_training_groups
from rsi_programmer.shortlist_pre_eval import shortlist_pre_evaluation_search

PROTOCOL_PATH = ROOT / "fixtures/benchmark_complete_shortlist_v1.json"
PROTOCOL_SHA256 = (
    "189fa147a11040c6a05ef173522a6146285de616996724adff49d17b5f66be0a"
)
EXTRA_COUNTS = (
    "skipped_descriptors",
    "scored_unexecuted_descriptors",
    "peak_live_descriptors",
    "peak_live_features",
    "peak_live_cheap_scores",
    "cheap_candidate_row_operations",
    "cheap_operand_value_lookups",
    "cheap_target_value_lookups",
    "cheap_residual_arithmetic_operations",
    "cheap_scored_descriptors", "shortlist_descriptors",
    "fallback_executions", "shortlist_executions", "descriptor_replays",
    "duplicate_signature_evaluations", "permanently_pruned_descriptors",
    "cheap_scored_unexecuted_descriptors",
)



def control_for(name: str, models: dict[str, Any], seed: int,
                task_id: str) -> Any:
    """Cheap-only guidance has no learned policy or model state."""
    return (None if name == "cheap_only"
            else _control_for(name, models, seed, task_id))


def _pinned_path(label: str) -> Path:
    """Resolve a safe repository-relative evidence path."""
    relative = Path(label)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("evidence path must stay inside the repository")
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError("evidence path must stay inside the repository")
    return path


def _read_pinned_json(label: str, expected_hash: str
                      ) -> dict[str, Any]:
    """Read a JSON artifact only after checking its pinned digest."""
    path = _pinned_path(label)
    data = path.read_bytes()
    if sha256_bytes(data) != expected_hash:
        raise ValueError("selection evidence hash mismatch: " + label)
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("selection evidence must be a JSON object")
    return value


def _selection_artifact_path(manifest_path: Path, label: str) -> Path:
    """Resolve manifest-listed files relative to repo or manifest folder."""
    relative = Path(label)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("selection artifact path is unsafe")
    candidates: list[Path] = []
    if (ROOT / relative).exists():
        candidates.append(_pinned_path(label))
    sibling = manifest_path.parent / relative
    if sibling.exists():
        resolved = sibling.resolve()
        if ROOT.resolve() not in resolved.parents:
            raise ValueError("selection artifact leaves repository")
        candidates.insert(0, resolved)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("selection artifact is absent: " + label)


def load_selection_evidence(protocol: dict[str, Any]
                            ) -> dict[str, Any]:
    """Validate selector output and return the externally charged receipt."""
    evidence = protocol["selection_evidence"]
    manifest = _read_pinned_json(evidence["path"], evidence["sha256"])
    execution = _read_pinned_json(
        evidence["execution_path"], evidence["execution_sha256"])
    if manifest.get("status") != "complete":
        raise ValueError("shortlist selection manifest is incomplete")
    if (manifest.get("actual_attempts") != manifest.get("expected_attempts")
            or type(manifest.get("actual_attempts")) is not int
            or manifest.get("errors") != 0
            or manifest.get("matrix_valid") is not True
            or manifest.get("models_frozen") is not True
            or manifest.get("repeats_behavior_identical") is not True
            or manifest.get("source_unchanged") is not True):
        raise ValueError("shortlist selection manifest failed validation")
    if evidence.get("status") != "complete":
        raise ValueError("protocol selection evidence is incomplete")
    if evidence.get("actual_attempts") != manifest["actual_attempts"]:
        raise ValueError("selection attempt count differs from manifest")
    method = protocol["method"]
    selected = {
        key: method[key]
        for key in ("window_size", "shortlist_size", "cheap_row_index")
    }
    if evidence.get("selected") != selected:
        raise ValueError("selected method differs from protocol")
    if (manifest.get("selected", {}).get("window_size")
            != method["window_size"]
            or manifest.get("selected", {}).get("shortlist_size")
            != method["shortlist_size"]):
        raise ValueError("selected window/shortlist differs from manifest")
    if (type(execution.get("exit_code")) is not int
            or execution["exit_code"] != 0):
        raise ValueError("selector execution receipt has nonzero exit code")
    if execution.get("source_unchanged") is not True:
        raise ValueError("selector execution observed source drift")
    files = evidence["files_sha256"]
    if files != manifest.get("files_sha256"):
        raise ValueError("selector output hash map differs from manifest")
    manifest_path = _pinned_path(evidence["path"])
    bundle_files: dict[str, Path] = {}
    for label, expected in files.items():
        artifact = _selection_artifact_path(manifest_path, label)
        if sha256_bytes(artifact.read_bytes()) != expected:
            raise ValueError("selector result file hash mismatch: " + label)
        bundle_files[Path(label).name] = artifact
    selection_report = json.loads(
        bundle_files["selection.json"].read_text())
    if (selection_report.get("complete_matrix") is not True
            or selection_report.get("actual_attempts")
            != selection_report.get("expected_attempts")
            or selection_report.get("errors") != 0
            or selection_report.get("models_frozen") is not True
            or selection_report.get("repeats_behavior_identical") is not True
            or selection_report.get("selection_rule_applied_without_changes")
            is not True
            or selection_report.get("corpus_sha256")
            != protocol["corpus"]["sha256"]
            or selection_report.get("model_reference_sha256")
            != protocol["model_reference"]["sha256"]):
        raise ValueError("selector result is incomplete or mismatched")
    if (type(evidence.get("coverage_fraction_each_control_both_pairs"))
            not in (int, float)
            or evidence["coverage_fraction_each_control_both_pairs"] != 1.0):
        raise ValueError("selector coverage is incomplete")
    methods = selection_report.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("selector result has no candidate methods")
    eligible_methods = [method_row for method_row in methods
                        if method_row.get("eligible") is True]
    if not eligible_methods:
        raise ValueError("selector produced no eligible method")
    eligible_pairs = [
        {"window_size": method_row["window_size"],
         "shortlist_size": method_row["shortlist_size"]}
        for method_row in eligible_methods
    ]
    if eligible_pairs != evidence.get("eligible_pairs"):
        raise ValueError("eligible methods differ from pinned evidence")
    for method_row in eligible_methods:
        if (method_row.get("complete") is not True
                or method_row.get("attempts_across_five_controls") != 240
                or method_row.get("mean_attempt_cpu_ns_across_five_controls")
                is None):
            raise ValueError("eligible selection method is incomplete")
        controls = method_row.get("controls", {})
        expected_controls = {"cheap_only", *protocol["evaluation"][
            "ranked_controls"]}
        if set(controls) != expected_controls:
            raise ValueError("selector control coverage is incomplete")
        for control in expected_controls:
            record = controls[control]
            if (record.get("attempts") != 48
                    or record.get("accepted") != 48
                    or record.get("streaming_accepted") != 48
                    or record.get("coverage_fraction") != 1.0
                    or record.get("coverage_at_least_streaming") is not True
                    or record.get("errors") != 0
                    or record.get("repeat_behavior_identical") is not True):
                raise ValueError("selector control failed coverage")
    selected_pair = (method["window_size"], method["shortlist_size"])
    if selected_pair not in {
            (row["window_size"], row["shortlist_size"])
            for row in eligible_methods}:
        raise ValueError("selected method was not eligible")
    best_method = min(
        eligible_methods,
        key=lambda row: (row["mean_attempt_cpu_ns_across_five_controls"],
                         row["window_size"]),
    )
    if selected_pair != (best_method["window_size"],
                         best_method["shortlist_size"]):
        raise ValueError("selected method differs from frozen cost rule")
    child_cpu = execution.get("child_total_cpu_seconds")
    cli_wall = execution.get("cli_wall_seconds")
    if (type(child_cpu) not in (int, float)
            or type(cli_wall) not in (int, float)
            or not math.isfinite(child_cpu)
            or not math.isfinite(cli_wall)
            or child_cpu < 0 or cli_wall < 0):
        raise ValueError("selector execution receipt lacks valid CPU/wall")
    if (execution.get("files_after") is not None
            and execution.get("files_after") != execution.get("files_before")):
        raise ValueError("selector execution observed source drift")
    charged_cpu = round(child_cpu * 1_000_000_000)
    charged_wall = round(cli_wall * 1_000_000_000)
    if (evidence.get("execution_charge_cpu_ns") != charged_cpu
            or evidence.get("execution_charge_wall_ns") != charged_wall):
        raise ValueError("selection charges differ from external receipt")
    return {
        **evidence,
        "internal_total_cpu_ns": manifest["total_cpu_ns"],
        "internal_total_wall_ns": manifest["total_wall_ns"],
        "total_cpu_ns": charged_cpu,
        "total_wall_ns": charged_wall,
        "external_execution_receipt": {
            "path": evidence["execution_path"],
            "sha256": evidence["execution_sha256"],
            "exit_code": execution["exit_code"],
            "child_total_cpu_seconds": child_cpu,
            "cli_wall_seconds": cli_wall,
        },
    }


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Reject altered protocol/reference inputs before setup or search."""
    data = path.read_bytes()
    if sha256_bytes(data) != PROTOCOL_SHA256:
        raise ValueError("complete shortlist protocol hash mismatch")
    protocol = json.loads(data)
    load_frozen_inputs()
    baseline = protocol["baseline_source"]
    for label, expected in {
        baseline["path"]: baseline["sha256"],
        **baseline["helper_sha256"],
        protocol["model_reference"]["path"]: protocol["model_reference"][
            "sha256"
        ],
        protocol["selection_evidence"]["path"]:
        protocol["selection_evidence"]["sha256"],
        protocol["selection_evidence"]["execution_path"]:
        protocol["selection_evidence"]["execution_sha256"],
    }.items():
        if sha256_bytes((ROOT / label).read_bytes()) != expected:
            raise ValueError("frozen source/model hash mismatch: " + label)
    load_selection_evidence(protocol)
    return protocol


def baseline_search(protocol: dict[str, Any]) -> Callable[..., Any]:
    """Load the immutable cached engine without global monkeypatches."""
    path = ROOT / protocol["baseline_source"]["path"]
    if (
        sha256_bytes(path.read_bytes())
        != protocol["baseline_source"]["sha256"]
    ):
        raise ValueError("cached baseline hash mismatch")
    spec = importlib.util.spec_from_file_location("_cached_v71", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load cached baseline")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pre_evaluation_search


def engine_order(
    seed: int, task: str, budget: int, control: str, repeat: int
) -> list[str]:
    """Counterbalance adjacent pairs in the frozen hash schedule."""
    key = f"82082:{seed}:{task}:{budget}:{control}".encode()
    bit = hashlib.sha256(key).digest()[0] % 2
    names = ["full_cached", "complete_shortlist"]
    if bit ^ repeat:
        names.reverse()
    return names


def snapshot(protocol: dict[str, Any]) -> dict[str, Any]:
    """Hash every runtime/test/script/fixture and active dependency source."""
    before = work._source_hashes(ROOT / protocol["baseline_source"]["path"])
    paths = set((ROOT / "tests").rglob("*.py"))
    paths.update((ROOT / "scripts").glob("*.py"))
    paths.update((ROOT / "fixtures").glob("*.json"))
    paths.add(ROOT / "justfile")
    before["source_files_sha256"].update(
        {
            str(p.relative_to(ROOT)): sha256_bytes(p.read_bytes())
            for p in sorted(paths)
        }
    )
    return {
        "source_sha256": before,
        "environment": work._environment(),
        "git": work._git_state(),
    }


def attempt(
    engine: str,
    fn: Callable[..., Any],
    policy: Any,
    task: dict[str, Any],
    budget: int,
) -> dict[str, Any]:
    """Use the same timed hash/search/independent verification wrapper."""
    row = work._attempt(engine, fn, policy, task, budget, task["id"])
    shortlisted = engine in {"complete_shortlist", "cheap_only"}
    for key in EXTRA_COUNTS:
        default = (None if key.startswith("peak_live_") else 0)
        if shortlisted:
            default = None
        row["work"][key] = row["search"].get(key, default)
    if shortlisted:
        row["work"]["scored_unexecuted_descriptors"] = row["search"].get(
            "scored_unexecuted_descriptors")
    else:
        row["work"]["scored_unexecuted_descriptors"] = (
            row["work"]["scored_descriptors"] - row["root_calls"]
            if engine != "streaming" else 0)
    row["behavior_fingerprint"] = work._json_hash({
        "original": row["behavior_fingerprint"], "work": row["work"],
        "layers": {k: v for k, v in row["search"].items()
                   if k.startswith("layer_")},
    })
    return row


def _shortlist_metrics_valid(row: dict[str, Any],
                             method: dict[str, Any]) -> bool:
    """Check measured shortlist work, execution, and memory invariants."""
    search = row.get("search", {})
    work_counts = row.get("work", {})
    required = EXTRA_COUNTS + (
        "signature_evaluations", "generated_descriptors",
        "scored_descriptors", "feature_builds", "policy_batches",
        "layer_widths", "layer_generated_descriptors",
        "layer_executed_candidates", "layer_shortlist_descriptors",
        "layer_shortlist_executions", "layer_fallback_executions",
    )
    if any(key not in search for key in required):
        return False
    numeric = required[:len(EXTRA_COUNTS)] + (
        "signature_evaluations", "generated_descriptors",
        "scored_descriptors", "feature_builds", "policy_batches",
    )
    if any(type(search[key]) is not int or search[key] < 0
           for key in numeric):
        return False
    if any(type(work_counts.get(key)) is not int
           or work_counts[key] < 0 for key in EXTRA_COUNTS):
        return False
    generated = search["generated_descriptors"]
    cheap_scored = search["cheap_scored_descriptors"]
    signature_evaluations = search["signature_evaluations"]
    shortlist_executions = search["shortlist_executions"]
    scored = search["scored_descriptors"]
    if (cheap_scored != generated
            or search["cheap_candidate_row_operations"] != cheap_scored
            or search["cheap_operand_value_lookups"] != 2 * cheap_scored
            or search["cheap_target_value_lookups"]
            != (1 if cheap_scored else 0)
            or search["cheap_residual_arithmetic_operations"]
            != 2 * cheap_scored
            or search["shortlist_executions"]
            + search["fallback_executions"] != signature_evaluations
            or shortlist_executions > search["shortlist_descriptors"]
            or search["scored_unexecuted_descriptors"]
            != (0 if row.get("engine") == "cheap_only"
                else scored - shortlist_executions)
            or search["cheap_scored_unexecuted_descriptors"]
            != cheap_scored - signature_evaluations
            or search["feature_builds"] != scored
            or scored > search["shortlist_descriptors"]):
        return False
    zero_counters = (
        "skipped_descriptors", "descriptor_replays",
        "duplicate_signature_evaluations",
        "permanently_pruned_descriptors",
    )
    if any(search[key] != 0 for key in zero_counters):
        return False
    if row.get("root_calls") != signature_evaluations:
        return False
    if row.get("engine") == "cheap_only" and any(
            search[key] != 0 for key in (
                "scored_descriptors", "feature_builds", "policy_batches",
                "feature_stat_cache_lookups",
                "feature_target_range_elements_scanned",
                "feature_operand_elements_scanned",
                "reference_gap_elements_scanned",
            )):
        return False
    layer_keys = (
        "layer_widths", "layer_generated_descriptors",
        "layer_executed_candidates", "layer_shortlist_descriptors",
        "layer_shortlist_executions", "layer_fallback_executions",
    )
    arrays = [search[key] for key in layer_keys]
    if (any(not isinstance(value, list) for value in arrays)
            or len({len(value) for value in arrays}) != 1):
        return False
    if any(type(width) is not int or width < 0
           for width in search["layer_widths"]):
        return False
    if any(type(value) is not int or value < 0
           for values in arrays[1:] for value in values):
        return False
    if (sum(search["layer_generated_descriptors"]) != generated
            or sum(search["layer_executed_candidates"])
            != signature_evaluations
            or sum(search["layer_shortlist_descriptors"])
            != search["shortlist_descriptors"]
            or sum(search["layer_shortlist_executions"])
            != shortlist_executions
            or sum(search["layer_fallback_executions"])
            != search["fallback_executions"]):
        return False
    return (
        search["peak_live_descriptors"] <= method["window_size"]
        and search["peak_live_features"] <= method["shortlist_size"]
        and search["peak_live_cheap_scores"] <= method["window_size"]
    )


def memory_child(
    connection: Any,
    engine: str,
    fn: Callable[..., Any],
    policy: Any,
    task: dict[str, Any],
    budget: int,
    mode: str,
) -> None:
    """Measure one isolated child; traced timing never feeds cost curves."""
    try:
        initial = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if mode == "tracemalloc":
            tracemalloc.start()
        row = attempt(engine, fn, policy, task, budget)
        current, peak = (
            tracemalloc.get_traced_memory()
            if mode == "tracemalloc"
            else (None, None)
        )
        if mode == "tracemalloc":
            tracemalloc.stop()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        connection.send(
            {
                **row,
                "memory_mode": mode,
                "initial_rss_kib": initial,
                "peak_rss_kib": rss,
                "peak_rss_delta_kib": rss - initial,
                "traced_current_bytes": current,
                "traced_peak_bytes": peak,
                "instrumented_timing_not_used": True,
            }
        )
    except BaseException as exc:
        connection.send({"error": type(exc).__name__ + ": " + str(exc)})
    finally:
        connection.close()


def measure_memory(
    engine: str,
    fn: Callable[..., Any],
    policy: Any,
    task: dict[str, Any],
    budget: int,
    mode: str,
) -> dict[str, Any]:
    """Fork one child at a time from the same initialized seed-zero parent."""
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=memory_child,
        args=(child, engine, fn, policy, task, budget, mode),
    )
    process.start()
    child.close()
    try:
        row = parent.recv()
    finally:
        parent.close()
        process.join()
    if process.exitcode != 0 or row.get("error"):
        raise RuntimeError(f"memory child failed: {process.exitcode}: {row}")
    return row


def expected_keys(
    protocol: dict[str, Any], corpus: dict[str, Any]
) -> set[tuple[Any, ...]]:
    """Build every required timed cell before accepting a result."""
    return {
        (seed, task["id"], engine, control, budget, repeat)
        for seed in protocol["training"]["seeds"]
        for task in corpus["splits"]["eval"]
        for engine, control, budgets in configurations(protocol)
        for budget in budgets
        for repeat in range(2)
    }


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        row[k]
        for k in ("seed", "task_id", "engine", "control", "budget", "repeat")
    )


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Retain each completed observation immediately, including errors."""
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def run(output: Path, protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Execute the fixed matrix once; failed infrastructure stays visible."""
    protocol = load_protocol(protocol_path)
    _, corpus = load_frozen_inputs()
    selection = load_selection_evidence(protocol)
    expected = expected_keys(protocol, corpus)
    expected_memory_count = (
        len(protocol["memory"]["modes"])
        * len(corpus["splits"]["eval"])
        * len(configurations(protocol))
    )
    if output.exists():
        raise ValueError("use a new output directory")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    full = baseline_search(protocol)
    method = protocol["method"]
    if (type(method.get("window_size")) is not int
            or method["window_size"] < 1
            or type(method.get("shortlist_size")) is not int
            or not 1 <= method["shortlist_size"] <= method["window_size"]
            or type(method.get("cheap_row_index")) is not int
            or method["cheap_row_index"] < 0):
        raise ValueError("invalid frozen shortlist method parameters")
    shortlist = partial(
        shortlist_pre_evaluation_search,
        window_size=method["window_size"],
        shortlist_size=method["shortlist_size"],
        cheap_row_index=method["cheap_row_index"],
    )
    fns = {
        "full_cached": full,
        "complete_shortlist": shortlist,
        "cheap_only": partial(shortlist, cheap_only=True),
        "streaming": full,
    }
    before = snapshot(protocol)
    cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.process_time_ns(), time.perf_counter_ns()
    groups, group_setup = build_training_groups(
        corpus["splits"]["train"],
        positive_cap=protocol["training"]["positive_cap"],
        negative_cap=protocol["training"]["negative_cap"],
        sampling_seed=protocol["training"]["sampling_seed"],
    )
    group_cpu = time.process_time_ns() - started[0]
    group_wall = time.perf_counter_ns() - started[1]
    if any(g.get("status") != "ok" for g in group_setup["groups"]):
        raise ValueError("training group generation failed")
    training: dict[str, Any] = {
        "status": "in_progress",
        "training_groups": group_setup,
        "training_group_cpu_ns": group_cpu,
        "training_group_wall_ns": group_wall,
        "seeds": [],
        "selection": selection,
    }
    records: list[dict[str, Any]] = []
    memory: list[dict[str, Any]] = []

    def metadata(
        seed: int,
        task: dict[str, Any],
        engine: str,
        control: str,
        budget: int,
        repeat: int,
        row: dict[str, Any],
        order: list[str],
    ) -> dict[str, Any]:
        expected = protocol["model_reference"]["expected_weights_after_fit"][
            str(seed)
        ][control]
        return {
            **row,
            "protocol_id": protocol["protocol_id"],
            "corpus_sha256": protocol["corpus"]["sha256"],
            "seed": seed,
            "task_id": task["id"],
            "family": task["family"],
            "engine": engine,
            "control": control,
            "budget": budget,
            "repeat": repeat,
            "engine_order": order,
            "bound": 5,
            "expected_weights_after_fit_sha256": expected,
            "weights_match_after_fit_reference": (
                row["weights_before_eval_sha256"] == expected
            ),
        }

    for seed in protocol["training"]["seeds"]:
        models, fitted, construction = work._make_models_with_cpu(
            seed,
            groups,
            40,
            protocol["model_reference"]["expected_weights_after_fit"][
                str(seed)
            ],
        )
        warmup = work._warmup_controls(models, seed)
        seed_record = {
            "seed": seed,
            "controls": fitted,
            "control_construction": construction,
            "warmup": warmup,
        }
        training["seeds"].append(seed_record)
        _atomic_json(output / "training.json", training)
        if seed == 0:
            for mode in protocol["memory"]["modes"]:
                for task in corpus["splits"]["eval"]:
                    for engine, control, _ in configurations(protocol):
                        policy = control_for(
                            control, models, seed, task["id"]
                        )
                        result = measure_memory(
                            engine, fns[engine], policy, task,
                            protocol["memory"]["budget"], mode
                        )
                        row = metadata(
                            seed,
                            task,
                            engine,
                            control,
                            32768,
                            0,
                            result,
                            [engine],
                        )
                        append_row(output / "memory.jsonl", row)
                        memory.append(row)
                print(
                    json.dumps(
                        {"memory_mode_complete": mode, "rows": len(memory)}
                    ),
                    flush=True,
                )
        for task in corpus["splits"]["eval"]:
            for control in protocol["controls"]:
                budgets = protocol["evaluation"][
                    "streaming_budgets"
                    if control == "streaming"
                    else "ranked_budgets"
                ]
                for budget in budgets:
                    for repeat in range(2):
                        order = (
                            ["streaming"]
                            if control == "streaming"
                            else ["cheap_only"] if control == "cheap_only"
                            else engine_order(
                                seed, task["id"], budget, control, repeat
                            )
                        )
                        for engine in order:
                            policy = control_for(
                                control, models, seed, task["id"]
                            )
                            result = attempt(
                                engine, fns[engine], policy, task, budget
                            )
                            row = metadata(
                                seed,
                                task,
                                engine,
                                control,
                                budget,
                                repeat,
                                result,
                                order,
                            )
                            append_row(output / "raw.jsonl", row)
                            records.append(row)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "task_complete": task["id"],
                        "timed_rows": len(records),
                    }
                ),
                flush=True,
            )
        final_states = {
            name: {
                "state_matches_after_fit": v1._state_hash(model)
                == fitted[name]["state_fingerprint_after_fit"],
                "weights_match_after_fit": v1._weights_hash(model)
                == fitted[name]["weights_after_fit_sha256"],
            }
            for name, model in models.items()
        }
        seed_record["state_after_all_evaluation"] = final_states
        seed_record["all_model_states_frozen"] = all(
            all(flags.values()) for flags in final_states.values()
        )
        _atomic_json(output / "training.json", training)
    after = snapshot(protocol)
    keys = [row_key(r) for r in records]
    grid = (len(keys) == len(set(keys)) and set(keys) == expected
            and len(expected) == protocol["evaluation"]["matrix_rows"])
    repeats: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in records:
        repeats.setdefault(row_key(row)[:-1], []).append(row)
    repeat_valid = all(
        len(rows) == 2 and len({r["behavior_fingerprint"] for r in rows}) == 1
        for rows in repeats.values()
    )
    indexed = {row_key(r): r for r in records}
    memory_keys = [(r["memory_mode"], *row_key(r)) for r in memory]
    expected_memory = {
        (mode, 0, task["id"], engine, control,
         protocol["memory"]["budget"], 0)
        for mode in protocol["memory"]["modes"]
        for task in corpus["splits"]["eval"]
        for engine, control, _ in configurations(protocol)
    }
    memory_valid = (
        len(memory_keys) == len(set(memory_keys)) == expected_memory_count
        and set(memory_keys) == expected_memory
        and all(
            r["behavior_fingerprint"]
            == indexed[row_key(r)]["behavior_fingerprint"]
            for r in memory
        )
    )
    all_rows = records + memory
    checks = {
        "grid_valid": grid,
        "memory_valid": memory_valid,
        "repeats_behavior_identical": repeat_valid,
        "model_states_frozen": all(
            r["state_frozen_during_eval"]
            and r["weights_match_after_fit_reference"]
            for r in all_rows
        )
        and all(s["all_model_states_frozen"] for s in training["seeds"]),
        "no_row_errors": not any(r["error"] for r in all_rows),
        "source_frozen": before["source_sha256"] == after["source_sha256"],
        "environment_frozen": before["environment"] == after["environment"],
        "shortlist_metrics_valid": all(
            _shortlist_metrics_valid(r, method)
            for r in all_rows
            if r["engine"] in {"complete_shortlist", "cheap_only"}
        ),
        "root_cap_valid": all(
            0 <= r["root_calls"] <= r["budget"] for r in all_rows
        ),
    }
    valid = all(checks.values())
    training["status"] = "complete"
    _atomic_json(output / "training.json", training)
    summary = summarize(records, training, protocol)
    summary["infrastructure_valid"] = valid
    summary["decision_supported"] = (
        valid and summary["decision_supported_on_cost"]
    )
    _atomic_json(output / "aggregate.json", summary)
    child_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    manifest = {
        "before": before,
        "after": after,
        **checks,
        "infrastructure_valid": valid,
        "protocol_sha256": sha256_bytes(protocol_path.read_bytes()),
        "row_count": len(records),
        "expected_row_count": len(expected),
        "memory_row_count": len(memory),
        "expected_memory_row_count": expected_memory_count,
        "files_sha256": {
            name: sha256_bytes((output / name).read_bytes())
            for name in (
                "raw.jsonl",
                "memory.jsonl",
                "training.json",
                "aggregate.json",
            )
        },
        "parent_cpu_ns_through_summary": time.process_time_ns() - cpu_start,
        "memory_children_cpu_seconds": (
            child_end.ru_utime
            + child_end.ru_stime
            - child_start.ru_utime
            - child_start.ru_stime
        ),
        "wall_ns_through_summary": time.perf_counter_ns() - wall_start,
        "final_manifest_serialization_included": False,
        "decision_supported": summary["decision_supported"],
    }
    _atomic_json(output / "manifest.json", manifest)
    return manifest
