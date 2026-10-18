"""Frozen bounded shortlist versus cached full-layer and streaming search."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import resource
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import torch

from scripts import benchmark_pre_eval as v1
from rsi_programmer import benchmark_total_work as work
from rsi_programmer.benchmark_bounded_summary import configurations, summarize
from rsi_programmer.benchmark_v2_runner import (
    ROOT,
    _atomic_json,
    _control_for,
    load_frozen_inputs,
    sha256_bytes,
)
from rsi_programmer.benchmark_v2_training import build_training_groups
from rsi_programmer.bounded_pre_eval import bounded_pre_evaluation_search

PROTOCOL_PATH = ROOT / "fixtures/benchmark_bounded_candidates_v1.json"
PROTOCOL_SHA256 = (
    "c235994867676059311cc05ac876060da0aa1517a6e3fa53cd73b80fabd3472d"
)
EXTRA_COUNTS = (
    "skipped_descriptors",
    "scored_unexecuted_descriptors",
    "peak_live_descriptors",
    "peak_live_features",
)


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Reject altered protocol/reference inputs before setup or search."""
    data = path.read_bytes()
    if sha256_bytes(data) != PROTOCOL_SHA256:
        raise ValueError("bounded protocol hash mismatch")
    protocol = json.loads(data)
    load_frozen_inputs()
    baseline = protocol["baseline_source"]
    for label, expected in {
        baseline["path"]: baseline["sha256"],
        **baseline["helper_sha256"],
        protocol["model_reference"]["path"]: protocol["model_reference"][
            "sha256"
        ],
    }.items():
        if sha256_bytes((ROOT / label).read_bytes()) != expected:
            raise ValueError("frozen source/model hash mismatch: " + label)
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
    key = f"81081:{seed}:{task}:{budget}:{control}".encode()
    bit = hashlib.sha256(key).digest()[0] % 2
    names = ["full_cached", "bounded"]
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
    for key in EXTRA_COUNTS:
        row["work"][key] = row["search"].get(key, 0)
    row["work"]["scored_unexecuted_descriptors"] = (
        row["work"]["scored_descriptors"] - row["root_calls"]
        if engine != "streaming"
        else 0
    )
    return row


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
    if output.exists():
        raise ValueError("use a new output directory")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    full = baseline_search(protocol)
    fns = {
        "full_cached": full,
        "bounded": bounded_pre_evaluation_search,
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
                        policy = _control_for(
                            control, models, seed, task["id"]
                        )
                        result = measure_memory(
                            engine, fns[engine], policy, task, 32768, mode
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
                            else engine_order(
                                seed, task["id"], budget, control, repeat
                            )
                        )
                        for engine in order:
                            policy = _control_for(
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
    grid = len(keys) == len(set(keys)) and set(keys) == expected_keys(
        protocol, corpus
    )
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
        (mode, 0, task["id"], engine, control, 32768, 0)
        for mode in protocol["memory"]["modes"]
        for task in corpus["splits"]["eval"]
        for engine, control, _ in configurations(protocol)
    }
    memory_valid = (
        len(memory_keys) == len(set(memory_keys)) == 864
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
        "bounded_peak_valid": all(
            r["search"]["peak_live_descriptors"] <= 256
            and r["search"]["peak_live_features"] <= 256
            for r in all_rows
            if r["engine"] == "bounded"
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
        "expected_row_count": 10656,
        "memory_row_count": len(memory),
        "expected_memory_row_count": 864,
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
