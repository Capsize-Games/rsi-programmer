"""Counterbalanced comparison with reward-updated input projections."""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

from rsi_programmer import online_benchmark as base
from rsi_programmer import online_counterbalanced as common
from rsi_programmer.online_diagnostics import INITIAL_FEATURES
from rsi_programmer.online_graph import check_result, run_episode
from rsi_programmer.online_plastic_encoder import OnlinePlasticEncoderPolicy
from rsi_programmer.online_spiking import (
    FEATURE_COUNT,
    HIDDEN_COUNT,
    STEPS,
)
from rsi_programmer.online_spiking import ACTION_ORDER
from rsi_programmer.online_plastic_summary import (
    summarize,
    summarize_retention,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "fixtures" / "online_plastic_encoder_v1.json"
PROTOCOL_SHA256 = (
    "8fd37e1cd6904ab28dae47d48baf95beab68e0819b5a554d920c3d63aa1ebbf9")
ARMS = (
    "persistent_snn", "frozen_snn", "episode_reset_snn",
    "online_nonspiking", "grammar_streaming", "plastic_snn",
    "reset_plastic_snn", "plastic_nonspiking",
)
SEEDS = (0, 1, 2)
BUDGETS = (144, 432)
NEURAL_ARMS = tuple(arm for arm in ARMS if arm != "grammar_streaming")
RESET_ARMS = ("episode_reset_snn", "reset_plastic_snn")
PLASTIC_ARMS = ("plastic_snn", "reset_plastic_snn",
                "plastic_nonspiking")
RETENTION_BOUNDARIES = (0, 16, 32, 48)


def _digest(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    """Encode JSON consistently for protocol-bound model-state hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _state_hash(policy: OnlinePlasticEncoderPolicy) -> str:
    """Hash the full boundary snapshot, including RNG and metrics."""
    return _digest(_json_bytes(policy.snapshot()))


def load_protocol(path: Path = PROTOCOL) -> dict[str, Any]:
    """Verify exact experiment pins, matrix, and policy update contract."""
    payload = path.read_bytes()
    if _digest(payload) != PROTOCOL_SHA256:
        raise ValueError("plastic-encoder protocol hash mismatch")
    protocol = json.loads(payload)["protocol"]
    corpus = base.load_corpus()
    schedule = common.load_schedule()
    matrix = protocol["matrix"]
    if (protocol["source_corpus"]["sha256"] != base.CORPUS_SHA256
            or protocol["source_schedule"]["sha256"]
            != common.SCHEDULE_SHA256
            or len(corpus["tasks"]) != 48
            or len(schedule["orders"]) != 6
            or tuple(protocol["arms"]) != ARMS
            or tuple(protocol["model_seeds"]) != SEEDS
            or tuple(protocol["budgets"]) != BUDGETS
            or matrix["episode_rows"] != 13824
            or protocol["plastic_encoder"]["encoder_lr"] != 0.01
            or protocol["plastic_encoder"]["surrogate_slope"] != 0.3):
        raise ValueError("plastic-encoder protocol matrix changed")
    return protocol


def policy_for(arm: str, seed: int) -> OnlinePlasticEncoderPolicy | None:
    """Create the matched seeded policy for one frozen arm."""
    if arm not in ARMS:
        raise ValueError("unknown plastic-encoder arm")
    if arm == "grammar_streaming":
        return None
    plastic = arm in PLASTIC_ARMS
    return OnlinePlasticEncoderPolicy(
        seed=seed,
        learning=arm != "frozen_snn",
        spiking=arm not in ("online_nonspiking", "plastic_nonspiking"),
        encoder_learning=plastic,
        encoder_lr=0.01,
    )


def _metric_delta(before: dict[str, Any],
                  after: dict[str, Any]) -> dict[str, int]:
    """Return only nonnegative integer work-counter increments."""
    return {key: after[key] - before[key]
            for key in after
            if type(after[key]) is int and type(before.get(key)) is int}


def _context_hash(observed: tuple[tuple[int, int, int], ...]) -> str:
    """Hash observed examples in their frozen source order."""
    return _digest(_json_bytes(observed))


def probe_retention(
    policy: OnlinePlasticEncoderPolicy,
    tasks: list[dict[str, Any]],
    *,
    order_id: str,
    phase_sequence: list[int],
    budget: int,
    seed: int,
    arm: str,
    completed_tasks: int,
) -> dict[str, Any]:
    """Probe cold root-lane preferences on copies without source mutation."""
    if completed_tasks not in RETENTION_BOUNDARIES:
        raise ValueError("retention boundary is not frozen")
    if len(tasks) != 48 or arm not in NEURAL_ARMS:
        raise ValueError("retention requires 48 tasks and a neural arm")
    started = time.process_time()
    source_before = _state_hash(policy)
    contexts = []
    code_set = set()
    aggregate_work: dict[str, int] = {}
    copy_hashes: dict[str, str] | None = None
    for task in tasks:
        clone = policy.evaluation_copy()
        initial = clone.metrics()
        if copy_hashes is None:
            snapshot = clone.snapshot()
            copy_hashes = {
                "evaluation_copy_state_sha256": _digest(
                    _json_bytes(snapshot)),
                "readout_hash": snapshot["readout_hash"],
                "encoder_hash": snapshot["encoder_hash"],
                "weight_hash": snapshot["weight_hash"],
            }
        observed = tuple(tuple(row) for row in task["observed"])
        clone.set_observed_rows(observed)
        clone.start_episode(f"retention:{task['id']}")
        rates, _spikes = clone._encode(list(INITIAL_FEATURES))
        logits = [
            sum(weight * rate for weight, rate in zip(row, rates))
            + clone._bias[index]
            for index, row in enumerate(clone._weights)
        ]
        probabilities_by_index = clone._masked_softmax(
            logits, tuple(range(len(ACTION_ORDER)))
        )
        probabilities = [probabilities_by_index[index]
                         for index in range(len(ACTION_ORDER))]
        clone.end_episode()
        final = clone.metrics()
        for name, value in _metric_delta(initial, final).items():
            aggregate_work[name] = aggregate_work.get(name, 0) + value
        code = tuple(round(value, 12) for value in rates)
        code_set.add(code)
        root_index = ACTION_ORDER.index(task["root"])
        contexts.append({
            "task_id": task["id"],
            "phase": task["phase"],
            "root": task["root"],
            "context_sha256": _context_hash(observed),
            "rates": rates,
            "probabilities": probabilities,
            "root_probability": probabilities[root_index],
            "argmax": ACTION_ORDER[max(
                range(len(probabilities)), key=probabilities.__getitem__)],
        })
    source_after = _state_hash(policy)
    if source_before != source_after:
        raise RuntimeError("retention probe changed source policy state")
    cpu = time.process_time() - started
    return {
        "order_id": order_id,
        "phase_sequence": phase_sequence,
        "budget": budget,
        "seed": seed,
        "arm": arm,
        "completed_tasks": completed_tasks,
        "contexts": contexts,
        "source_state_sha256_before": source_before,
        "source_state_sha256_after": source_after,
        **(copy_hashes or {}),
        "distinct_rate_codes": len(code_set),
        "probe_count": len(contexts),
        "probe_cpu_seconds": cpu,
        "probe_work_counters": aggregate_work,
        "explicit_probe_work": {
            "evaluation_copies": len(contexts),
            "observed_context_sets": len(contexts),
            "cold_feature_forwards": len(contexts),
            "root_softmaxes": len(contexts),
            "neuron_ticks": len(contexts) * STEPS * HIDDEN_COUNT,
            "base_encoder_multiply_accumulates": (
                len(contexts) * STEPS * HIDDEN_COUNT * FEATURE_COUNT),
            "context_additions": len(contexts) * STEPS * HIDDEN_COUNT,
            "readout_multiply_accumulates": (
                len(contexts) * len(ACTION_ORDER) * HIDDEN_COUNT),
            "context_projection_multiply_accumulates": (
                aggregate_work.get("projection_mac", 0)),
        },
    }


def _attempt_row(
    task: dict[str, Any],
    *,
    order_id: str,
    phase_sequence: list[int],
    stream_index: int,
    arm: str,
    seed: int,
    budget: int,
    policy: OnlinePlasticEncoderPolicy | None,
    draw_offset: int,
) -> tuple[dict[str, Any], int]:
    """Execute one task and keep search, check, and state costs separate."""
    reset_cpu = 0.0
    if policy is not None and arm in RESET_ARMS:
        tick = time.process_time()
        policy.reset_learning()
        reset_cpu = time.process_time() - tick
    state_hash_cpu = 0.0
    if policy is not None:
        tick = time.process_time()
        before_snapshot = policy.snapshot()
        before_payload = _json_bytes(before_snapshot)
        before_state_hash = _digest(before_payload)
        state_hash_cpu += time.process_time() - tick
        readout_before = before_snapshot["readout_hash"]
        encoder_before = before_snapshot["encoder_hash"]
        model_before = before_snapshot["weight_hash"]
    else:
        before_snapshot = None
        before_state_hash = None
        readout_before = encoder_before = model_before = None

    observed = tuple(tuple(row) for row in task["observed"])
    context_cpu = 0.0
    context_work: dict[str, int] = {}
    before_context = policy.metrics() if policy else {}
    if policy is not None:
        tick = time.process_time()
        policy.set_observed_rows(observed)
        context_cpu = time.process_time() - tick
        context_work = _metric_delta(before_context, policy.metrics())
    before = policy.metrics() if policy else {}
    tick = time.process_time()
    result = run_episode(policy, observed, budget=budget,
                         episode_id=task["id"])
    episode_cpu = time.process_time() - tick
    after = policy.metrics() if policy else {}
    work = _metric_delta(before, after)

    held = tuple(tuple(row) for row in task["held"])
    polynomial = tuple(tuple(term) for term in task["polynomial"])
    tick = time.process_time()
    verdict = check_result(result["graph"], held, polynomial)
    checker_cpu = time.process_time() - tick
    if policy is not None:
        tick = time.process_time()
        after_snapshot = policy.snapshot()
        after_payload = _json_bytes(after_snapshot)
        after_state_hash = _digest(after_payload)
        state_hash_cpu += time.process_time() - tick
        readout_after = after_snapshot["readout_hash"]
        encoder_after = after_snapshot["encoder_hash"]
        model_after = after_snapshot["weight_hash"]
    else:
        after_snapshot = None
        after_state_hash = None
        readout_after = encoder_after = model_after = None
        after_payload = b""

    actions = result["attempts"]
    row = {
        "order_id": order_id,
        "phase_sequence": phase_sequence,
        "stream_index": stream_index,
        "phase_position": stream_index // 16,
        "phase_ordinal": stream_index % 16,
        "arm": arm,
        "seed": seed,
        "budget": budget,
        "task_id": task["id"],
        "phase": task["phase"],
        "root": task["root"],
        "first_lane_correct": bool(actions)
        and actions[0]["action"] == task["root"],
        "rng_draw_offset_before": draw_offset,
        "rng_draw_offset_after": draw_offset + result["decision_calls"],
        "readout_hash_before": readout_before,
        "readout_hash_after": readout_after,
        "encoder_hash_before": encoder_before,
        "encoder_hash_after": encoder_after,
        "learning_model_hash_before": model_before,
        "learning_model_hash_after": model_after,
        "full_state_sha256_before": before_state_hash,
        "full_state_sha256_after": after_state_hash,
        "state_hash_cpu_seconds": state_hash_cpu,
        "reset_cpu_seconds": reset_cpu,
        "context_cpu_seconds": context_cpu,
        "context_work": context_work,
        "policy_work": work,
        **result,
        **verdict,
    }
    row.update({
        "engine_episode_cpu_seconds": result["episode_cpu_seconds"],
        "engine_checker_cpu_seconds": verdict["checker_cpu_seconds"],
        "episode_cpu_seconds": episode_cpu,
        "checker_cpu_seconds": checker_cpu,
        "state_hash_cpu_seconds": state_hash_cpu,
        "model_state_json_bytes": len(after_payload),
    })
    return row, row["rng_draw_offset_after"]


def run(output: Path) -> dict[str, Any]:
    """Execute the exact frozen matrix and keep all raw episodes and probes."""
    load_protocol()
    corpus = base.load_corpus()
    schedule = common.load_schedule()
    tasks = {task["id"]: task for task in corpus["tasks"]}
    canonical_tasks = corpus["tasks"]
    output.mkdir(parents=True, exist_ok=False)
    sources_before = base.source_manifest()
    started = time.perf_counter()
    process_started = time.process_time()
    compact_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    setup_cpu: dict[str, float] = {}
    probe_cpu: dict[str, float] = {}
    with gzip.open(output / "episodes.jsonl.gz", "wt",
                   encoding="utf-8") as episode_stream, gzip.open(
            output / "retention.jsonl.gz", "wt",
            encoding="utf-8") as retention_stream:
        for order in schedule["orders"]:
            order_id = order["order_id"]
            phase_sequence = list(order["phase_sequence"])
            for budget in BUDGETS:
                for seed in SEEDS:
                    for arm in ARMS:
                        key = f"{order_id}:{budget}:{arm}:{seed}"
                        tick = time.process_time()
                        policy = policy_for(arm, seed)
                        setup_cpu[key] = time.process_time() - tick
                        draw_offset = 0
                        if policy is not None:
                            probe = probe_retention(
                                policy, canonical_tasks,
                                order_id=order_id,
                                phase_sequence=phase_sequence,
                                budget=budget, seed=seed, arm=arm,
                                completed_tasks=0,
                            )
                            retention_rows.append(probe)
                            probe_cpu[key] = probe["probe_cpu_seconds"]
                            retention_stream.write(json.dumps(
                                probe, sort_keys=True,
                                allow_nan=False) + "\n")
                        else:
                            probe_cpu[key] = 0.0
                        for stream_index, task_id in enumerate(
                                order["task_ids"]):
                            row, draw_offset = _attempt_row(
                                tasks[task_id], order_id=order_id,
                                phase_sequence=phase_sequence,
                                stream_index=stream_index, arm=arm,
                                seed=seed, budget=budget, policy=policy,
                                draw_offset=draw_offset,
                            )
                            episode_stream.write(json.dumps(
                                row, sort_keys=True,
                                allow_nan=False) + "\n")
                            compact_rows.append({
                                key: value for key, value in row.items()
                                if key != "attempts"
                            })
                            completed = stream_index + 1
                            if (policy is not None
                                    and completed in RETENTION_BOUNDARIES[1:]):
                                probe = probe_retention(
                                    policy, canonical_tasks,
                                    order_id=order_id,
                                    phase_sequence=phase_sequence,
                                    budget=budget, seed=seed, arm=arm,
                                    completed_tasks=completed,
                                )
                                retention_rows.append(probe)
                                probe_cpu[key] += probe["probe_cpu_seconds"]
                                retention_stream.write(json.dumps(
                                    probe, sort_keys=True,
                                    allow_nan=False) + "\n")
                        print(order_id, budget, seed, arm, flush=True)
    sources_after = base.source_manifest()
    if sources_before != sources_after:
        raise RuntimeError("source changed during plastic-encoder benchmark")
    expected_episodes = (len(schedule["orders"]) * len(BUDGETS)
                         * len(SEEDS) * len(ARMS) * len(canonical_tasks))
    expected_probes = (len(schedule["orders"]) * len(BUDGETS)
                       * len(SEEDS) * len(NEURAL_ARMS)
                       * len(RETENTION_BOUNDARIES))
    if (len(compact_rows) != expected_episodes
            or len(retention_rows) != expected_probes):
        raise RuntimeError("plastic-encoder matrix is incomplete")
    summary = summarize(compact_rows, setup_cpu, probe_cpu,
                        schedule["orders"])
    summary["retention"] = summarize_retention(
        retention_rows, schedule["orders"])
    full_cpu = time.process_time() - process_started
    metadata = {
        "protocol_sha256": PROTOCOL_SHA256,
        "corpus_sha256": base.CORPUS_SHA256,
        "schedule_sha256": common.SCHEDULE_SHA256,
        "source_files": sources_before,
        "source_unchanged": True,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            text=True).strip(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "snntorch")},
        "episode_rows": len(compact_rows),
        "retention_rows": len(retention_rows),
        "setup_cpu_seconds": setup_cpu,
        "probe_cpu_seconds": probe_cpu,
        "full_process_cpu_seconds": full_cpu,
        "wall_seconds": time.perf_counter() - started,
        "whole_process_peak_rss_kib": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss,
        "rss_scope": "Process-wide cumulative peak; no per-arm peak claim.",
        "per_arm_cpu_scope": "setup + reset + observed-context setup + "
        "episode and held check + full-state hash + all read-only probes",
        "exclusions": "Output compression and provenance hashing outside "
        "the explicitly timed state/probe hash operations.",
        "retention_raw_sha256": _digest(
            (output / "retention.jsonl.gz").read_bytes()),
        "episodes_raw_sha256": _digest(
            (output / "episodes.jsonl.gz").read_bytes()),
    }
    for name, payload in (("summary.json", summary),
                          ("manifest.json", metadata)):
        (output / name).write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary
