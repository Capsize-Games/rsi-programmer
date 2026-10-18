"""Frozen observed-example context comparison on the online graph stream."""
from __future__ import annotations

import gzip
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from rsi_programmer import online_benchmark as base
from rsi_programmer import online_counterbalanced as common
from rsi_programmer.online_diagnostics import INITIAL_FEATURES, inspect_policy
from rsi_programmer.online_graph import check_result, run_episode
from rsi_programmer.online_observed_context import ObservedContextPolicy

PROTOCOL = base.ROOT / "fixtures" / "online_observed_context_v1.json"
PROTOCOL_SHA256 = (
    "d8897dc237ceed1a8781dd2dd828b4507842bbdf53473558edffa928fe6485fb")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_protocol(path: Path = PROTOCOL) -> dict[str, Any]:
    """Reject changes to the prospective context representation/control set."""
    payload = path.read_bytes()
    if _sha(payload) != PROTOCOL_SHA256:
        raise ValueError("observed context protocol hash mismatch")
    protocol = json.loads(payload)["protocol"]
    if (protocol["source_corpus"]["sha256"] != base.CORPUS_SHA256
            or protocol["source_schedule"]["sha256"] != common.SCHEDULE_SHA256
            or tuple(protocol["arms"]) != base.ARMS
            or tuple(protocol["model_seeds"]) != base.SEEDS
            or tuple(protocol["budgets"]) != base.BUDGETS):
        raise ValueError("observed context control matrix changed")
    return protocol


def policy_for(arm: str, seed: int) -> ObservedContextPolicy | None:
    """Use matched observation context in every neural control."""
    if arm not in base.ARMS:
        raise ValueError("unknown context arm")
    if arm == "grammar_streaming":
        return None
    return ObservedContextPolicy(seed=seed, learning=arm != "frozen_snn",
                                 spiking=arm != "online_nonspiking")


def context_fields(policy: ObservedContextPolicy | None,
                   observed: tuple[tuple[int, int, int], ...]
                   ) -> dict[str, Any]:
    """Prepare only current observed triples and record setup work."""
    if policy is None:
        return {"context_input_sha256": None,
                "context_projection_sha256": None, "context_work": {}}
    policy.set_observed_rows(observed)
    encoded = json.dumps(policy._pending_context, separators=(",", ":"),
                         allow_nan=False).encode()
    return {"context_input_sha256": _sha(encoded),
            "context_projection_sha256": policy._context_projection_hash,
            "context_work": {"observation_scalars": 18,
                             "projection_mac": 288}}


def encoder_capacity(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe fixed cold codes without learning or selecting a recipe."""
    rows = []
    for seed in base.SEEDS:
        for spiking in (True, False):
            policy = ObservedContextPolicy(seed=seed, learning=False,
                                           spiking=spiking)
            codes = []
            for task in tasks:
                policy.set_observed_rows(tuple(tuple(row)
                                               for row in task["observed"]))
                policy.start_episode("capacity")
                rates, _spikes = policy._encode(list(INITIAL_FEATURES))
                policy.end_episode()
                codes.append(tuple(rates))
            rows.append({"seed": seed, "spiking": spiking,
                         "distinct_cold_codes": len({tuple(round(v, 12)
                             for v in code) for code in codes}),
                         "tasks": len(tasks), "rates": codes})
    return {"rows": rows, "scope": "Fixed-encoder side study; no tuning"}


def run(output: Path) -> dict[str, Any]:
    """Run every frozen context control, retaining failed episodes."""
    load_protocol()
    schedule, corpus = common.load_schedule(), base.load_corpus()
    tasks = {task["id"]: task for task in corpus["tasks"]}
    output.mkdir(parents=True, exist_ok=False)
    sources = base.source_manifest()
    started = time.perf_counter()
    tick = time.process_time()
    capacity = encoder_capacity(corpus["tasks"])
    capacity_cpu = time.process_time() - tick
    (output / "capacity.json").write_text(json.dumps(capacity, indent=2)
                                         + "\n")
    compact, setup = [], {}
    with gzip.open(output / "episodes.jsonl.gz", "wt") as stream:
        for order in schedule["orders"]:
            identity = order["order_id"]
            for budget in base.BUDGETS:
                for seed in base.SEEDS:
                    for arm in base.ARMS:
                        tick = time.process_time()
                        policy = policy_for(arm, seed)
                        setup[f"{identity}:{budget}:{arm}:{seed}"] = (
                            time.process_time() - tick)
                        draw_offset = 0
                        for index, task_id in enumerate(order["task_ids"]):
                            task = tasks[task_id]
                            observed = tuple(tuple(row)
                                             for row in task["observed"])
                            tick = time.process_time()
                            if policy and arm == "episode_reset_snn":
                                policy.reset_learning()
                            reset_cpu = time.process_time() - tick
                            tick = time.process_time()
                            context = context_fields(policy, observed)
                            context_cpu = time.process_time() - tick
                            tick = time.process_time()
                            diagnostic = inspect_policy(policy)
                            diagnostic_cpu = time.process_time() - tick
                            before = policy.metrics() if policy else {}
                            result = run_episode(policy, observed,
                                                 budget=budget,
                                                 episode_id=task_id)
                            after = policy.metrics() if policy else {}
                            work = {key: after[key] - before[key]
                                    for key in after
                                    if type(after[key]) is int}
                            verdict = check_result(
                                result["graph"], tuple(tuple(row)
                                for row in task["held"]),
                                tuple(tuple(term)
                                      for term in task["polynomial"]))
                            rewards = [attempt["reward"]
                                       for attempt in result["attempts"]]
                            row = {
                                "order_id": identity,
                                "phase_sequence": order["phase_sequence"],
                                "stream_index": index,
                                "phase_position": index // 16,
                                "phase_ordinal": index % 16,
                                "arm": arm, "seed": seed, "budget": budget,
                                "task_id": task_id, "phase": task["phase"],
                                "root": task["root"],
                                "first_lane_correct": result["attempts"][0]
                                ["action"] == task["root"],
                                "policy_work": work,
                                "weight_before": before.get("weight_hash"),
                                "weight_after": after.get("weight_hash"),
                                "rng_draw_offset_before": draw_offset,
                                "rng_draw_offset_after": draw_offset
                                + result["decision_calls"],
                                "diagnostic_before": diagnostic,
                                "diagnostic_cpu_seconds": diagnostic_cpu,
                                "reset_cpu_seconds": reset_cpu,
                                "context_cpu_seconds": context_cpu,
                                "reward_sum": sum(rewards),
                                "mean_reward": sum(rewards) / len(rewards),
                                **context, **result, **verdict,
                            }
                            draw_offset = row["rng_draw_offset_after"]
                            stream.write(json.dumps(row, sort_keys=True,
                                                    allow_nan=False) + "\n")
                            compact.append({key: value
                                            for key, value in row.items()
                                            if key != "attempts"})
                        print(identity, budget, seed, arm, flush=True)
    if sources != base.source_manifest():
        raise RuntimeError("source changed during context comparison")
    charged = [{**row, "reset_cpu_seconds": row["reset_cpu_seconds"]
                + row["context_cpu_seconds"]} for row in compact]
    summary = common.summarize(charged, setup, schedule["orders"])
    manifest = {
        "source_files": sources, "source_unchanged": True,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "context_protocol_sha256": PROTOCOL_SHA256,
        "schedule_sha256": common.SCHEDULE_SHA256,
        "corpus_sha256": base.CORPUS_SHA256,
        "episode_rows": len(compact), "setup_cpu_seconds": setup,
        "wall_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "raw_sha256": _sha((output / "episodes.jsonl.gz").read_bytes()),
        "capacity_sha256": _sha((output / "capacity.json").read_bytes()),
        "capacity_cpu_seconds": capacity_cpu,
        "full_experiment_cpu_seconds": capacity_cpu + sum(setup.values())
        + sum(row["reset_cpu_seconds"] + row["context_cpu_seconds"]
              + row["diagnostic_cpu_seconds"] + row["episode_cpu_seconds"]
              + row["checker_cpu_seconds"] for row in compact),
        "per_arm_cpu_scope": "setup/reset + context + cold diagnostic + "
                     "policy/execution/update/checker; excludes compression, "
                     "provenance and separately reported capacity side study",
        "experiment_cpu_scope": "Sum of all per-arm search costs plus "
                                "capacity side-study CPU",
        "information_set": "Current observed examples; no held/phase/root",
    }
    for name, value in (("summary.json", summary),
                        ("manifest.json", manifest)):
        (output / name).write_text(json.dumps(value, indent=2,
                                             sort_keys=True) + "\n")
    return summary
