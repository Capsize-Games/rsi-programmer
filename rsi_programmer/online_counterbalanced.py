"""Counterbalanced diagnosis with the unchanged online graph learner."""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from rsi_programmer import online_benchmark as base
from rsi_programmer.online_diagnostics import inspect_policy
from rsi_programmer.online_graph import check_result, run_episode

SCHEDULE = base.ROOT / "fixtures" / "online_counterbalanced_v1.json"
SCHEDULE_SHA256 = (
    "0ffbd7eadac93aa63b8df68d9ba250cd4da5a55a178e8339f9505d81638af3f1")


def load_schedule(path: Path = SCHEDULE) -> dict[str, Any]:
    """Verify the exact schedule and unchanged within-phase task orders."""
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SCHEDULE_SHA256:
        raise ValueError("counterbalanced schedule hash mismatch")
    protocol = json.loads(payload)["protocol"]
    corpus = base.load_corpus()
    orders = protocol["orders"]
    if [tuple(order["phase_sequence"]) for order in orders] != list(
            itertools.permutations(range(3))):
        raise ValueError("schedule must contain all six phase permutations")
    for order in orders:
        expected = [task["id"] for phase in order["phase_sequence"]
                    for task in corpus["tasks"] if task["phase"] == phase]
        if order["task_ids"] != expected:
            raise ValueError("schedule changed a within-phase task order")
    if (tuple(protocol["arms"]) != base.ARMS
            or tuple(protocol["budgets"]) != base.BUDGETS
            or tuple(protocol["model_seeds"]) != base.SEEDS):
        raise ValueError("counterbalanced control matrix changed")
    return protocol


def summarize(rows: list[dict[str, Any]], setup: dict[str, float],
              orders: list[dict[str, Any]]) -> dict[str, Any]:
    """Charge inspection and separate family from presentation position."""
    per_order = []
    for order in orders:
        key = order["order_id"]
        selected = [row for row in rows if row["order_id"] == key]
        charged = [{**row, "reset_cpu_seconds": row["reset_cpu_seconds"]
                    + row["diagnostic_cpu_seconds"]} for row in selected]
        costs = {name.removeprefix(key + ":"): value
                 for name, value in setup.items()
                 if name.startswith(key + ":")}
        summary = base.summarize(charged, costs)
        per_order.append({"order_id": key,
                          "phase_sequence": order["phase_sequence"],
                          **summary})
    strata: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["order_id"], row["budget"], row["arm"], row["seed"],
                row["phase"], row["phase_position"])].append(row)
    breakdown = []
    for key, selected in sorted(strata.items()):
        breakdown.append({
            **dict(zip(("order_id", "budget", "arm", "seed", "phase",
                        "phase_position"), key)),
            "tasks": len(selected),
            "accepted": sum(row["accepted"] for row in selected),
            "first_lane_correct": sum(row["first_lane_correct"]
                                      for row in selected),
            "candidate_evaluations": sum(row["candidate_evaluations"]
                                         for row in selected),
            "updates": sum(row["policy_work"].get("updates", 0)
                           for row in selected),
            "cold_mean_probabilities": [sum(
                row["diagnostic_before"]["probabilities"][i]
                for row in selected) / len(selected) for i in range(3)],
            "cold_mean_entropy": sum(
                row["diagnostic_before"]["entropy_nats"]
                for row in selected) / len(selected),
            "task_mean_reward": sum(row["mean_reward"] for row in selected)
                                / len(selected),
            "attempt_mean_reward": sum(row["reward_sum"] for row in selected)
                                   / sum(row["candidate_evaluations"]
                                         for row in selected),
        })

    def aggregate(order: dict[str, Any], arm: str, metric: str) -> int:
        return sum(row[metric] for row in order["per_seed"]
                   if row["arm"] == arm and row["budget"] == 144)

    strict = all(all(
        aggregate(order, "persistent_snn", metric)
        < aggregate(order, control, metric)
        for control in ("frozen_snn", "episode_reset_snn")
        for metric in ("accepted", "first_lane_correct"))
        for order in per_order)
    pooled = all(
        sum(aggregate(order, "persistent_snn", "accepted")
            for order in per_order)
        < sum(aggregate(order, control, "accepted") for order in per_order)
        and sum(aggregate(order, "persistent_snn", "first_lane_correct")
                for order in per_order)
        <= sum(aggregate(order, control, "first_lane_correct")
               for order in per_order)
        for control in ("frozen_snn", "episode_reset_snn"))
    deficits = sum(aggregate(order, "persistent_snn", "accepted")
                   < aggregate(order, "episode_reset_snn", "accepted")
                   for order in per_order)
    return {"per_order": per_order, "phase_strata": breakdown,
            "diagnosis": {"strict_all_orders_underperformance": strict,
                          "pooled_regression": pooled,
                          "orders_below_reset_acceptance": deficits,
                          "exploratory_mechanism_trigger": pooled
                          and deficits >= 4},
            "all_order_gates": {name: all(order["gates"][name]
                                          for order in per_order)
                                for name in ("adaptation", "snn_specific",
                                             "streaming_deployment")},
            "scope": "Six orders of one 48-task stream; three model seeds"}


def run(output: Path) -> dict[str, Any]:
    """Execute every frozen order with fresh groups; preserve raw failures."""
    protocol, corpus = load_schedule(), base.load_corpus()
    tasks = {task["id"]: task for task in corpus["tasks"]}
    output.mkdir(parents=True, exist_ok=False)
    sources = base.source_manifest()
    compact, setup = [], {}
    started = time.perf_counter()
    with gzip.open(output / "episodes.jsonl.gz", "wt") as stream:
        for order in protocol["orders"]:
            identity = order["order_id"]
            for budget in base.BUDGETS:
                for seed in base.SEEDS:
                    for arm in base.ARMS:
                        tick = time.process_time()
                        policy = base._policy(arm, seed)
                        setup[f"{identity}:{budget}:{arm}:{seed}"] = (
                            time.process_time() - tick)
                        draw_offset = 0
                        for index, task_id in enumerate(order["task_ids"]):
                            task = tasks[task_id]
                            tick = time.process_time()
                            if policy and arm == "episode_reset_snn":
                                policy.reset_learning()
                            reset_cpu = time.process_time() - tick
                            tick = time.process_time()
                            diagnostic = inspect_policy(policy)
                            diagnostic_cpu = time.process_time() - tick
                            before = policy.metrics() if policy else {}
                            observed = tuple(tuple(row)
                                             for row in task["observed"])
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
                                "reward_sum": sum(rewards),
                                "mean_reward": sum(rewards) / len(rewards),
                                **result, **verdict,
                            }
                            draw_offset = row["rng_draw_offset_after"]
                            stream.write(json.dumps(row, sort_keys=True,
                                                    allow_nan=False) + "\n")
                            compact.append({key: value
                                            for key, value in row.items()
                                            if key != "attempts"})
                        print(identity, budget, seed, arm, flush=True)
    if sources != base.source_manifest():
        raise RuntimeError("source changed during counterbalanced run")
    summary = summarize(compact, setup, protocol["orders"])
    manifest = {
        "source_files": sources, "source_unchanged": True,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "schedule_sha256": SCHEDULE_SHA256,
        "corpus_sha256": base.CORPUS_SHA256,
        "episode_rows": len(compact), "setup_cpu_seconds": setup,
        "wall_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "raw_sha256": hashlib.sha256(
            (output / "episodes.jsonl.gz").read_bytes()).hexdigest(),
        "cpu_scope": "setup/reset + policy/execution/update/checker + "
                     "cold diagnostic; excludes compression/provenance",
    }
    for name, value in (("summary.json", summary),
                        ("manifest.json", manifest)):
        (output / name).write_text(json.dumps(value, indent=2,
                                             sort_keys=True) + "\n")
    return summary
