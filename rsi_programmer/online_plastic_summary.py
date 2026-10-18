"""Per-seed and per-order summaries for the plastic-encoder pilot."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

BUDGETS = (144, 432)
SEEDS = (0, 1, 2)


def _totals(items: list[dict[str, Any]], setup_cpu: float,
            probe_cpu: float) -> dict[str, Any]:
    """Aggregate one fresh policy's tasks and separate probe overhead."""
    search_cpu = setup_cpu + sum(
        item["reset_cpu_seconds"] + item["context_cpu_seconds"]
        + item["episode_cpu_seconds"] + item["checker_cpu_seconds"]
        + item["state_hash_cpu_seconds"]
        for item in items)
    policy_work: dict[str, int] = defaultdict(int)
    context_work: dict[str, int] = defaultdict(int)
    for item in items:
        for name, count in item["policy_work"].items():
            policy_work[name] += count
        for name, count in item["context_work"].items():
            context_work[name] += count
    total = {
        "tasks": len(items),
        "accepted": sum(bool(item["accepted"]) for item in items),
        "first_lane_correct": sum(
            bool(item["first_lane_correct"]) for item in items),
        "candidate_evaluations": sum(
            int(item["candidate_evaluations"]) for item in items),
        "updates": sum(item["policy_work"].get("updates", 0)
                        for item in items),
        "plastic_encoder_updates": sum(
            item["policy_work"].get("plastic_encoder_updates", 0)
            for item in items),
        "node_row_operations": sum(
            item["node_row_operations"] for item in items),
        "interpreter_rows": sum(item["interpreter_rows"]
                                 for item in items),
        "policy_work_counters": dict(policy_work),
        "context_work_counters": dict(context_work),
        "setup_cpu_seconds": setup_cpu,
        "search_only_cpu_seconds": search_cpu,
        "retention_probe_cpu_seconds": probe_cpu,
        "total_cpu_seconds": search_cpu + probe_cpu,
    }
    return total


def _passed_margin(ours: dict[str, Any], other: dict[str, Any],
                   margin: float, each_seed: bool) -> bool:
    """Check a first-choice margin within one phase order and budget."""
    if ours["first_lane_correct"] - other["first_lane_correct"] < (
            margin * ours["tasks"]):
        return False
    if not each_seed:
        return True
    return all(ours["seed_metrics"][seed]["first_lane_correct"]
               > other["seed_metrics"][seed]["first_lane_correct"]
               for seed in SEEDS)


def summarize(rows: list[dict[str, Any]],
              setup_cpu: dict[str, float],
              probe_cpu: dict[str, float],
              orders: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all episodes and compute every frozen gate per order."""
    grouped: dict[tuple[str, int, str, int], list[dict[str, Any]]]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["order_id"], row["budget"], row["arm"],
                 row["seed"])].append(row)

    per_order = []
    for order in orders:
        identity = order["order_id"]
        seeds_by_key: dict[tuple[int, str], dict[int, dict[str, Any]]]
        seeds_by_key = defaultdict(dict)
        for budget in BUDGETS:
            for arm in ("persistent_snn", "frozen_snn",
                        "episode_reset_snn", "online_nonspiking",
                        "grammar_streaming", "plastic_snn",
                        "reset_plastic_snn", "plastic_nonspiking"):
                for seed in SEEDS:
                    key = f"{identity}:{budget}:{arm}:{seed}"
                    items = grouped.get((identity, budget, arm, seed), [])
                    metrics = _totals(items, setup_cpu.get(key, 0.0),
                                      probe_cpu.get(key, 0.0))
                    metrics["seed"] = seed
                    seeds_by_key[(budget, arm)][seed] = metrics
        seed_rows = []
        aggregate: dict[tuple[int, str], dict[str, Any]] = {}
        for (budget, arm), seed_map in sorted(seeds_by_key.items()):
            combined = {"budget": budget, "arm": arm,
                        "tasks": sum(item["tasks"]
                                      for item in seed_map.values()),
                        "seed_metrics": seed_map}
            for metric in ("accepted", "first_lane_correct",
                           "candidate_evaluations", "updates",
                           "plastic_encoder_updates",
                           "search_only_cpu_seconds",
                           "retention_probe_cpu_seconds",
                           "total_cpu_seconds"):
                combined[metric] = sum(item[metric]
                                       for item in seed_map.values())
            aggregate[(budget, arm)] = combined
            seed_rows.extend({"budget": budget, "arm": arm, **item}
                             for item in seed_map.values())

        matrix_complete = all(
            len(grouped.get((identity, budget, arm, seed), [])) == 48
            for budget in BUDGETS
            for arm in ("persistent_snn", "frozen_snn",
                        "episode_reset_snn", "online_nonspiking",
                        "grammar_streaming", "plastic_snn",
                        "reset_plastic_snn", "plastic_nonspiking")
            for seed in SEEDS)
        primary = aggregate[(144, "plastic_snn")]
        fixed = aggregate[(144, "persistent_snn")]
        reset_plastic = aggregate[(144, "reset_plastic_snn")]
        plastic_ann = aggregate[(144, "plastic_nonspiking")]
        primary_vs_fixed = (
            _passed_margin(primary, fixed, 0.10, True)
            and primary["accepted"] >= fixed["accepted"]
            and primary["accepted"] >= reset_plastic["accepted"])
        snn_specific = (
            _passed_margin(primary, plastic_ann, 0.10, False)
            and primary["accepted"] >= plastic_ann["accepted"]
            and primary["total_cpu_seconds"]
            <= plastic_ann["total_cpu_seconds"])
        streaming = all(
            aggregate[(budget, "plastic_snn")]["accepted"]
            >= aggregate[(budget, "grammar_streaming")]["accepted"]
            and aggregate[(budget, "plastic_snn")]["total_cpu_seconds"]
            <= aggregate[(budget, "grammar_streaming")]["total_cpu_seconds"]
            for budget in BUDGETS)
        gates = {
            "plastic_snn_vs_persistent_snn": primary_vs_fixed,
            "plastic_snn_coverage_vs_reset_plastic_snn": (
                primary["accepted"] >= reset_plastic["accepted"]),
            "plastic_snn_vs_plastic_nonspiking": snn_specific,
            "streaming_deployment": streaming,
        }
        if not matrix_complete:
            gates = {name: False for name in gates}
        per_order.append({
            "order_id": identity,
            "matrix_complete": matrix_complete,
            "phase_sequence": order["phase_sequence"],
            "per_seed": sorted(seed_rows,
                                key=lambda item: (item["budget"],
                                                  item["arm"], item["seed"])),
            "pooled": [aggregate[key] for key in sorted(aggregate)],
            "gates": gates,
        })

    gate_names = tuple(per_order[0]["gates"] if per_order else ())
    return {
        "per_order": per_order,
        "all_order_gates": {
            name: all(order["gates"][name] for order in per_order)
            for name in gate_names
        },
        "scope": "One 48-task stream repeated over six orders, three seeds, "
                 "and two budgets; task repetitions are not independent.",
    }


def summarize_retention(rows: list[dict[str, Any]],
                        orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare cold root preference at phase boundaries with final probes."""
    indexed = {(row["order_id"], row["budget"], row["seed"], row["arm"],
                row["completed_tasks"]): row for row in rows}
    summaries = []
    for order in orders:
        sequence = order["phase_sequence"]
        for budget in BUDGETS:
            for seed in SEEDS:
                for arm in ("persistent_snn", "frozen_snn",
                            "episode_reset_snn", "online_nonspiking",
                            "plastic_snn", "reset_plastic_snn",
                            "plastic_nonspiking"):
                    final = indexed[(order["order_id"], budget, seed, arm, 48)]
                    final_by_task = {item["task_id"]: item
                                     for item in final["contexts"]}
                    for phase_index, phase in enumerate(sequence):
                        end = indexed[(order["order_id"], budget, seed, arm,
                                       (phase_index + 1) * 16)]
                        context_map = {item["task_id"]: item
                                       for item in end["contexts"]}
                        selected = [item for item in context_map.values()
                                    if item["phase"] == phase]
                        task_ids = [item["task_id"] for item in selected]
                        if len(task_ids) != 16:
                            raise ValueError(
                                "retention phase must contain 16 contexts")
                        summaries.append({
                            "order_id": order["order_id"],
                            "budget": budget, "seed": seed, "arm": arm,
                            "phase": phase,
                            "phase_end_completed_tasks": (phase_index + 1)
                            * 16,
                            "tasks": len(task_ids),
                            "mean_phase_end_root_probability": sum(
                                context_map[task_id]["root_probability"]
                                for task_id in task_ids) / len(task_ids),
                            "mean_final_root_probability": sum(
                                final_by_task[task_id]["root_probability"]
                                for task_id in task_ids) / len(task_ids),
                        })
    return summaries
