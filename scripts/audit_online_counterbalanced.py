"""Independently audit the counterbalanced online-graph matrix."""
from __future__ import annotations

import copy
import gzip
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from rsi_programmer import online_benchmark as base
from rsi_programmer.online_spiking import (
    ACTION_ORDER,
    BETA,
    HIDDEN_COUNT,
    MAX_WEIGHT,
    STEPS,
    THRESHOLD,
    OnlineSpikingPolicy,
)
from scripts.audit_online_graph import (
    _close,
    _phase_counts,
    _policy,
    _sha256,
    audit_episode_row,
    verify_source_hashes,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "fixtures" / "online_counterbalanced_v1.json"
# Updated together with the final premeasurement schedule pin.
SCHEDULE_SHA256 = (
    "0ffbd7eadac93aa63b8df68d9ba250cd4da5a55a178e8339f9505d81638af3f1"
)
ARMS = base.ARMS
SEEDS = base.SEEDS
BUDGETS = base.BUDGETS
ORDERS = tuple(itertools.permutations(range(3)))
TOLERANCE = 1e-12


def _diagnostic(policy: OnlineSpikingPolicy | None) -> dict[str, Any]:
    """Independently recompute the cold-membrane readout without mutation."""
    if policy is None:
        probabilities = [1.0, 0.0, 0.0]
        logits = None
        baseline = None
        values: list[float] = []
    else:
        weight_hash = policy.weight_hash()
        rng_state = copy.deepcopy(policy._rng.getstate())
        membrane = policy._membrane[:]
        baseline_before = policy._baseline
        features = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        neuron_membrane = [0.0] * HIDDEN_COUNT
        counts = [0.0] * HIDDEN_COUNT
        for _step in range(STEPS):
            for neuron in range(HIDDEN_COUNT):
                current = sum(weight * feature
                              for weight, feature in zip(
                                  policy._encoder[neuron], features))
                voltage = BETA * neuron_membrane[neuron] + current
                if policy._spiking and voltage >= THRESHOLD:
                    spike = 1
                    voltage = 0.0
                else:
                    spike = 0
                neuron_membrane[neuron] = voltage
                counts[neuron] += (spike if policy._spiking
                                   else math.tanh(voltage))
        rates = [count / STEPS for count in counts]
        logits = [sum(weight * rate
                      for weight, rate in zip(policy._weights[index], rates))
                  + policy._bias[index]
                  for index in range(len(ACTION_ORDER))]
        maximum = max(logits)
        exp_values = [math.exp(value - maximum) for value in logits]
        denominator = sum(exp_values)
        probabilities = [value / denominator for value in exp_values]
        baseline = float(policy._baseline)
        values = [value for row in policy._weights for value in row]
        values.extend(policy._bias)
        if (policy.weight_hash() != weight_hash
                or policy._rng.getstate() != rng_state
                or policy._membrane != membrane
                or policy._baseline != baseline_before):
            raise ValueError("diagnostic mutated policy state")
    entropy = -sum(value * math.log(value)
                   for value in probabilities if value > 0.0)
    return {
        "logits": logits,
        "probabilities": probabilities,
        "entropy_nats": entropy,
        "max_probability": max(probabilities),
        "readout_l2": math.sqrt(sum(value * value for value in values)),
        "clipped_weights": sum(abs(value) >= MAX_WEIGHT for value in values),
        "baseline": baseline,
    }


def _equal(actual: Any, expected: Any, label: str) -> None:
    """Compare nested diagnostic and summary data with numeric tolerance."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label} keys mismatch")
        for key, value in expected.items():
            _equal(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, (list, tuple)):
        if (not isinstance(actual, (list, tuple))
                or len(actual) != len(expected)):
            raise ValueError(f"{label} length mismatch")
        for index, value in enumerate(expected):
            _equal(actual[index], value, f"{label}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        _close(actual, float(expected), label)
    elif actual != expected:
        raise ValueError(f"{label} mismatch")


def _validate_schedule(protocol: dict[str, Any],
                       tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Verify all phase permutations and their original task subsequences."""
    orders = protocol.get("orders")
    if not isinstance(orders, list) or len(orders) != 6:
        raise ValueError("schedule must contain six orders")
    sequences = [tuple(item.get("phase_sequence", ())) for item in orders]
    if sequences != list(ORDERS):
        raise ValueError("schedule is not the six canonical permutations")
    if (tuple(protocol.get("arms", ())) != ARMS
            or tuple(protocol.get("budgets", ())) != BUDGETS
            or tuple(protocol.get("model_seeds", ())) != SEEDS
            or protocol.get("fresh_policy_scope") != [
                "phase_order", "model_seed", "arm", "budget"]):
        raise ValueError("counterbalanced matrix or reset scope changed")
    task_by_id = {task["id"]: task for task in tasks}
    base_phase_ids = {
        phase: [task["id"] for task in tasks if task["phase"] == phase]
        for phase in range(3)}
    seen_orders: set[str] = set()
    for order in orders:
        identity = "-".join(str(value) for value in order["phase_sequence"])
        if order.get("order_id") != identity or identity in seen_orders:
            raise ValueError("order ID mismatch or duplicate")
        seen_orders.add(identity)
        expected = [task_id for phase in order["phase_sequence"]
                    for task_id in base_phase_ids[phase]]
        if order.get("task_ids") != expected:
            raise ValueError("schedule changed phase task order")
        if len(set(expected)) != len(tasks):
            raise ValueError("order omitted or repeated a corpus task")
        expected_phases = [phase for phase in order["phase_sequence"]
                           for _ in range(16)]
        if any(task_by_id[task_id]["phase"] != expected_phases[index]
               for index, task_id in enumerate(expected)):
            raise ValueError("schedule phase labels do not align")
    return orders


def _expected_groups(rows: list[dict[str, Any]],
                     setup: dict[str, float],
                     order_id: str) -> list[dict[str, Any]]:
    """Recompute per-seed aggregate metrics and CPU charges for one order."""
    selected = [row for row in rows if row["order_id"] == order_id]
    result: list[dict[str, Any]] = []
    group_keys = sorted((budget, arm, seed) for budget in BUDGETS
                        for arm in ARMS for seed in SEEDS)
    for budget, arm, seed in group_keys:
        items = [row for row in selected if row["budget"] == budget
                 and row["arm"] == arm and row["seed"] == seed]
        key = f"{order_id}:{budget}:{arm}:{seed}"
        if len(items) != 48 or key not in setup:
            raise ValueError("order group incomplete")
        cpu = setup[key] + sum(
            row["episode_cpu_seconds"]
            + row["checker_cpu_seconds"]
            + row["reset_cpu_seconds"]
            + row["diagnostic_cpu_seconds"] for row in items)
        result.append({
            "budget": budget, "arm": arm, "seed": seed,
            "tasks": len(items),
            "accepted": sum(bool(row["accepted"]) for row in items),
            "observed_only_false_positives": sum(
                bool(row["observed_solved"] and not row["accepted"])
                for row in items),
            "first_lane_correct": sum(
                bool(row["first_lane_correct"]) for row in items),
            "candidate_evaluations": sum(
                row["candidate_evaluations"] for row in items),
            "node_row_operations": sum(
                row["node_row_operations"] for row in items),
            "total_cpu_seconds": cpu,
            "policy_cpu_seconds": sum(
                row["policy_cpu_seconds"] for row in items),
            "update_cpu_seconds": sum(
                row["update_cpu_seconds"] for row in items),
            "phase_counts": _phase_counts(items),
        })
    return result


def _order_gates(rows: list[dict[str, Any]]) -> dict[str, bool]:
    """Independently calculate the frozen v1 deployment/adaptation gates."""
    def total(arm: str, field: str, budget: int = 144) -> float:
        return sum(row[field] for row in rows
                   if row["arm"] == arm and row["budget"] == budget)

    def first_better(arm: str, require_each_seed: bool = True) -> bool:
        pooled = (total("persistent_snn", "first_lane_correct")
                  - total(arm, "first_lane_correct"))
        per_seed = all(
            next(row["first_lane_correct"] for row in rows
                 if row["arm"] == "persistent_snn"
                 and row["budget"] == 144 and row["seed"] == seed)
            > next(row["first_lane_correct"] for row in rows
                   if row["arm"] == arm and row["budget"] == 144
                   and row["seed"] == seed) for seed in SEEDS)
        return pooled >= 14.4 and (per_seed or not require_each_seed)

    def enough_accepts(arm: str, budget: int = 144) -> bool:
        return (total("persistent_snn", "accepted", budget)
                >= total(arm, "accepted", budget))

    def cheap_enough(arm: str, budget: int = 144) -> bool:
        return (total("persistent_snn", "total_cpu_seconds", budget)
                <= total(arm, "total_cpu_seconds", budget))

    adaptation = all(
        first_better(arm) and enough_accepts(arm)
        for arm in ("frozen_snn", "episode_reset_snn"))
    specific = (adaptation
                and first_better("online_nonspiking", require_each_seed=False)
                and enough_accepts("online_nonspiking")
                and cheap_enough("online_nonspiking"))
    streaming = all(
        enough_accepts("grammar_streaming", budget)
        and cheap_enough("grammar_streaming", budget)
        for budget in BUDGETS)
    return {"adaptation": adaptation, "snn_specific": specific,
            "streaming_deployment": streaming}


def _validate_counterbalanced_row(
    row: dict[str, Any], order: dict[str, Any], task: dict[str, Any],
    stream_index: int, budget: int, draw_offset: int,
    policy: OnlineSpikingPolicy | None,
) -> int:
    """Check sequence identity, pure probe, replay, and draw offsets."""
    sequence = order["phase_sequence"]
    position, ordinal = divmod(stream_index, 16)
    expected_metadata = {
        "order_id": order["order_id"], "phase_sequence": sequence,
        "stream_index": stream_index, "phase_position": position,
        "phase_ordinal": ordinal, "phase": task["phase"],
        "task_id": task["id"], "root": task["root"], "budget": budget,
    }
    for key, value in expected_metadata.items():
        if row.get(key) != value:
            raise ValueError(f"counterbalanced row {key} mismatch")
    if row.get("rng_draw_offset_before") != draw_offset:
        raise ValueError("incoming RNG draw offset mismatch")
    expected_diagnostic = _diagnostic(policy)
    _equal(row.get("diagnostic_before"), expected_diagnostic,
           "diagnostic_before")
    diag_cpu = row.get("diagnostic_cpu_seconds")
    if (isinstance(diag_cpu, bool) or not isinstance(diag_cpu, (int, float))
            or not math.isfinite(float(diag_cpu)) or diag_cpu < 0):
        raise ValueError("invalid diagnostic CPU time")
    if policy and row.get("weight_before") != policy.weight_hash():
        raise ValueError("pre-episode weight hash mismatch")
    if policy is None and (
            row.get("weight_before") is not None
            or row.get("weight_after") is not None
            or row.get("policy_work") != {}):
        raise ValueError("streaming row contains policy state or work")
    if policy and row["arm"] == "episode_reset_snn":
        initial = OnlineSpikingPolicy(seed=row["seed"]).weight_hash()
        if policy.weight_hash() != initial:
            raise ValueError("episode-reset policy carried readout weights")
    audit_episode_row(row, task, budget, policy)
    count = row.get("decision_calls")
    if row.get("rng_draw_offset_after") != draw_offset + count:
        raise ValueError("outgoing RNG draw offset mismatch")
    rewards = [attempt["reward"] for attempt in row["attempts"]]
    if not rewards:
        raise ValueError("empty reward trace")
    _close(row.get("reward_sum"), sum(rewards), "reward sum")
    _close(row.get("mean_reward"), sum(rewards) / len(rewards),
           "task mean reward")
    return draw_offset + count


def audit_bundle(bundle: Path, repo_root: Path = ROOT) -> dict[str, Any]:
    """Verify provenance, full order matrix, policy replay and summaries."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    schedule_path = repo_root / "fixtures" / SCHEDULE.name
    schedule_bytes = schedule_path.read_bytes()
    if _sha256(schedule_bytes) != SCHEDULE_SHA256:
        raise ValueError("counterbalanced schedule SHA mismatch")
    if manifest.get("schedule_sha256") != SCHEDULE_SHA256:
        raise ValueError("manifest schedule SHA mismatch")
    corpus_bytes = (repo_root / "fixtures" / "online_graph_v1.json"
                    ).read_bytes()
    if _sha256(corpus_bytes) != base.CORPUS_SHA256:
        raise ValueError("base corpus SHA mismatch")
    if manifest.get("corpus_sha256") != base.CORPUS_SHA256:
        raise ValueError("manifest base corpus SHA mismatch")
    if manifest.get("source_unchanged") is not True:
        raise ValueError("runner did not attest stable sources")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("source hash manifest missing")
    verify_source_hashes(source_files, repo_root)
    raw_path = bundle / "episodes.jsonl.gz"
    raw_bytes = raw_path.read_bytes()
    if _sha256(raw_bytes) != manifest.get("raw_sha256"):
        raise ValueError("episode stream SHA mismatch")
    corpus = json.loads(corpus_bytes)
    tasks = corpus["tasks"]
    protocol = json.loads(schedule_bytes)["protocol"]
    orders = _validate_schedule(protocol, tasks)
    expected_count = len(orders) * len(BUDGETS) * len(SEEDS) * len(ARMS) \
        * len(tasks)
    if manifest.get("episode_rows") != expected_count:
        raise ValueError("manifest episode count mismatch")
    task_by_id = {task["id"]: task for task in tasks}
    expected_sequence = [
        (order, budget, seed, arm, task_id)
        for order in orders for budget in BUDGETS for seed in SEEDS
        for arm in ARMS for task_id in order["task_ids"]]
    compact: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str, str]] = set()
    policies: dict[tuple[str, int, int, str],
                   OnlineSpikingPolicy | None] = {}
    offsets: dict[tuple[str, int, int, str], int] = {}
    group_counts: dict[tuple[str, int, int, str], int] = {}
    groups = iter(expected_sequence)
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            try:
                order, budget, seed, arm, task_id = next(groups)
            except StopIteration as exc:
                raise ValueError("extra result row") from exc
            if (row.get("order_id"), row.get("budget"), row.get("seed"),
                    row.get("arm"), row.get("task_id")) != (
                    order["order_id"], budget, seed, arm, task_id):
                raise ValueError(f"row order/key mismatch at {line_number}")
            key = (order["order_id"], budget, seed, arm, task_id)
            if key in seen:
                raise ValueError("duplicate row")
            seen.add(key)
            group = order["order_id"], budget, seed, arm
            policy = policies.get(group)
            if group not in policies:
                policy = _policy(arm, seed)
                policies[group] = policy
                offsets[group] = 0
                group_counts[group] = 0
            if policy and arm == "episode_reset_snn":
                policy.reset_learning()
            expected_index = group_counts[group]
            new_offset = _validate_counterbalanced_row(
                row, order, task_by_id[task_id], expected_index, budget,
                offsets[group], policy)
            offsets[group] = new_offset
            group_counts[group] += 1
            compact.append({key: value for key, value in row.items()
                            if key != "attempts"})
    if len(seen) != expected_count:
        raise ValueError(f"incomplete episode matrix: {len(seen)}")
    return _audit_summary(bundle, manifest, compact, orders, protocol)


def _audit_summary(bundle: Path, manifest: dict[str, Any],
                   rows: list[dict[str, Any]], orders: list[dict[str, Any]],
                   protocol: dict[str, Any]) -> dict[str, Any]:
    """Recompute phase strata, per-order costs, gates, and diagnosis."""
    setup = manifest.get("setup_cpu_seconds")
    if not isinstance(setup, dict):
        raise ValueError("setup CPU map missing")
    for name, value in setup.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or value < 0):
            raise ValueError("invalid setup CPU value")
    expected_setup_keys = {
        f"{order['order_id']}:{budget}:{arm}:{seed}"
        for order in orders for budget in BUDGETS
        for arm in ARMS for seed in SEEDS}
    if set(setup) != expected_setup_keys:
        raise ValueError("setup CPU keys do not match the frozen matrix")
    per_order: list[dict[str, Any]] = []
    for order in orders:
        identity = order["order_id"]
        seeds = _expected_groups(rows, setup, identity)
        gates = _order_gates(seeds)
        per_order.append({"order_id": identity,
                          "phase_sequence": order["phase_sequence"],
                          "per_seed": seeds, "gates": gates,
                          "scope": (
                              "48 shared synthetic tasks; three model seeds; "
                              "pilot only")})
    strata: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["order_id"], row["budget"], row["arm"], row["seed"],
               row["phase"], row["phase_position"])
        strata[key].append(row)
    phase_rows = []
    for key, selected in sorted(strata.items()):
        evaluations = sum(row["candidate_evaluations"] for row in selected)
        probabilities = [row["diagnostic_before"]["probabilities"]
                         for row in selected]
        phase_rows.append({
            **dict(zip(("order_id", "budget", "arm", "seed", "phase",
                        "phase_position"), key)),
            "tasks": len(selected),
            "accepted": sum(bool(row["accepted"]) for row in selected),
            "first_lane_correct": sum(bool(row["first_lane_correct"])
                                       for row in selected),
            "candidate_evaluations": evaluations,
            "updates": sum(row["policy_work"].get("updates", 0)
                           for row in selected),
            "cold_mean_probabilities": [
                sum(values[index] for values in probabilities) / len(selected)
                for index in range(3)],
            "cold_mean_entropy": sum(
                row["diagnostic_before"]["entropy_nats"]
                for row in selected) / len(selected),
            "task_mean_reward": sum(row["mean_reward"] for row in selected)
                                / len(selected),
            "attempt_mean_reward": sum(row["reward_sum"]
                                       for row in selected) / evaluations,
        })
    strict = all(
        _order_metric(order, "persistent_snn", "accepted")
        < _order_metric(order, control, "accepted")
        and _order_metric(order, "persistent_snn", "first_lane_correct")
        < _order_metric(order, control, "first_lane_correct")
        for order in per_order
        for control in ("frozen_snn", "episode_reset_snn"))
    pooled = all(
        sum(_order_metric(order, "persistent_snn", "accepted")
            for order in per_order)
        < sum(_order_metric(order, control, "accepted")
              for order in per_order)
        and sum(_order_metric(order, "persistent_snn", "first_lane_correct")
                for order in per_order)
        <= sum(_order_metric(order, control, "first_lane_correct")
               for order in per_order)
        for control in ("frozen_snn", "episode_reset_snn"))
    deficits = sum(
        _order_metric(order, "persistent_snn", "accepted")
        < _order_metric(order, "episode_reset_snn", "accepted")
        for order in per_order)
    diagnosis = {
        "strict_all_orders_underperformance": strict,
        "pooled_regression": pooled,
        "orders_below_reset_acceptance": deficits,
        "exploratory_mechanism_trigger": pooled and deficits >= 4,
    }
    gates = {name: all(order["gates"][name] for order in per_order)
             for name in ("adaptation", "snn_specific",
                          "streaming_deployment")}
    expected = {"per_order": per_order, "phase_strata": phase_rows,
                "diagnosis": diagnosis, "all_order_gates": gates,
                "scope": "Six orders of one 48-task stream; three model seeds"}
    actual = json.loads((bundle / "summary.json").read_text())
    _equal(actual, expected, "summary")
    return {"status": "passed", "episode_rows": len(rows),
            "orders": len(orders), "schedule_sha256": SCHEDULE_SHA256,
            "raw_sha256": manifest["raw_sha256"], "diagnosis": diagnosis}


def _order_metric(order: dict[str, Any], arm: str, field: str) -> int:
    """Sum one primary-budget metric across model seeds in an order."""
    return sum(row[field] for row in order["per_seed"]
               if row["arm"] == arm and row["budget"] == 144)


def main() -> None:
    """Audit one completed result bundle."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_bundle(args.bundle), sort_keys=True))


if __name__ == "__main__":
    main()
