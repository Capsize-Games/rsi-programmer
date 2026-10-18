"""Independently audit a frozen observed-context comparison bundle."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from rsi_programmer import online_benchmark as base
from rsi_programmer import online_counterbalanced as common
from rsi_programmer.online_observed_context import (
    CONTEXT_COUNT,
    CONTEXT_SCALE,
    HIDDEN_COUNT,
    OBSERVED_ROWS,
    PROJECTION_SEED_PREFIX,
    STEPS,
    THRESHOLD,
    BETA,
    ObservedContextPolicy,
)
from rsi_programmer.online_spiking import ACTION_ORDER, MAX_WEIGHT
from scripts.audit_online_counterbalanced import _audit_summary, _equal
from scripts.audit_online_graph import (
    _close,
    _policy,
    _sha256,
    audit_episode_row,
    verify_source_hashes,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "fixtures" / "online_observed_context_v1.json"
# Replace with the final premeasurement protocol digest before publication.
PROTOCOL_SHA256 = (
    "d8897dc237ceed1a8781dd2dd828b4507842bbdf53473558edffa928fe6485fb")
SCHEDULE_SHA256 = common.SCHEDULE_SHA256
CORPUS_SHA256 = base.CORPUS_SHA256
TOLERANCE = 1e-12


def _protocol_payload() -> dict[str, Any]:
    """Verify and return the frozen observed-context protocol object."""
    payload = PROTOCOL_PATH.read_bytes()
    if _sha256(payload) != PROTOCOL_SHA256:
        raise ValueError("observed-context protocol SHA mismatch")
    document = json.loads(payload)
    protocol = document.get("protocol", document)
    source_corpus = protocol.get("source_corpus", {})
    source_schedule = protocol.get("source_schedule", {})
    corpus_digest = protocol.get(
        "source_corpus_sha256", source_corpus.get("sha256"))
    schedule_digest = protocol.get(
        "source_schedule_sha256", source_schedule.get("sha256"))
    if (corpus_digest != CORPUS_SHA256
            or schedule_digest != SCHEDULE_SHA256
            or tuple(protocol.get("arms", ())) != base.ARMS
            or tuple(protocol.get("model_seeds", ())) != base.SEEDS
            or tuple(protocol.get("budgets", ())) != base.BUDGETS):
        raise ValueError("observed-context matrix or source pin changed")
    context = protocol.get("context", {})
    if (context.get("feature_count") != CONTEXT_COUNT
            or context.get("hidden_count") != HIDDEN_COUNT
            or context.get("normalization") != "v / (1 + abs(v))"
            or context.get("flattening") != "row-major triples, then rows"
            or context.get("held_rows_or_outputs_available") is not False
            or context.get("answer_metadata_available_as_choice_features")
            is not False
            or context.get("episode_id_semantics") != (
                "Opaque string for boundary validation only; never encoded "
                "or used in choices/updates.")
            or context.get("trainable") is not False):
        raise ValueError("observed-context information contract changed")
    capacity = protocol.get("capacity_diagnostic", {})
    if (capacity.get("contexts") != 48
            or capacity.get("run_before_measurement") is not True
            or capacity.get("diagnostic_only") is not True
            or capacity.get("no_retuning_or_exclusion_based_on_result")
            is not True):
        raise ValueError("capacity side-study contract changed")
    return protocol


def _projection(seed: int) -> list[list[int]]:
    """Rebuild the fixed context matrix independently from its generator."""
    rng = random.Random(f"{PROJECTION_SEED_PREFIX}{seed}")
    return [[rng.choice((-1, 0, 1)) for _ in range(HIDDEN_COUNT)]
            for _ in range(CONTEXT_COUNT)]


def _projection_digest(matrix: list[list[int]]) -> str:
    """Hash the compact JSON representation pinned by the protocol."""
    payload = json.dumps(matrix, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return _sha256(payload)


def _normalized_context(
    observed: tuple[tuple[int, int, int], ...],
) -> list[float]:
    """Normalize the current six observed triples in row-major order."""
    if (len(observed) != OBSERVED_ROWS
            or any(len(row) != 3
                   or any(type(value) is not int for value in row)
                   for row in observed)):
        raise ValueError("invalid observed rows for context")
    return [value / (1 + abs(value))
            for row in observed for value in row]


def _context_current(context: list[float],
                     matrix: list[list[int]]) -> list[float]:
    """Independently project normalized rows into hidden-cell currents."""
    if len(context) != CONTEXT_COUNT:
        raise ValueError("context width mismatch")
    return [sum(matrix[index][neuron] * context[index]
                for index in range(CONTEXT_COUNT)) / CONTEXT_SCALE
            for neuron in range(HIDDEN_COUNT)]


def _context_diagnostic(
    policy: ObservedContextPolicy | None,
    context: list[float] | None,
    current: list[float] | None,
) -> dict[str, Any]:
    """Recompute cold context-conditioned rates and readout diagnostics."""
    if policy is None:
        probabilities = [1.0, 0.0, 0.0]
        logits = None
        baseline = None
        weights: list[float] = []
    else:
        if context is None or current is None:
            raise ValueError("neural diagnostic lacks observed context")
        state = (policy.weight_hash(), copy.deepcopy(policy._rng.getstate()),
                 policy._membrane[:], policy._baseline,
                 copy.deepcopy(policy._pending_context),
                 copy.deepcopy(policy._active_context),
                 copy.deepcopy(policy._context_current))
        rates = [0.0] * HIDDEN_COUNT
        membrane = [0.0] * HIDDEN_COUNT
        for _step in range(STEPS):
            for neuron in range(HIDDEN_COUNT):
                base_current = sum(
                    weight * feature
                    for weight, feature in zip(
                        policy._encoder[neuron], (1.0, 0.0, 0.0,
                                                  0.0, 1.0, 0.0)))
                voltage = (membrane[neuron] * BETA
                           + (base_current + current[neuron]))
                spike = int(policy._spiking and voltage >= THRESHOLD)
                if spike:
                    voltage = 0.0
                membrane[neuron] = voltage
                rates[neuron] += (spike if policy._spiking
                                  else math.tanh(voltage))
        rates = [value / STEPS for value in rates]
        logits = [sum(weight * rate for weight, rate in zip(
                    policy._weights[index], rates)) + policy._bias[index]
                  for index in range(len(ACTION_ORDER))]
        maximum = max(logits)
        exponentials = [math.exp(value - maximum) for value in logits]
        total = sum(exponentials)
        probabilities = [value / total for value in exponentials]
        baseline = float(policy._baseline)
        weights = [value for row in policy._weights for value in row]
        weights.extend(policy._bias)
        after = (policy.weight_hash(), policy._rng.getstate(),
                 policy._membrane[:], policy._baseline,
                 policy._pending_context, policy._active_context,
                 policy._context_current)
        if state != after:
            raise ValueError("context diagnostic mutated policy")
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
    return {
        "logits": logits,
        "probabilities": probabilities,
        "entropy_nats": entropy,
        "max_probability": max(probabilities),
        "readout_l2": math.sqrt(sum(value * value for value in weights)),
        "clipped_weights": sum(abs(value) >= MAX_WEIGHT for value in weights),
        "baseline": baseline,
    }


def _validate_projection_fixture(protocol: dict[str, Any]) -> None:
    """Require every seed's exposed matrix and hash to match the generator."""
    context = protocol.get("context", {})
    matrices = context.get("projection_matrices")
    hashes = context.get("projection_matrix_sha256")
    expected_seeds = {str(seed) for seed in base.SEEDS}
    if not isinstance(matrices, dict) or set(matrices) != expected_seeds:
        raise ValueError("context projection matrices missing or extra")
    if not isinstance(hashes, dict) or set(hashes) != expected_seeds:
        raise ValueError("context projection hashes missing or extra")
    for seed in base.SEEDS:
        matrix = _projection(seed)
        if matrices[str(seed)] != matrix:
            raise ValueError("context projection matrix does not replay")
        digest = _projection_digest(matrix)
        if hashes[str(seed)] != digest:
            raise ValueError("context projection digest does not replay")


def _validate_context_row(
    row: dict[str, Any], task: dict[str, Any],
    policy: ObservedContextPolicy | None,
    metrics_before_context: dict[str, Any] | None,
) -> tuple[list[float] | None, list[float] | None]:
    """Check observed-only context preparation, work, and diagnostic values."""
    if policy is None:
        if metrics_before_context is not None:
            raise ValueError("streaming row has context policy metrics")
        expected = {"context_input_sha256": None,
                    "context_projection_sha256": None,
                    "context_work": {}}
        context, current = None, None
    else:
        observed = tuple(tuple(values) for values in task["observed"])
        context = _normalized_context(observed)
        matrix = _projection(row["seed"])
        current = _context_current(context, matrix)
        if (policy._pending_context != context
                or policy._context_current != current
                or policy._active_context is not None):
            raise ValueError("policy does not hold current observed context")
        input_digest = _sha256(json.dumps(
            context, separators=(",", ":"), allow_nan=False).encode())
        projection_digest = _projection_digest(matrix)
        if projection_digest != policy._context_projection_hash:
            raise ValueError("policy context projection hash mismatch")
        expected = {
            "context_input_sha256": input_digest,
            "context_projection_sha256": projection_digest,
            "context_work": {"observation_scalars": 18,
                              "projection_mac": 288},
        }
        if metrics_before_context is None:
            raise ValueError("neural context metrics were not sampled")
        metrics_after_context = policy.metrics()
        actual_work = {
            "observation_scalars": (
                metrics_after_context["observation_scalars"]
                - metrics_before_context["observation_scalars"]),
            "projection_mac": (metrics_after_context["projection_mac"]
                               - metrics_before_context["projection_mac"]),
        }
        if actual_work != expected["context_work"]:
            raise ValueError("observed-context setup work counters mismatch")
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"context row {key} mismatch")
    context_cpu = row.get("context_cpu_seconds")
    if (isinstance(context_cpu, bool)
            or not isinstance(context_cpu, (int, float))
            or not math.isfinite(float(context_cpu)) or context_cpu < 0):
        raise ValueError("invalid context CPU time")
    diagnostic_cpu = row.get("diagnostic_cpu_seconds")
    if (isinstance(diagnostic_cpu, bool)
            or not isinstance(diagnostic_cpu, (int, float))
            or not math.isfinite(float(diagnostic_cpu))
            or diagnostic_cpu < 0):
        raise ValueError("invalid cold diagnostic CPU time")
    _equal(row.get("diagnostic_before"),
           _context_diagnostic(policy, context, current),
           "context diagnostic")
    return context, current


def _capacity_rates(
    task: dict[str, Any], seed: int, spiking: bool,
    encoder: list[list[float]],
) -> tuple[float, ...]:
    """Compute one cold six-step code using the fixed projection directly."""
    observed = tuple(tuple(values) for values in task["observed"])
    current = _context_current(_normalized_context(observed),
                               _projection(seed))
    features = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    membrane = [0.0] * HIDDEN_COUNT
    totals = [0.0] * HIDDEN_COUNT
    for _step in range(STEPS):
        for neuron in range(HIDDEN_COUNT):
            base_current = sum(weight * feature
                               for weight, feature in zip(
                                   encoder[neuron], features))
            voltage = (membrane[neuron] * BETA
                       + (base_current + current[neuron]))
            spike = int(spiking and voltage >= THRESHOLD)
            if spike:
                voltage = 0.0
            membrane[neuron] = voltage
            totals[neuron] += (spike if spiking else math.tanh(voltage))
    return tuple(value / STEPS for value in totals)


def _audit_capacity(bundle: Path, manifest: dict[str, Any],
                    tasks: list[dict[str, Any]]) -> None:
    """Validate fixed-capacity rates and their separate CPU receipt."""
    path = bundle / "capacity.json"
    raw = path.read_bytes()
    if _sha256(raw) != manifest.get("capacity_sha256"):
        raise ValueError("capacity side-study SHA mismatch")
    duration = manifest.get("capacity_cpu_seconds")
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration)) or duration < 0):
        raise ValueError("invalid capacity CPU time")
    capacity = json.loads(raw)
    expected_rows = []
    for seed in base.SEEDS:
        probe = base._policy("frozen_snn", seed)
        assert probe is not None
        for spiking in (True, False):
            rates = [_capacity_rates(task, seed, spiking, probe._encoder)
                     for task in tasks]
            rounded = {tuple(round(value, 12) for value in code)
                       for code in rates}
            expected_rows.append({
                "seed": seed, "spiking": spiking,
                "distinct_cold_codes": len(rounded), "tasks": len(tasks),
                "rates": rates,
            })
    expected = {"rows": expected_rows,
                "scope": "Fixed-encoder side study; no tuning"}
    _equal(capacity, expected, "capacity")


def _validate_row(
    row: dict[str, Any], order: dict[str, Any], task: dict[str, Any],
    stream_index: int, budget: int, draw_offset: int,
    policy: ObservedContextPolicy | None,
    metrics_before_context: dict[str, Any] | None,
) -> int:
    """Replay context setup, diagnostics, and full task execution."""
    expected = {
        "order_id": order["order_id"],
        "phase_sequence": order["phase_sequence"],
        "stream_index": stream_index,
        "phase_position": stream_index // 16,
        "phase_ordinal": stream_index % 16,
        "phase": task["phase"], "task_id": task["id"],
        "root": task["root"], "budget": budget,
        "arm": row["arm"], "seed": row["seed"],
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"counterbalanced row {key} mismatch")
    if row.get("rng_draw_offset_before") != draw_offset:
        raise ValueError("incoming RNG draw offset mismatch")
    _validate_context_row(row, task, policy, metrics_before_context)
    if policy is not None:
        if row.get("weight_before") != policy.weight_hash():
            raise ValueError("pre-episode weight hash mismatch")
        if row["arm"] == "episode_reset_snn":
            original = ObservedContextPolicy(seed=row["seed"])
            if row["weight_before"] != original.weight_hash():
                raise ValueError("episode-reset arm carried readout weights")
    elif (row.get("weight_before") is not None
          or row.get("weight_after") is not None
          or row.get("policy_work") != {}):
        raise ValueError("streaming row contains context policy state")
    audit_episode_row(row, task, budget, policy)
    decisions = row.get("decision_calls")
    minimum_decisions = 1 if policy is not None else 0
    if (type(decisions) is not int or decisions < minimum_decisions
            or row.get("rng_draw_offset_after") != draw_offset + decisions):
        raise ValueError("outgoing RNG draw offset mismatch")
    rewards = [attempt["reward"] for attempt in row["attempts"]]
    if not rewards:
        raise ValueError("empty attempt reward list")
    _close(row.get("reward_sum"), sum(rewards), "reward sum")
    _close(row.get("mean_reward"), sum(rewards) / len(rewards),
           "task mean reward")
    return draw_offset + decisions


def audit_bundle(bundle: Path, repo_root: Path = ROOT) -> dict[str, Any]:
    """Verify provenance, context integrity, every replay, and summary."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    protocol = _protocol_payload()
    if manifest.get("context_protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("manifest context protocol SHA mismatch")
    if manifest.get("schedule_sha256") != SCHEDULE_SHA256:
        raise ValueError("manifest schedule SHA mismatch")
    if manifest.get("corpus_sha256") != CORPUS_SHA256:
        raise ValueError("manifest corpus SHA mismatch")
    schedule_bytes = (repo_root / "fixtures/online_counterbalanced_v1.json"
                      ).read_bytes()
    if _sha256(schedule_bytes) != SCHEDULE_SHA256:
        raise ValueError("counterbalanced schedule SHA mismatch")
    corpus_bytes = (repo_root / "fixtures/online_graph_v1.json").read_bytes()
    if _sha256(corpus_bytes) != CORPUS_SHA256:
        raise ValueError("online corpus SHA mismatch")
    if manifest.get("source_unchanged") is not True:
        raise ValueError("runner did not attest stable sources")
    if manifest.get("information_set") != (
            "Current observed examples; no held/phase/root"):
        raise ValueError("runner information-set declaration changed")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("source hash manifest missing")
    verify_source_hashes(source_files, repo_root)
    raw_path = bundle / "episodes.jsonl.gz"
    if _sha256(raw_path.read_bytes()) != manifest.get("raw_sha256"):
        raise ValueError("episode stream SHA mismatch")
    corpus = json.loads(corpus_bytes)
    tasks = corpus["tasks"]
    _validate_projection_fixture(protocol)
    orders = protocol.get("orders")
    base_orders = json.loads(schedule_bytes)["protocol"]["orders"]
    if orders != base_orders:
        raise ValueError("context schedule differs from frozen base schedule")
    expected_rows = len(orders) * len(base.BUDGETS) * len(base.SEEDS) \
        * len(base.ARMS) * len(tasks)
    if manifest.get("episode_rows") != expected_rows:
        raise ValueError("manifest episode count mismatch")
    task_by_id = {task["id"]: task for task in tasks}
    sequence = [(order, budget, seed, arm, task_id)
                for order in orders for budget in base.BUDGETS
                for seed in base.SEEDS for arm in base.ARMS
                for task_id in order["task_ids"]]
    policies: dict[tuple[str, int, int, str],
                   ObservedContextPolicy | None] = {}
    offsets: dict[tuple[str, int, int, str], int] = {}
    counts: dict[tuple[str, int, int, str], int] = {}
    compact: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str, str]] = set()
    iterator = iter(sequence)
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                order, budget, seed, arm, task_id = next(iterator)
            except StopIteration as exc:
                raise ValueError("extra context result row") from exc
            row = json.loads(line)
            identity = (order["order_id"], budget, seed, arm, task_id)
            if (row.get("order_id"), row.get("budget"), row.get("seed"),
                    row.get("arm"), row.get("task_id")) != identity:
                raise ValueError(
                    f"context row order mismatch at {line_number}")
            if identity in seen:
                raise ValueError("duplicate context result row")
            seen.add(identity)
            group = order["order_id"], budget, seed, arm
            if group not in policies:
                policies[group] = ObservedContextPolicy(
                    seed=seed, learning=arm != "frozen_snn",
                    spiking=arm != "online_nonspiking") if arm != \
                    "grammar_streaming" else None
                offsets[group] = 0
                counts[group] = 0
            policy = policies[group]
            if policy is not None and arm == "episode_reset_snn":
                policy.reset_learning()
            task = task_by_id[task_id]
            metrics_before_context = (policy.metrics()
                                      if policy is not None else None)
            if policy is not None:
                observed = tuple(tuple(values) for values in task["observed"])
                policy.set_observed_rows(observed)
            index = counts[group]
            offsets[group] = _validate_row(
                row, order, task, index, budget, offsets[group], policy,
                metrics_before_context)
            counts[group] += 1
            compact.append({key: value for key, value in row.items()
                            if key != "attempts"})
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("incomplete context matrix")
    if len(seen) != expected_rows:
        raise ValueError("incomplete context matrix")
    _audit_capacity(bundle, manifest, tasks)
    analysis_rows = [{**row,
                      "reset_cpu_seconds": row["reset_cpu_seconds"]
                      + row["context_cpu_seconds"]}
                     for row in compact]
    audited = _audit_summary(bundle, manifest, analysis_rows, orders, protocol)
    full_cpu = (manifest["capacity_cpu_seconds"]
                + sum(manifest["setup_cpu_seconds"].values())
                + sum(row["reset_cpu_seconds"] + row["context_cpu_seconds"]
                      + row["diagnostic_cpu_seconds"]
                      + row["episode_cpu_seconds"]
                      + row["checker_cpu_seconds"] for row in compact))
    _close(manifest.get("full_experiment_cpu_seconds"), full_cpu,
           "full experiment CPU")
    return {**audited, "context_protocol_sha256": PROTOCOL_SHA256,
            "capacity_sha256": manifest["capacity_sha256"],
            "capacity_cpu_seconds": manifest["capacity_cpu_seconds"],
            "full_experiment_cpu_seconds": full_cpu}


def main() -> None:
    """Audit one completed context bundle."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_bundle(args.bundle), sort_keys=True))


if __name__ == "__main__":
    main()
