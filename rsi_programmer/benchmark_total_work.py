"""Paired end-to-end work accounting for frozen pre-eval engines."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import spikeforge
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch

from scripts import benchmark_pre_eval as v1
from rsi_programmer.benchmark_policy import state_fingerprint
from rsi_programmer.benchmark_policy_heuristic import (
    ClassicalHeuristicPolicy,
)
from rsi_programmer.benchmark_fit_work import fit_with_work
from rsi_programmer.benchmark_mlp import MLPControl
from rsi_programmer.benchmark_policy_random import SeededRandomPolicy
from rsi_programmer.benchmark_recurrent import RecurrentControl
from rsi_programmer.benchmark_v2_runner import (
    CONTROL_NAMES,
    ROOT,
    _atomic_json,
    _control_for,
    load_frozen_inputs,
    sha256_bytes,
    validate_graph_exact,
)
from rsi_programmer.benchmark_v2_training import build_training_groups
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.pre_eval import pre_evaluation_search

BASELINE_PATH = ROOT / "benchmarks/reference/pre_eval_code_v65.py"
BASELINE_SHA256 = (
    "e5af5fe565f24a3bb08d97c3bf2e7545fbd2a34e83ee99786c2cf95a0b0c49d4"
)
BASELINE_REVISION = "168a388899fa756fb41257faefc6315bc46030dc"
TOTAL_PROTOCOL_SHA256 = (
    "b5399f0ab7f90df519a2b72c1dd3a17706bb4d64538e616e44512cbe76a1130"
)
ENGINE_NAMES = ("baseline", "optimized")
COUNT_KEYS = (
    "signature_evaluations", "tried", "generated_descriptors",
    "scored_descriptors", "feature_builds", "policy_batches",
    "policy_batch_ticks", "policy_row_steps", "snn_steps",
    "graph_validation_calls", "oracle_calls",
    "operand_signature_evaluations", "operand_vector_lookups",
    "retained_operand_vectors", "committed_nodes",
    "candidate_row_operations", "feature_stat_cache_lookups",
    "feature_stat_cache_hits", "feature_stat_cache_misses",
    "feature_target_range_elements_scanned",
    "feature_operand_elements_scanned",
    "reference_feature_target_range_elements_scanned",
    "reference_feature_operand_elements_scanned",
    "reference_gap_elements_scanned",
)
TARGETS = (0.25, 0.50, 0.75, 1.00)

SearchFn = Callable[..., dict[str, Any]]


def load_baseline_search(path: Path = BASELINE_PATH) -> SearchFn:
    """Import the frozen baseline file as a private module."""
    data = path.read_bytes()
    actual = sha256_bytes(data)
    if actual != BASELINE_SHA256:
        raise ValueError("baseline pre-eval source hash mismatch")
    spec = importlib.util.spec_from_file_location(
        "_benchmark_total_work_baseline_v65", path,
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not load frozen baseline module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    search_fn = getattr(module, "pre_evaluation_search", None)
    if not callable(search_fn):
        raise ValueError("baseline source has no pre_evaluation_search")
    return search_fn


def engine_order(seed: int, control: str, task_id: str, budget: int,
                 repeat: int, order_seed: int = 83080) -> tuple[str, str]:
    """Return a balanced pair; repeat one reverses repeat zero."""
    if repeat not in (0, 1):
        raise ValueError("exactly two timing repeats are supported")
    key = f"{order_seed}:{seed}:{task_id}:{budget}:{control}".encode()
    bit = hashlib.sha256(key).digest()[0] & 1
    first = ENGINE_NAMES[bit]
    if repeat == 1:
        first = "optimized" if first == "baseline" else "baseline"
    second = "optimized" if first == "baseline" else "baseline"
    return first, second


def validate_total_work_protocol(
        protocol: dict[str, Any],
        v2_protocol: dict[str, Any],
        corpus: dict[str, Any]) -> None:
    """Check this run's declared dimensions against the frozen v2 inputs."""
    if protocol.get("protocol_id") != "benchmark-total-work-v1":
        raise ValueError("unexpected total-work protocol ID")
    if protocol.get("corpus", {}).get("sha256") != (
            v2_protocol["corpus"]["sha256"]):
        raise ValueError("total-work protocol does not pin the v2 corpus")
    if protocol.get("status") != "frozen-before-measurement":
        raise ValueError("total-work protocol was not frozen before timing")
    training = protocol.get("training", {})
    if (training.get("seeds") != [0, 1, 2]
            or training.get("epochs") != 40
            or training.get("learning_rate") != 0.05
            or training.get("positive_cap") != 16
            or training.get("negative_cap") != 256
            or training.get("sampling_seed") != 82078):
        raise ValueError("total-work training configuration differs from v2")
    comparison = protocol.get("comparison", {})
    if comparison.get("engines") != list(ENGINE_NAMES):
        raise ValueError("engine axis must be baseline then optimized")
    if comparison.get("baseline_source", {}).get("sha256") != BASELINE_SHA256:
        raise ValueError("protocol baseline hash differs from runner pin")
    model_reference = comparison.get("model_reference", {})
    expected_reference_hash = (
        "47b02c532f5a11faa4808f5ae9646ba976ab8a56d80a0aadfcaf062239d82b6b"
    )
    if model_reference.get("sha256") != expected_reference_hash:
        raise ValueError("model reference hash differs from protocol pin")
    evaluation = protocol.get("evaluation", {})
    if evaluation.get("candidate_scan_budgets") != [512, 2048, 8192, 32768]:
        raise ValueError("root budgets differ from frozen v2 budgets")
    if evaluation.get("repeats") != 2:
        raise ValueError("exactly two timing repeats are required")
    if evaluation.get("engine_order_seed") != 83080:
        raise ValueError("engine order seed differs from protocol")
    if evaluation.get("matrix_rows") != 16128:
        raise ValueError("total-work matrix row count differs from protocol")
    if evaluation.get("tree_bound") != 5:
        raise ValueError("tree bound differs from frozen v2")
    if protocol.get("controls") != list(CONTROL_NAMES):
        raise ValueError("control set or order differs from frozen v2")
    if (protocol.get("aggregation", {}).get("matched_success_targets")
            != list(TARGETS)):
        raise ValueError("matched-success targets differ from protocol")
    if protocol.get("amortization", {}).get("task_counts") != [48, 480, 4800]:
        raise ValueError("amortization task counts differ from protocol")
    # Reuse v2's full validator, including exact row and witness checks.
    from rsi_programmer.benchmark_v2_runner import (
        validate_corpus, validate_protocol,
    )

    validate_protocol(v2_protocol)
    validate_corpus(corpus, v2_protocol)


def _make_models_with_cpu(
        seed: int,
        groups: list[list[tuple[list[float], int]]],
        epochs: int,
        expected_weights: dict[str, str],
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build v2 controls and measure fitting CPU independently per model."""
    construction: dict[str, Any] = {}

    def construct(name: str, factory: Callable[[], Any]) -> Any:
        cpu_start = time.process_time_ns()
        wall_start = time.perf_counter_ns()
        model = factory()
        construction[name] = {
            "cpu_ns": time.process_time_ns() - cpu_start,
            "wall_ns": time.perf_counter_ns() - wall_start,
        }
        return model

    trained_snn = construct(
        "trained_snn",
        lambda: SearchPolicy(seed=seed, hidden=16, steps=6, lr=0.05),
    )
    untrained_snn = construct(
        "untrained_snn",
        lambda: SearchPolicy(seed=seed, hidden=16, steps=6, lr=0.05),
    )
    heuristic = construct("heuristic", ClassicalHeuristicPolicy)
    mlp = construct("mlp", lambda: MLPControl(seed=seed, lr=0.05))
    recurrent = construct(
        "recurrent", lambda: RecurrentControl(seed=seed, lr=0.05),
    )
    models: dict[str, Any] = {
        "heuristic": heuristic,
        "untrained_snn": untrained_snn,
        "trained_snn": trained_snn,
        "mlp": mlp,
        "recurrent": recurrent,
    }
    training: dict[str, Any] = {}
    for name, model in models.items():
        metadata = v1._model_metadata(model)
        before_weights = v1._weights_hash(model)
        before_state = v1._state_hash(model)
        trainable = name in {"trained_snn", "mlp", "recurrent"}
        cpu_start = time.process_time_ns()
        wall_start = time.perf_counter_ns()
        if trainable:
            loss, fit_work = fit_with_work(
                model, groups, epochs,
                int(metadata.get("simulation_steps", 1)),
            )
        else:
            loss = 0.0
            fit_work = {"optimizer_updates": 0, "forward_batches": 0,
                        "scored_feature_rows": 0,
                        "simulation_row_steps": 0}
        fit_cpu_ns = time.process_time_ns() - cpu_start
        fit_wall_ns = time.perf_counter_ns() - wall_start
        after_weights = v1._weights_hash(model)
        after_state = v1._state_hash(model)
        expected = expected_weights.get(name)
        if expected is not None and expected != after_weights:
            raise ValueError(
                f"{name} post-fit weights differ from pinned v2 hash"
            )
        training[name] = {
            "metadata": metadata,
            "weights_before_fit_sha256": before_weights,
            "weights_after_fit_sha256": after_weights,
            "expected_weights_after_fit_sha256": expected,
            "state_fingerprint_before_fit": before_state,
            "state_fingerprint_after_fit": after_state,
            "fit_cpu_ns": fit_cpu_ns,
            "fit_wall_ns": fit_wall_ns,
            "fit_performed": trainable,
            "fit_changed_weights": before_weights != after_weights,
            "loss": float(loss),
            "fit_work": fit_work,
        }
    random_policy = construct(
        "seeded_random", lambda: SeededRandomPolicy(seed),
    )
    random_hash = v1._weights_hash(random_policy)
    random_expected = expected_weights.get("seeded_random")
    if random_expected not in (None, random_hash):
        raise ValueError("seeded_random post-fit weights differ from v2")
    training["seeded_random"] = {
        "metadata": random_policy.metadata(),
        "weights_after_fit_sha256": random_hash,
        "expected_weights_after_fit_sha256": random_expected,
        "fit_cpu_ns": 0,
        "fit_wall_ns": 0,
        "fit_performed": False,
    }
    streaming_expected = expected_weights.get("streaming")
    if streaming_expected not in (None, "none"):
        raise ValueError("streaming has unexpected weight hash")
    training["streaming"] = {
        "metadata": {"architecture": "no-ranking-policy"},
        "weights_after_fit_sha256": "none",
        "expected_weights_after_fit_sha256": streaming_expected,
        "fit_cpu_ns": 0,
        "fit_wall_ns": 0,
        "fit_performed": False,
    }
    return models, training, construction


def _work_vector(search: dict[str, Any],
                 validation: dict[str, Any], engine: str,
                 observed_rows: int) -> dict[str, int | None]:
    """Keep unlike work units as separate named coordinates."""
    work: dict[str, int | None] = {
        key: (int(search[key]) if key in search else None)
        for key in COUNT_KEYS
    }
    if "node_row_operations" in search:
        work["candidate_row_operations"] = int(
            search["node_row_operations"])
    if engine == "baseline":
        feature_builds = int(search.get("feature_builds", 0))
        analytical = 4 * observed_rows * feature_builds
        work["reference_feature_target_range_elements_scanned"] = analytical
        work["reference_feature_operand_elements_scanned"] = analytical
        work["reference_gap_elements_scanned"] = None
    work["validation_calls_post_search"] = int(
        validation.get("validation_calls", 0))
    work["validation_node_row_operations_post_search"] = int(
        validation.get("node_row_operations", 0))
    return work


def _warmup_controls(models: dict[str, Any], seed: int) -> dict[str, Any]:
    """Warm ranking controls once on fixed zeros before timed evaluation."""
    features = [[0.0] * 12 for _ in range(16)]
    reports: dict[str, Any] = {}
    for name in CONTROL_NAMES:
        if name == "streaming":
            continue
        policy = _control_for(name, models, seed, "total-work-warmup")
        before = v1._state_hash(policy)
        weights_before = v1._weights_hash(policy)
        order = policy.rank(features, None, 0.0)
        after = v1._state_hash(policy)
        weights_after = v1._weights_hash(policy)
        if (not isinstance(order, list)
                or set(order) != set(range(len(features)))):
            raise ValueError(f"{name} warmup did not return a permutation")
        if before != after or weights_before != weights_after:
            raise ValueError(f"{name} state changed during warmup")
        reports[name] = {
            "feature_rows": len(features),
            "feature_dimensions": 12,
            "rank_order_sha256": _json_hash(order),
            "state_fingerprint_before": before,
            "state_fingerprint_after": after,
            "weights_before_sha256": weights_before,
            "weights_after_sha256": weights_after,
            "state_unchanged": True,
        }
    return reports


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _attempt(
        engine: str, search_fn: SearchFn, policy: Any,
        task: dict[str, Any], budget: int, task_id: str,
        ) -> dict[str, Any]:
    """Run one attempt and time search-only plus full measured attempt."""
    attempt_cpu_start = time.process_time_ns()
    attempt_wall_start = time.perf_counter_ns()
    state_before = "none" if policy is None else v1._state_hash(policy)
    weights_before = "none" if policy is None else v1._weights_hash(policy)
    search_cpu_start = time.process_time_ns()
    search_wall_start = time.perf_counter_ns()
    error = ""
    try:
        result = v1._search_once(policy, task, budget, task_id, search_fn)
    except Exception as exc:
        result = {"solved": False, "graph": None, "verdict": "exception"}
        error = type(exc).__name__ + ": " + str(exc)
    search_cpu_ns = time.process_time_ns() - search_cpu_start
    search_wall_ns = time.perf_counter_ns() - search_wall_start
    validation = validate_graph_exact(result.get("graph"), task)
    accepted = bool(
        result.get("solved", False)
        and validation["train_valid"]
        and validation["held_out_valid"]
        and validation["exact_function_match"]
    )
    state_after = "none" if policy is None else v1._state_hash(policy)
    weights_after = "none" if policy is None else v1._weights_hash(policy)
    attempt_cpu_ns = time.process_time_ns() - attempt_cpu_start
    attempt_wall_ns = time.perf_counter_ns() - attempt_wall_start
    search_record = {key: value for key, value in result.items()
                     if key != "graph"}
    deterministic_keys = (
        "signature_evaluations", "tried", "generated_descriptors",
        "scored_descriptors", "feature_builds", "policy_batches",
        "policy_batch_ticks", "policy_row_steps", "snn_steps",
        "node_row_operations", "graph_validation_calls", "oracle_calls",
        "operand_signature_evaluations", "operand_vector_lookups",
        "retained_operand_vectors", "committed_nodes",
        "equivalent_alternatives_dropped",
        "equivalent_alternatives_retained", "budget_exhausted",
        "signature_budget_exhausted",
    )
    behavior = {
        "solved": bool(result.get("solved", False)),
        "accepted": accepted,
        "verdict": result.get("verdict", ""),
        "graph_sha256": _json_hash(result.get("graph")),
        "rejected_sha256": _json_hash(result.get("rejected", [])),
        "error": error,
        "deterministic_search_metrics": {
            key: result[key] for key in deterministic_keys
            if key in result
        },
        "validation_flags": {
            key: validation.get(key) for key in (
                "train_valid", "held_out_valid", "exact_function_match",
                "validation_calls", "node_row_operations", "error",
            )
        },
    }
    return {
        "engine": engine,
        "solved": bool(result.get("solved", False)),
        "accepted": accepted,
        "censored": bool(result.get("budget_exhausted", False)
                          or result.get("signature_budget_exhausted", False)),
        "error": error,
        "validation": validation,
        "search": search_record,
        "returned_graph": result.get("graph"),
        "behavior_fingerprint": _json_hash(behavior),
        "work": _work_vector(
            result, validation, engine, len(task["train"]),
        ),
        "root_calls": int(result.get(
            "signature_evaluations", result.get("tried", 0),
        )),
        "search_cpu_ns": search_cpu_ns,
        "search_wall_ns": search_wall_ns,
        "attempt_cpu_ns_including_hashes_and_validation": attempt_cpu_ns,
        "attempt_wall_ns_including_hashes_and_validation": attempt_wall_ns,
        "state_fingerprint_before_eval": state_before,
        "state_fingerprint_after_eval": state_after,
        "weights_before_eval_sha256": weights_before,
        "weights_after_eval_sha256": weights_after,
        "state_frozen_during_eval": (
            state_before == state_after and weights_before == weights_after),
    }


def _environment() -> dict[str, Any]:
    import spikeforge

    path = Path(spikeforge.__file__).resolve()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "spikeforge_path": str(path),
        "spikeforge_module_sha256": sha256_bytes(path.read_bytes()),
    }


def _source_hashes(baseline_path: Path) -> dict[str, str]:
    model_reference = ROOT / (
        "artifacts/benchmark-pre-eval-v2-20261018/training.json"
    )
    paths = set((ROOT / "rsi_programmer").rglob("*.py"))
    paths.update((ROOT / "scripts").glob("benchmark*.py"))
    paths.update((ROOT / "scripts").glob("generate_benchmark_pre_eval_v2.py"))
    paths.update({
        baseline_path,
        ROOT / "fixtures/benchmark_total_work_v1.json",
        ROOT / "fixtures/benchmark_pre_eval_corpus_v2.json",
        ROOT / "fixtures/benchmark_pre_eval_v2.json",
        ROOT / "scripts/generate_benchmark_pre_eval_v2.py",
        ROOT / "rsi_programmer/benchmark_v2_corpus.py",
        model_reference,
        ROOT / "requirements.txt",
        ROOT / "uv.lock",
    })
    spikeforge_root = Path(spikeforge.__file__).resolve().parent
    paths.update(spikeforge_root.rglob("*.py"))
    hashes = {
        (str(path.relative_to(ROOT)) if ROOT in path.parents else str(path)):
        sha256_bytes(path.read_bytes())
        for path in sorted(paths) if path.exists()
    }
    return {
        "source_files_sha256": hashes,
        "baseline_source_sha256": sha256_bytes(baseline_path.read_bytes()),
        "optimized_pre_eval_sha256": sha256_bytes(
            (ROOT / "rsi_programmer/pre_eval.py").read_bytes()),
        "total_work_runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "protocol_sha256": sha256_bytes(
            (ROOT / "fixtures/benchmark_total_work_v1.json").read_bytes()),
        "corpus_sha256": sha256_bytes((
            ROOT / "fixtures/benchmark_pre_eval_corpus_v2.json"
        ).read_bytes()),
        "model_reference_sha256": sha256_bytes(model_reference.read_bytes()),
    }


def _git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, check=True,
                              text=True, capture_output=True).stdout.strip()

    return {"revision": run("rev-parse", "HEAD"),
            "status_porcelain": run("status", "--porcelain")}


def _summarize(records: list[dict[str, Any]], protocol: dict[str, Any],
               training: list[dict[str, Any]]) -> dict[str, Any]:
    budgets = protocol["evaluation"]["candidate_scan_budgets"]
    targets = protocol["aggregation"]["matched_success_targets"]
    points = []
    for engine in ENGINE_NAMES:
        for control in CONTROL_NAMES:
            for budget in budgets:
                for repeat in (0, 1, None):
                    rows = [row for row in records
                            if row["engine"] == engine
                            and row["control"] == control
                            and row["budget"] == budget
                            and (repeat is None
                                 or row["repeat"] == repeat)]
                    if not rows:
                        continue
                    rows.sort(key=lambda item: (item["seed"],
                                                item["task_id"],
                                                item["repeat"]))
                    work_totals: dict[str, int | None] = {}
                    for key in rows[0]["work"]:
                        values = [row["work"][key] for row in rows
                                  if row["work"][key] is not None]
                        work_totals[key] = sum(values) if values else None
                    points.append({
                        "engine": engine, "control": control,
                        "budget": budget, "repeat": repeat,
                        "attempts": len(rows),
                        "accepted": sum(row["accepted"] for row in rows),
                        "success_fraction": (
                            sum(row["accepted"] for row in rows) / len(rows)),
                        "attempt_cpu_ns_mean": sum(
                            row[
                                "attempt_cpu_ns_including_hashes_"
                                "and_validation"
                            ]
                            for row in rows) / len(rows),
                        "attempt_wall_ns_mean": sum(
                            row[
                                "attempt_wall_ns_including_hashes_"
                                "and_validation"
                            ]
                            for row in rows) / len(rows),
                        "censored_attempts": sum(row["censored"]
                                                  for row in rows),
                        "work_totals_by_axis": work_totals,
                    })
    matched = []
    for engine in ENGINE_NAMES:
        for control in CONTROL_NAMES:
            for repeat in (0, 1, None):
                rows_by_budget = {
                    point["budget"]: point for point in points
                    if point["engine"] == engine
                    and point["control"] == control
                    and point["repeat"] == repeat
                }
                for target in targets:
                    eligible = [budget for budget in budgets
                                if budget in rows_by_budget
                                and rows_by_budget[budget][
                                    "success_fraction"] >= target]
                    budget = min(eligible) if eligible else None
                    point = rows_by_budget.get(budget) if budget else None
                    matched.append({
                        "engine": engine, "control": control,
                        "repeat": repeat, "target": target,
                        "status": "observed" if point else "censored",
                        "budget": budget,
                        "censored_at_budget": (
                            None if point else max(budgets)),
                        "success_fraction": (point["success_fraction"]
                                             if point else None),
                        "overshoot": (point["success_fraction"] - target
                                      if point else None),
                        "mean_attempt_cpu_ns": (
                            point["attempt_cpu_ns_mean"] if point else None),
                        "mean_attempt_wall_ns": (
                            point["attempt_wall_ns_mean"] if point else None),
                        "work_totals_by_axis": (
                            point["work_totals_by_axis"] if point else None),
                    })
    paired_cost_ratios = []
    for control in CONTROL_NAMES:
        for budget in budgets:
            for repeat in (0, 1, None):
                engine_sums = {}
                for engine in ENGINE_NAMES:
                    rows = [row for row in records
                            if row["engine"] == engine
                            and row["control"] == control
                            and row["budget"] == budget
                            and (repeat is None
                                 or row["repeat"] == repeat)]
                    engine_sums[engine] = {
                        "cpu_ns": sum(row[
                            "attempt_cpu_ns_including_hashes_and_validation"]
                            for row in rows),
                        "wall_ns": sum(row[
                            "attempt_wall_ns_including_hashes_and_validation"]
                            for row in rows),
                    }
                paired_cost_ratios.append({
                    "control": control, "budget": budget,
                    "repeat": repeat,
                    "baseline_cpu_ns": engine_sums["baseline"]["cpu_ns"],
                    "optimized_cpu_ns": engine_sums["optimized"]["cpu_ns"],
                    "optimized_over_baseline_cpu_ratio": (
                        engine_sums["optimized"]["cpu_ns"]
                        / engine_sums["baseline"]["cpu_ns"]
                        if engine_sums["baseline"]["cpu_ns"] else None),
                    "baseline_wall_ns": engine_sums["baseline"]["wall_ns"],
                    "optimized_wall_ns": engine_sums["optimized"]["wall_ns"],
                    "optimized_over_baseline_wall_ratio": (
                        engine_sums["optimized"]["wall_ns"]
                        / engine_sums["baseline"]["wall_ns"]
                        if engine_sums["baseline"]["wall_ns"] else None),
                })
    amortization = []
    train_by_seed = {record["seed"]: record for record in training}
    for engine in ENGINE_NAMES:
        for control in CONTROL_NAMES:
            for budget in budgets:
                point = [item for item in points if item["engine"] == engine
                         and item["control"] == control
                         and item["budget"] == budget
                         and item["repeat"] is None]
                mean_attempt = (
                    point[0]["attempt_cpu_ns_mean"] if point else None)
                mean_wall_attempt = (
                    point[0]["attempt_wall_ns_mean"] if point else None)
                training_cpu = 0
                training_wall = 0
                if control in {"trained_snn", "mlp", "recurrent"}:
                    fit_setup_cpu = []
                    fit_setup_wall = []
                    for seed_record in train_by_seed.values():
                        fit_setup_cpu.append(
                            seed_record["control_construction"][control][
                                "cpu_ns"]
                            + seed_record["controls"][control]["fit_cpu_ns"]
                        )
                        fit_setup_wall.append(
                            seed_record["control_construction"][control][
                                "wall_ns"]
                            + seed_record["controls"][control]["fit_wall_ns"]
                        )
                    if train_by_seed:
                        training_cpu = (
                            next(iter(train_by_seed.values()))[
                                "training_group_cpu_ns"]
                            + sum(fit_setup_cpu) / len(fit_setup_cpu)
                        )
                        training_wall = (
                            next(iter(train_by_seed.values()))[
                                "training_group_wall_ns"]
                            + sum(fit_setup_wall) / len(fit_setup_wall)
                        )
                for task_count in protocol["amortization"]["task_counts"]:
                    amortization.append({
                        "engine": engine, "control": control,
                        "budget": budget, "independent_task_count": task_count,
                        "training_cpu_ns_mean_seed_including_full_setup": (
                            training_cpu),
                        "training_wall_ns_mean_seed_including_full_setup": (
                            training_wall),
                        "training_cpu_ns_per_task": (
                            training_cpu / task_count),
                        "mean_attempt_cpu_ns": mean_attempt,
                        "mean_attempt_wall_ns": mean_wall_attempt,
                        "amortized_cpu_ns_per_task": (
                            None if mean_attempt is None else
                            training_cpu / task_count + mean_attempt),
                        "amortized_wall_ns_per_task": (
                            None if mean_wall_attempt is None else
                            training_wall / task_count + mean_wall_attempt),
                        "training_cost_included": training_cpu > 0,
                    })
    for item in matched:
        item["training_amortized_cpu_ns_per_task_by_N"] = {
            entry["independent_task_count"]:
                entry["amortized_cpu_ns_per_task"]
            for entry in amortization
            if item["status"] == "observed"
            and entry["engine"] == item["engine"]
            and entry["control"] == item["control"]
            and entry["budget"] == item["budget"]
        }
        item["training_amortized_wall_ns_per_task_by_N"] = {
            entry["independent_task_count"]:
                entry["amortized_wall_ns_per_task"]
            for entry in amortization
            if item["status"] == "observed"
            and entry["engine"] == item["engine"]
            and entry["control"] == item["control"]
            and entry["budget"] == item["budget"]
        }
    return {"points": points, "matched_success_cpu_cost": matched,
            "paired_cost_ratios": paired_cost_ratios,
            "amortization": amortization}


def run_total_work(
        output: Path,
        protocol: dict[str, Any],
        baseline_search: SearchFn | None = None,
        optimized_search: SearchFn = pre_evaluation_search,
        baseline_path: Path = BASELINE_PATH,
        ) -> dict[str, Any]:
    """Run the frozen paired matrix; leave all uncompleted attempts visible."""
    all_cpu_start = time.process_time_ns()
    all_wall_start = time.perf_counter_ns()
    v2_protocol, corpus = load_frozen_inputs()
    validate_total_work_protocol(protocol, v2_protocol, corpus)
    protocol_path = ROOT / "fixtures/benchmark_total_work_v1.json"
    protocol_bytes = protocol_path.read_bytes()
    if sha256_bytes(protocol_bytes) != TOTAL_PROTOCOL_SHA256:
        raise ValueError("total-work protocol bytes differ from frozen hash")
    if json.loads(protocol_bytes) != protocol:
        raise ValueError("provided protocol differs from frozen protocol file")
    expected_reference = protocol["comparison"]["model_reference"]["sha256"]
    reference_path = ROOT / protocol["comparison"]["model_reference"]["path"]
    if sha256_bytes(reference_path.read_bytes()) != expected_reference:
        raise ValueError("historical model reference hash mismatch")
    baseline_search = baseline_search or load_baseline_search(baseline_path)
    if sha256_bytes(baseline_path.read_bytes()) != BASELINE_SHA256:
        raise ValueError("baseline source changed before run")
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "raw.jsonl"
    raw_path.touch()
    torch.set_num_threads(1)
    before = {"environment": _environment(),
              "source_sha256": _source_hashes(baseline_path),
              "git": _git_state()}
    groups_cpu_start = time.process_time_ns()
    groups_wall_start = time.perf_counter_ns()
    groups, group_setup = build_training_groups(
        corpus["splits"]["train"],
        positive_cap=protocol["training"]["positive_cap"],
        negative_cap=protocol["training"]["negative_cap"],
        sampling_seed=protocol["training"]["sampling_seed"],
    )
    groups_cpu_ns = time.process_time_ns() - groups_cpu_start
    groups_wall_ns = time.perf_counter_ns() - groups_wall_start
    seeds: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    def save_training_checkpoint(status: str) -> None:
        _atomic_json(output / "training.json", {
            "status": status,
            "training_group_cpu_ns": groups_cpu_ns,
            "training_group_wall_ns": groups_wall_ns,
            "training_groups": group_setup,
            "seeds": seeds,
        })

    save_training_checkpoint("training_in_progress")
    expected_weights = protocol["comparison"]["model_reference"][
        "expected_weights_after_fit"]
    for seed in protocol["training"]["seeds"]:
        seed_expected = expected_weights.get(str(seed), {})
        models, control_training, construction = _make_models_with_cpu(
            seed, groups, protocol["training"]["epochs"], seed_expected,
        )
        warmup = _warmup_controls(models, seed)
        seeds.append({
            "seed": seed,
            "control_construction": construction,
            "training_group_cpu_ns": (groups_cpu_ns if not seeds else 0),
            "training_group_wall_ns": (groups_wall_ns if not seeds else 0),
            "controls": control_training,
            "warmup": warmup,
        })
        save_training_checkpoint("evaluation_in_progress")
        for task in corpus["splits"]["eval"]:
            for budget in protocol["evaluation"][
                    "candidate_scan_budgets"]:
                for control in CONTROL_NAMES:
                    for repeat in range(2):
                        order = engine_order(
                            seed, control, task["id"], budget, repeat,
                            protocol["evaluation"]["engine_order_seed"],
                        )
                        pair: list[dict[str, Any]] = []
                        for engine in order:
                            search_fn = (
                                baseline_search if engine == "baseline"
                                else optimized_search)
                            policy = _control_for(
                                control, models, seed, task["id"],
                            )
                            result = _attempt(
                                engine, search_fn, policy, task, budget,
                                task["id"],
                            )
                            row = {
                                "protocol_id": protocol["protocol_id"],
                                "corpus_sha256": protocol["corpus"]["sha256"],
                                "seed": seed, "control": control,
                                "task_id": task["id"],
                                "family": task["family"], "budget": budget,
                                "repeat": repeat, "engine": engine,
                                "engine_order": list(order),
                                "bound": protocol["evaluation"]["tree_bound"],
                                "expected_weights_after_fit_sha256": (
                                    control_training[control][
                                        "weights_after_fit_sha256"]),
                                "weights_match_after_fit_reference": (
                                    None if control in {"streaming",
                                                       "seeded_random"}
                                    else control_training[control][
                                        "weights_after_fit_sha256"]
                                        == result[
                                            "weights_before_eval_sha256"]),
                                **result,
                            }
                            pair.append(row)
                        baseline_behavior = next(
                            row["behavior_fingerprint"] for row in pair
                            if row["engine"] == "baseline"
                        )
                        for row in pair:
                            row["paired_behavior_matches_baseline"] = (
                                None if row["engine"] == "baseline" else
                                row["behavior_fingerprint"]
                                == baseline_behavior
                            )
                            with raw_path.open(
                                    "a", encoding="utf-8") as handle:
                                handle.write(json.dumps(
                                    row, sort_keys=True, allow_nan=False,
                                ) + "\n")
                                handle.flush()
                            records.append(row)
        final_states = {}
        for name, model in models.items():
            final_state = v1._state_hash(model)
            final_weights = v1._weights_hash(model)
            fit_record = control_training[name]
            final_states[name] = {
                "state_fingerprint_after_all_evaluation": final_state,
                "weights_after_all_evaluation_sha256": final_weights,
                "state_matches_after_fit": (
                    final_state == fit_record[
                        "state_fingerprint_after_fit"]),
                "weights_match_after_fit": (
                    final_weights == fit_record[
                        "weights_after_fit_sha256"]),
            }
        seeds[-1]["state_after_all_evaluation"] = final_states
        seeds[-1]["all_model_states_frozen"] = all(
            item["state_matches_after_fit"]
            and item["weights_match_after_fit"]
            for item in final_states.values()
        )
        save_training_checkpoint("seed_complete")
    after = {"environment": _environment(),
             "source_sha256": _source_hashes(baseline_path),
             "git": _git_state()}
    source_frozen = before["source_sha256"] == after["source_sha256"]
    expected_keys = {
        (seed, control, task["id"], budget, repeat, engine)
        for seed in protocol["training"]["seeds"]
        for control in CONTROL_NAMES
        for task in corpus["splits"]["eval"]
        for budget in protocol["evaluation"]["candidate_scan_budgets"]
        for repeat in range(2)
        for engine in ENGINE_NAMES
    }
    expected_count = len(expected_keys)
    grid_keys = [(row["seed"], row["control"], row["task_id"],
                  row["budget"], row["repeat"], row["engine"])
                 for row in records]
    grid_valid = (len(grid_keys) == expected_count
                  and set(grid_keys) == expected_keys)
    pair_valid = all(
        row["paired_behavior_matches_baseline"] is True
        for row in records if row["engine"] == "optimized"
    )
    repeat_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in records:
        key = (row["engine"], row["seed"], row["control"],
               row["task_id"], row["budget"])
        repeat_groups.setdefault(key, []).append(row)
    repeats_valid = all(
        len(rows) == 2
        and len({row["behavior_fingerprint"] for row in rows}) == 1
        for rows in repeat_groups.values()
    )
    model_states_frozen = all(
        row["state_frozen_during_eval"] for row in records
    ) and all(seed_record.get("all_model_states_frozen", False)
              for seed_record in seeds)
    fit_reference_valid = all(
        row["weights_match_after_fit_reference"] is not False
        for row in records
    )
    groups_valid = all(
        group.get("status") == "ok"
        for group in group_setup["groups"]
    )
    no_row_errors = not any(row["error"] for row in records)
    environment_frozen = before["environment"] == after["environment"]
    infrastructure_valid = (
        grid_valid and pair_valid and repeats_valid and model_states_frozen
        and fit_reference_valid and groups_valid and no_row_errors
        and source_frozen and environment_frozen
    )
    summary = _summarize(records, protocol, seeds)
    ratios = {
        item["control"]: item["optimized_over_baseline_cpu_ratio"]
        for item in summary["paired_cost_ratios"]
        if item["budget"] == protocol["decision_rule"]["primary_budget"]
        and item["repeat"] is None
    }
    required_controls = protocol["decision_rule"]["required_controls"]
    optimization_supported = infrastructure_valid and all(
        ratios.get(control) is not None
        and ratios[control] <= 1.0 - protocol["decision_rule"][
            "minimum_cpu_reduction"]
        for control in required_controls
    )
    summary["optimization_gate"] = {
        "budget": protocol["decision_rule"]["primary_budget"],
        "minimum_cpu_reduction": protocol["decision_rule"][
            "minimum_cpu_reduction"],
        "required_controls": required_controls,
        "paired_cpu_ratios": ratios,
        "supported": optimization_supported,
    }
    save_training_checkpoint("complete")
    _atomic_json(output / "aggregate.json", {
        **summary,
        "grid_valid": grid_valid,
        "paired_behavior_identical": pair_valid,
        "repeats_behavior_identical": repeats_valid,
        "model_states_frozen": model_states_frozen,
        "fit_reference_valid": fit_reference_valid,
        "training_groups_valid": groups_valid,
        "no_row_errors": no_row_errors,
        "environment_frozen": environment_frozen,
        "infrastructure_valid": infrastructure_valid,
    })
    total_cpu_ns = time.process_time_ns() - all_cpu_start
    total_wall_ns = time.perf_counter_ns() - all_wall_start
    manifest = {
        "protocol_sha256": before["source_sha256"]["protocol_sha256"],
        "corpus_sha256": before["source_sha256"]["corpus_sha256"],
        "baseline_revision": BASELINE_REVISION,
        "baseline_sha256": BASELINE_SHA256,
        "before": before, "after": after,
        "raw_jsonl_sha256": sha256_bytes(raw_path.read_bytes()),
        "training_json_sha256": sha256_bytes(
            (output / "training.json").read_bytes()),
        "aggregate_json_sha256": sha256_bytes(
            (output / "aggregate.json").read_bytes()),
        "row_count": len(records), "expected_row_count": expected_count,
        "grid_valid": grid_valid,
        "paired_behavior_identical": pair_valid,
        "repeats_behavior_identical": repeats_valid,
        "model_states_frozen": model_states_frozen,
        "fit_reference_valid": fit_reference_valid,
        "training_groups_valid": groups_valid,
        "no_row_errors": no_row_errors,
        "environment_frozen": environment_frozen,
        "source_frozen": source_frozen,
        "total_cpu_ns_through_raw_and_summary_output": total_cpu_ns,
        "total_wall_ns_through_raw_and_summary_output": total_wall_ns,
        "final_manifest_serialization_included": False,
        "infrastructure_valid": infrastructure_valid,
        "optimization_supported": optimization_supported,
    }
    _atomic_json(output / "manifest.json", manifest)
    return manifest
