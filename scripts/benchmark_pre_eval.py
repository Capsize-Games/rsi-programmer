#!/usr/bin/env python3
"""Run the frozen pre-evaluation search benchmark.

The protocol and corpus are immutable inputs. Results are written as raw
JSONL rows plus environment and per-seed training manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from rsi_programmer.benchmark_mlp import MLPControl
from rsi_programmer.benchmark_policy import Group, state_fingerprint
from rsi_programmer.benchmark_policy_heuristic import (
    ClassicalHeuristicPolicy,
)
from rsi_programmer.benchmark_policy_random import SeededRandomPolicy
from rsi_programmer.benchmark_recurrent import RecurrentControl
from rsi_programmer.benchmark_fit_work import fit_with_work
from rsi_programmer.policy import SearchPolicy, encode, layer_triples
from rsi_programmer.pre_eval import pre_evaluation_search
from rsi_programmer.probe import Expr, interpret, signature, validate_graph

PROTOCOL_PATH = ROOT / "fixtures/benchmark_pre_eval_v1.json"
CORPUS_PATH = ROOT / "fixtures/benchmark_pre_eval_corpus_v1.json"
GENERATOR_PATH = ROOT / "scripts/generate_benchmark_pre_eval_corpus.js"
PROTOCOL_SHA256 = (
    "75ef7e327b02d8e7665ddba10edcdecaaed034f403b55050e44d58f229d07e84"
)
CORPUS_SHA256 = (
    "81dba9b0d683e725e4598dd19c16da55c1b3d8b8c73000df2aba54df8df4b5c7"
)
CONTROL_NAMES = (
    "streaming", "heuristic", "seeded_random", "untrained_snn",
    "trained_snn", "mlp", "recurrent",
)


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase SHA256 digest."""
    return hashlib.sha256(data).hexdigest()


def frozen_inputs(protocol_data: bytes, corpus_data: bytes
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify pinned bytes before parsing benchmark configuration."""
    if sha256_bytes(protocol_data) != PROTOCOL_SHA256:
        raise ValueError("frozen benchmark protocol hash mismatch")
    if sha256_bytes(corpus_data) != CORPUS_SHA256:
        raise ValueError("frozen benchmark corpus hash mismatch")
    protocol = json.loads(protocol_data)
    corpus = json.loads(corpus_data)
    if protocol["corpus"]["sha256"] != CORPUS_SHA256:
        raise ValueError("protocol does not pin the frozen corpus")
    generator_data = GENERATOR_PATH.read_bytes()
    expected_generator = protocol["corpus"]["generator_source"]["sha256"]
    if sha256_bytes(generator_data) != expected_generator:
        raise ValueError("frozen corpus generator hash mismatch")
    validate_corpus(corpus, protocol)
    return protocol, corpus


def validate_corpus(corpus: dict[str, Any],
                    protocol: dict[str, Any]) -> None:
    """Reject split overlap, missing rows, or generator leakage."""
    splits = corpus.get("splits", {})
    train = splits.get("train", [])
    evaluation = splits.get("eval", [])
    expected = protocol["corpus"]["counts"]
    if len(train) != expected["train_tasks"]:
        raise ValueError("wrong number of training tasks")
    if len(evaluation) != expected["eval_tasks"]:
        raise ValueError("wrong number of evaluation tasks")
    ids: set[str] = set()
    programs: set[str] = set()
    row_pairs: set[tuple[int, int]] = set()
    family_counts: dict[str, dict[str, int]] = {
        "train": defaultdict(int), "eval": defaultdict(int),
    }
    for split_name, tasks in (("train", train), ("eval", evaluation)):
        for task in tasks:
            task_id = task.get("id")
            generator = task.get("generator")
            if not isinstance(task_id, str) or task_id in ids:
                raise ValueError("task IDs must be unique strings")
            ids.add(task_id)
            if not isinstance(generator, dict):
                raise ValueError("each task needs generation provenance")
            program_key = generator.get("program_key")
            if not isinstance(program_key, str) or program_key in programs:
                raise ValueError("generating structures must be unique")
            if program_key != _expression_key(generator.get("program")):
                raise ValueError("program key does not match witness")
            programs.add(program_key)
            family = task.get("family")
            if family not in protocol["corpus"]["families"]:
                raise ValueError("unknown structural family")
            expression = generator.get("program")
            if not _matches_family(expression, family):
                raise ValueError("expression does not match its family")
            family_counts[split_name][family] += 1
            if task.get("max_tree_ops") != protocol["evaluation"][
                    "tree_bound"]:
                raise ValueError("task bound differs from frozen protocol")
            for key, count_key in (("train", "train"),
                                   ("held_out", "held_out")):
                rows = task.get(key)
                want = protocol["corpus"]["rows_per_task"][count_key]
                if not isinstance(rows, list) or len(rows) != want:
                    raise ValueError("wrong row count in frozen task")
                for row in rows:
                    if (not isinstance(row, list) or len(row) != 3
                            or any(type(value) is not int for value in row)):
                        raise ValueError("rows must be integer triples")
                    pair = (row[0], row[1])
                    if pair in row_pairs:
                        raise ValueError("input coordinates overlap globally")
                    row_pairs.add(pair)
                    if (_evaluate_expression(expression, row[0], row[1])
                            != row[2]):
                        raise ValueError("frozen row does not match witness")
    wanted_families = set(protocol["corpus"]["families"])
    if set(family_counts["train"]) != wanted_families:
        raise ValueError("training split is missing a family")
    if set(family_counts["eval"]) != wanted_families:
        raise ValueError("evaluation split is missing a family")
    if any(count != expected["tasks_per_eval_family"]
           for count in family_counts["eval"].values()):
        raise ValueError("evaluation family counts differ from protocol")
    if any(count != expected["train_per_family"]
           for count in family_counts["train"].values()):
        raise ValueError("training family counts differ from protocol")
    if len(programs) != len(train) + len(evaluation):
        raise ValueError("program structure leakage across splits")


def _matches_family(node: dict[str, Any], family: str) -> bool:
    """Check the corpus syntax bucket without solver outcomes."""
    if family == "balanced":
        return (isinstance(node, dict)
                and isinstance(node.get("left"), dict)
                and isinstance(node.get("right"), dict)
                and node["left"].get("op") in {"add", "sub", "mul"}
                and node["right"].get("op") in {"add", "sub", "mul"}
                and node["left"].get("left", {}).get("op") == "input"
                and node["left"].get("right", {}).get("op") == "input"
                and node["right"].get("left", {}).get("op") == "input"
                and node["right"].get("right", {}).get("op") == "input")
    if family == "repeated_subtree":
        return (isinstance(node, dict)
                and node.get("left") == node.get("right")
                and isinstance(node.get("left"), dict)
                and node["left"].get("op") in {"add", "sub", "mul"})
    if family == "skewed":
        right = node.get("right") if isinstance(node, dict) else None
        return (isinstance(node, dict)
                and node.get("left", {}).get("op") == "input"
                and isinstance(right, dict)
                and right.get("left", {}).get("op") == "input"
                and isinstance(right.get("right"), dict)
                and right["right"].get("left", {}).get("op") == "input"
                and right["right"].get("right", {}).get("op") == "input")
    return False


def _evaluate_expression(node: dict[str, Any], x: int, y: int) -> int:
    """Interpret the frozen generator witness without executing source."""
    if not isinstance(node, dict):
        raise ValueError("malformed generator expression")
    if node.get("op") == "input":
        name = node.get("name")
        if name not in {"x", "y"}:
            raise ValueError("unknown generator input")
        return x if name == "x" else y
    operator = node.get("op")
    if operator not in {"add", "sub", "mul"}:
        raise ValueError("unknown generator operator")
    left = _evaluate_expression(node.get("left"), x, y)
    right = _evaluate_expression(node.get("right"), x, y)
    if operator == "add":
        return left + right
    if operator == "sub":
        return left - right
    return left * right


def _expression_key(node: dict[str, Any]) -> str:
    """Return the generator's canonical syntax key."""
    if not isinstance(node, dict):
        raise ValueError("malformed generator expression")
    if node.get("op") == "input":
        name = node.get("name")
        if name not in {"x", "y"}:
            raise ValueError("unknown generator input")
        return name
    operator = node.get("op")
    if operator not in {"add", "sub", "mul"}:
        raise ValueError("unknown generator operator")
    return (operator + "(" + _expression_key(node.get("left"))
            + "," + _expression_key(node.get("right")) + ")")


def training_groups(tasks: list[dict[str, Any]]) -> tuple[list[Group],
                                                          dict[str, Any]]:
    """Build labels from train rows only and account for setup work."""
    generated: list[tuple[list[Any], Any, int, int]] = []
    generation_started = time.perf_counter()
    counters = {
        "terminal_vector_builds": 0,
        "candidate_signature_evaluations": 0,
        "candidate_row_operations": 0,
        "operand_vector_lookups": 0,
    }
    for task in tasks:
        rows = tuple((row[0], row[1]) for row in task["train"])
        target = tuple(row[2] for row in task["train"])
        bound = int(task["max_tree_ops"])
        levels: dict[int, list[Expr]] = {
            0: [Expr("input", "x"), Expr("input", "y")],
        }
        vectors = {id(term): signature(term, rows)
                   for term in levels[0]}
        counters["terminal_vector_builds"] += len(vectors)
        counters["candidate_row_operations"] += len(vectors) * len(rows)
        solved_layer = 0
        for size in range(1, bound + 1):
            triples = layer_triples(levels, size)
            seen: set[tuple[str, tuple[int, ...]]] = set()
            current: list[Expr] = []
            for op, left, right in triples:
                counters["operand_vector_lookups"] += 2
                left_sig = vectors[id(left)]
                right_sig = vectors[id(right)]
                if op == "add":
                    sig = tuple(a + b for a, b in zip(left_sig,
                                                       right_sig))
                elif op == "sub":
                    sig = tuple(a - b for a, b in zip(left_sig,
                                                       right_sig))
                else:
                    sig = tuple(a * b for a, b in zip(left_sig,
                                                       right_sig))
                counters["candidate_signature_evaluations"] += 1
                counters["candidate_row_operations"] += len(rows)
                expr = Expr(op, left, right, size)
                key = (op, sig)
                if key in seen:
                    continue
                seen.add(key)
                current.append(expr)
                vectors[id(expr)] = sig
            levels[size] = current
            if any(vectors[id(expr)] == target for expr in current):
                solved_layer = size
                break
        if not solved_layer:
            raise ValueError("training task target is not reachable")
        generated.append((triples, vectors, solved_layer, bound))
    generation_seconds = time.perf_counter() - generation_started

    labeling_started = time.perf_counter()
    groups: list[Group] = []
    final_candidates = 0
    label_counters = {
        "feature_builds": 0,
        "feature_row_statistic_observations": 0,
        "target_range_row_reads": 0,
        "operand_vector_lookups": 0,
        "candidate_row_operations": 0,
        "positive_labels": 0,
        "negative_labels": 0,
    }
    for task, generated_task in zip(tasks, generated):
        rows = tuple((row[0], row[1]) for row in task["train"])
        target = tuple(row[2] for row in task["train"])
        bound = int(task["max_tree_ops"])
        triples, vectors, size, _max_ops = generated_task
        candidates = triples
        final_candidates += len(candidates)
        width = len(candidates)
        group: Group = []
        for index, (op, left, right) in enumerate(candidates):
            label_counters["operand_vector_lookups"] += 2
            left_signature = vectors[id(left)]
            right_signature = vectors[id(right)]
            feature = encode(
                op, left_signature, right_signature, target,
                left.size, right.size, size, bound, index + 1, width,
            )
            label_counters["feature_builds"] += 1
            label_counters[
                "feature_row_statistic_observations"
            ] += 4 * len(rows)
            label_counters["target_range_row_reads"] += 4 * len(rows)
            if op == "add":
                output = tuple(a + b for a, b in zip(left_signature,
                                                      right_signature))
            elif op == "sub":
                output = tuple(a - b for a, b in zip(left_signature,
                                                      right_signature))
            else:
                output = tuple(a * b for a, b in zip(left_signature,
                                                      right_signature))
            label = int(output == target)
            label_counters["candidate_row_operations"] += len(rows)
            label_counters["positive_labels"] += label
            label_counters["negative_labels"] += 1 - label
            group.append((feature, label))
        groups.append(group)
    labeling_seconds = time.perf_counter() - labeling_started
    return groups, {
        "train_tasks": len(tasks),
        "groups": len(groups),
        "labeled_candidates": final_candidates,
        "generation_counters": counters,
        "labeling_counters": label_counters,
        "generation_seconds": generation_seconds,
        "labeling_seconds": labeling_seconds,
        "total_seconds": generation_seconds + labeling_seconds,
    }


def _state_hash(policy: Any) -> str:
    """Hash model weights with the model's declared semantics."""
    metadata_fn = getattr(policy, "metadata", None)
    if callable(metadata_fn):
        metadata = metadata_fn()
    else:
        metadata = {
            "name": "spikeforge_search_policy",
            "architecture": "12->16_leaky->1",
            "trainable_parameters": sum(
                parameter.numel() for parameter in
                policy._net.parameters()
            ),
            "feature_dimension": 12,
            "simulation_steps": policy._steps,
            "state_semantics": "fresh_state_per_rank_drive",
            "seed": policy._seed,
            "feature_encoding": "layer-ordinal-v1",
        }
    return state_fingerprint(policy.weights(), metadata)


def _weights_hash(policy: Any) -> str:
    """Hash pure policy weight/configuration values without metadata."""
    return state_fingerprint(policy.weights(), {})


def _model_metadata(policy: Any) -> dict[str, Any]:
    """Return metadata for built-in controls and SearchPolicy."""
    metadata_fn = getattr(policy, "metadata", None)
    if callable(metadata_fn):
        return metadata_fn()
    return {
        "name": "spikeforge_search_policy",
        "architecture": "12->16_leaky->1",
        "trainable_parameters": sum(
            parameter.numel() for parameter in policy._net.parameters()
        ),
        "feature_dimension": 12,
        "simulation_steps": policy._steps,
        "state_semantics": "fresh_state_per_rank_drive",
        "fit_mode": "adam_margin_lr_0.05",
        "seed": policy._seed,
        "feature_encoding": "layer-ordinal-v1",
    }


def _fingerprint(policy: Any) -> str:
    """Compute a stable fingerprint for SearchPolicy or a control."""
    fingerprint = getattr(policy, "fingerprint", None)
    return (fingerprint() if callable(fingerprint) else
            _state_hash(policy))


def _grid_valid(expected: set[tuple[int, str, str, int]], row_count: int,
                completed: set[tuple[int, str, str, int]]) -> bool:
    """Require the complete unique seed/control/task/budget matrix."""
    return row_count == len(expected) and completed == expected


def _new_models(seed: int, groups: list[Group], epochs: int
                ) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Initialize controls, fit only declared learned controls, and report."""
    initialization_started = time.perf_counter()
    snn = SearchPolicy(seed=seed, hidden=16, steps=6, lr=0.05)
    untrained = SearchPolicy(seed=seed, hidden=16, steps=6, lr=0.05)
    heuristic = ClassicalHeuristicPolicy()
    mlp = MLPControl(seed=seed, lr=0.05)
    recurrent = RecurrentControl(seed=seed, lr=0.05)
    models: dict[str, Any] = {
        "heuristic": heuristic,
        "untrained_snn": untrained,
        "trained_snn": snn,
        "mlp": mlp,
        "recurrent": recurrent,
    }
    initialization_seconds = time.perf_counter() - initialization_started
    training: dict[str, Any] = {}
    for name, model in models.items():
        metadata = _model_metadata(model)
        expected_parameters = {
            "trained_snn": 225, "untrained_snn": 225,
            "mlp": 225, "recurrent": 497,
        }
        expected_steps = {
            "trained_snn": 6, "untrained_snn": 6,
            "mlp": 1, "recurrent": 6,
        }
        if name in expected_parameters and (
                metadata.get("trainable_parameters")
                != expected_parameters[name]
                or metadata.get("simulation_steps") != expected_steps[name]):
            raise ValueError(f"{name} architecture differs from protocol")
        eligible = [group for group in groups if
                    any(label == 1 for _, label in group)
                    and any(label == 0 for _, label in group)]
        positives = sum(sum(label == 1 for _, label in group)
                        for group in eligible)
        negatives = sum(sum(label == 0 for _, label in group)
                        for group in eligible)
        margin_pairs = sum(
            sum(label == 1 for _, label in group)
            * sum(label == 0 for _, label in group)
            for group in eligible
        )
        before_weights = _weights_hash(model)
        before_state = _state_hash(model)
        started = time.perf_counter()
        trainable = name in {"trained_snn", "mlp", "recurrent"}
        if trainable:
            loss, fit_work = fit_with_work(
                model, groups, epochs,
                int(metadata.get("simulation_steps", 1)),
            )
        else:
            loss = 0.0
            fit_work = {
                "eligible_groups": len(eligible),
                "positive_features": positives,
                "negative_features": negatives,
                "margin_pairs_per_epoch": margin_pairs,
                "optimizer_updates": 0, "forward_batches": 0,
                "scored_feature_rows": 0, "simulation_row_steps": 0,
            }
        duration = time.perf_counter() - started
        after_weights = _weights_hash(model)
        after_state = _state_hash(model)
        training[name] = {
            "metadata": metadata,
            "weights_before_fit_sha256": before_weights,
            "weights_after_fit_sha256": after_weights,
            "state_fingerprint_before_fit": before_state,
            "state_fingerprint_after_fit": after_state,
            "fingerprint_after_fit": _fingerprint(model),
            "fit_seconds": duration,
            "loss": float(loss),
            "requested_epochs": epochs,
            "fit_performed": trainable,
            "fit_changed_weights": before_weights != after_weights,
            "fit_changed_state": before_state != after_state,
            "fit_work": fit_work,
        }
    random_policy = SeededRandomPolicy(seed)
    training["seeded_random"] = {
        "metadata": {
            **random_policy.metadata(),
            "stream_seed_rule": (
                "SHA256 base seed and task ID; each raw row carries the "
                "actual stream fingerprint"
            ),
        },
        "weights_before_fit_sha256": _weights_hash(random_policy),
        "weights_after_fit_sha256": _weights_hash(random_policy),
        "state_fingerprint_before_fit": random_policy.fingerprint(),
        "state_fingerprint_after_fit": random_policy.fingerprint(),
        "fingerprint_after_fit": random_policy.fingerprint(),
        "fit_seconds": 0.0,
        "loss": None,
        "requested_epochs": epochs,
        "fit_performed": False,
        "fit_changed_weights": False,
        "fit_work": {
            "eligible_groups": 0, "positive_examples": 0,
            "negative_examples": 0,
            "pairwise_margin_comparisons_per_epoch": 0,
            "optimizer_updates": 0, "candidate_score_evaluations": 0,
            "snn_steps": 0,
        },
    }
    training["streaming"] = {
        "metadata": {
            "name": "streaming_probe_order",
            "architecture": "no-ranking-policy",
            "trainable_parameters": 0,
            "simulation_steps": 0,
            "state_semantics": "probe-order enumeration",
            "fit_mode": "none",
        },
        "weights_before_fit_sha256": "none",
        "weights_after_fit_sha256": "none",
        "state_fingerprint_before_fit": "none",
        "state_fingerprint_after_fit": "none",
        "fingerprint_after_fit": "none",
        "fit_seconds": 0.0,
        "loss": None,
        "requested_epochs": epochs,
        "fit_performed": False,
        "fit_changed_weights": False,
        "fit_work": {
            "eligible_groups": 0, "positive_examples": 0,
            "negative_examples": 0,
            "pairwise_margin_comparisons_per_epoch": 0,
            "optimizer_updates": 0, "candidate_score_evaluations": 0,
            "snn_steps": 0,
        },
    }
    return models, training, initialization_seconds


def _random_seed(seed: int, task_id: str) -> int:
    """Derive a stable per-task random stream from the declared seed."""
    value = f"{seed}:{task_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _held_out_valid(graph: Any, task: dict[str, Any]) -> tuple[bool, str]:
    """Validate a returned graph only after search has completed."""
    if graph is None:
        return False, "no-graph"
    try:
        validate_graph(graph)
        for x, y, expected in task["held_out"]:
            if interpret(graph, x, y) != expected:
                return False, "held-out-mismatch"
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        return False, "held-out-validator: " + str(exc)
    return True, ""


def _validate_returned_graph(graph: Any, task: dict[str, Any]
                             ) -> dict[str, Any]:
    """Check any returned graph against observed and held rows."""
    report: dict[str, Any] = {
        "train_valid": False,
        "held_out_valid": False,
        "validation_calls": 0,
        "node_row_operations": 0,
        "elapsed_seconds": 0.0,
        "error": "no-graph" if graph is None else "",
    }
    if graph is None:
        return report
    started = time.perf_counter()
    try:
        report["validation_calls"] += 1
        validate_graph(graph)
        nodes = len(graph["nodes"])
        train_valid = True
        for x, y, expected in task["train"]:
            report["validation_calls"] += 1
            report["node_row_operations"] += nodes
            if interpret(graph, x, y) != expected:
                train_valid = False
        held_valid = True
        for x, y, expected in task["held_out"]:
            report["validation_calls"] += 1
            report["node_row_operations"] += nodes
            if interpret(graph, x, y) != expected:
                held_valid = False
        report["train_valid"] = train_valid
        report["held_out_valid"] = held_valid
        if not train_valid:
            report["error"] = "training-row-mismatch"
        elif not held_valid:
            report["error"] = "held-out-mismatch"
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
    report["elapsed_seconds"] = time.perf_counter() - started
    return report


def _accepted(search_solved: bool,
              validation: dict[str, Any]) -> bool:
    """Require the search verdict and both independent row checks."""
    return (search_solved and validation.get("train_valid", False)
            and validation.get("held_out_valid", False))


def _search_once(policy: Any, task: dict[str, Any], budget: int,
                 task_id: str, search_fn: Callable[..., dict[str, Any]]
                 ) -> dict[str, Any]:
    """Search on the six observed examples; never pass held-out values."""
    rows = tuple((row[0], row[1]) for row in task["train"])
    expected = tuple(row[2] for row in task["train"])
    return search_fn(
        policy, rows, expected, int(task["max_tree_ops"]), budget,
        task_id=task_id, oracle=None, consts=(),
        alternatives_per_signature=1, signature_budget=None,
    )


def _source_manifest() -> dict[str, str]:
    """Hash runner, implementation, fixtures, and dependency inputs."""
    paths = set((ROOT / "rsi_programmer").rglob("*.py"))
    paths.update({
        ROOT / "scripts/benchmark_pre_eval.py",
        ROOT / "scripts/generate_benchmark_pre_eval_corpus.js",
        PROTOCOL_PATH,
        CORPUS_PATH,
        ROOT / "requirements.txt",
    })
    spikeforge = Path(__import__("spikeforge").__file__).resolve().parent
    paths.update(spikeforge.rglob("*.py"))
    uv_lock = ROOT / "uv.lock"
    if uv_lock.exists():
        paths.add(uv_lock)
    manifest: dict[str, str] = {}
    for path in sorted(paths):
        label = (str(path.relative_to(ROOT)) if ROOT in path.parents else
                 str(path))
        manifest[label] = sha256_bytes(path.read_bytes())
    return manifest


def _git_state() -> dict[str, Any]:
    """Return the checked-out revision and dirty status."""
    def command(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()

    return {
        "revision": command("rev-parse", "HEAD"),
        "status_porcelain": command("status", "--porcelain"),
    }


def _environment() -> dict[str, Any]:
    """Capture runtime and host details relevant to repeatability."""
    import spikeforge

    spikeforge_path = str(Path(spikeforge.__file__).resolve())
    spikeforge_hash = sha256_bytes(Path(spikeforge_path).read_bytes())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "spikeforge_path": spikeforge_path,
        "spikeforge_module_sha256": spikeforge_hash,
        "git": _git_state(),
    }


def _atomic_json(path: Path, value: Any) -> None:
    """Write a JSON artifact through a temporary sibling file."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def _run(output: Path,
        search_fn: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    """Execute the complete pinned seed/control/task/budget matrix."""
    protocol_data = PROTOCOL_PATH.read_bytes()
    corpus_data = CORPUS_PATH.read_bytes()
    protocol, corpus = frozen_inputs(protocol_data, corpus_data)
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw.jsonl").touch()
    before = {
        "environment": _environment(),
        "source_sha256": _source_manifest(),
    }
    train_tasks = corpus["splits"]["train"]
    eval_tasks = corpus["splits"]["eval"]
    budgets = protocol["evaluation"]["candidate_scan_budgets"]
    seeds = protocol["training"]["seeds"]
    epochs = protocol["training"]["epochs"]
    training_records: list[dict[str, Any]] = []
    summary_counts: dict[tuple[str, int, str, int], list[float]] = defaultdict(
        lambda: [0, 0, 0.0, 0.0, 0.0],
    )
    infrastructure_errors: list[dict[str, str]] = []
    completed_keys: set[tuple[int, str, str, int]] = set()
    row_count_seen = 0
    for seed in seeds:
        groups, group_cost = training_groups(train_tasks)
        models, fit_records, initialization_seconds = _new_models(
            seed, groups, epochs,
        )
        seed_training = {
            "seed": seed,
            "group_generation": group_cost,
            "model_initialization_seconds": initialization_seconds,
            "controls": fit_records,
        }
        training_records.append(seed_training)
        snapshots = {name: _state_hash(model)
                     for name, model in models.items()}
        with (output / "raw.jsonl").open("a", encoding="utf-8") as handle:
            for task in eval_tasks:
                for budget in budgets:
                    for control in CONTROL_NAMES:
                        if control == "streaming":
                            model = None
                        elif control == "seeded_random":
                            model = SeededRandomPolicy(
                                _random_seed(seed, task["id"]),
                            )
                        else:
                            model = models[control]
                        started = time.perf_counter()
                        error = ""
                        state_before = ("streaming-order-none"
                                        if model is None else
                                        _state_hash(model))
                        weights_before = ("none" if model is None else
                                          _weights_hash(model))
                        try:
                            result = _search_once(
                                model, task, budget, task["id"], search_fn,
                            )
                        except Exception as exc:  # preserve failed attempts
                            result = {"solved": False, "graph": None}
                            error = type(exc).__name__ + ": " + str(exc)
                        validation = _validate_returned_graph(
                            result.get("graph"), task,
                        )
                        train_valid = validation["train_valid"]
                        held_valid = validation["held_out_valid"]
                        after_state = (None if model is None else
                                       _state_hash(model))
                        after_weights = (None if model is None else
                                         _weights_hash(model))
                        state_frozen = (after_state == state_before
                                        if model is not None else True)
                        if error:
                            infrastructure_errors.append({
                                "seed": str(seed),
                                "control": control,
                                "task_id": task["id"],
                                "error": error,
                            })
                        if not state_frozen:
                            infrastructure_errors.append({
                                "seed": str(seed),
                                "control": control,
                                "task_id": task["id"],
                                "error": "policy state changed during eval",
                            })
                        accepted = _accepted(
                            bool(result.get("solved", False)), validation,
                        )
                        row = {
                            "protocol_id": protocol["protocol_id"],
                            "corpus_sha256": CORPUS_SHA256,
                            "seed": seed,
                            "control": control,
                            "task_id": task["id"],
                            "family": task["family"],
                            "budget": budget,
                            "bound": protocol["evaluation"]["tree_bound"],
                            "solved": bool(result.get("solved", False)),
                            "accepted": accepted,
                            "train_valid": train_valid,
                            "held_out_valid": held_valid,
                            "validation": validation,
                            "error": error,
                            "wall_seconds_including_wrapper": (
                                time.perf_counter() - started),
                            "search": {key: value for key, value in
                                       result.items() if key != "graph"},
                            "returned_graph": result.get("graph"),
                            "state_fingerprint_before_eval": state_before,
                            "state_fingerprint_after_eval": after_state,
                            "weights_before_eval_sha256": weights_before,
                            "weights_after_eval_sha256": after_weights,
                            "state_frozen_during_eval": state_frozen,
                        }
                        handle.write(json.dumps(
                            row, sort_keys=True, allow_nan=False,
                        ) + "\n")
                        handle.flush()
                        completed_keys.add((seed, control, task["id"],
                                            budget))
                        row_count_seen += 1
                        summary = summary_counts[(control, budget,
                                                 task["family"], seed)]
                        summary[0] += int(accepted)
                        summary[1] += 1
                        summary[2] += float(result.get("tried", 0))
                        summary[3] += float(
                            row["wall_seconds_including_wrapper"])
                        summary[4] += float(result.get("snn_steps", 0))
        for control, metadata in seed_training["controls"].items():
            if control in models:
                final_snapshot = _state_hash(models[control])
                final_weights = _weights_hash(models[control])
                metadata["state_fingerprint_after_eval"] = final_snapshot
                metadata["weights_after_eval_sha256"] = final_weights
                metadata["weights_frozen_during_eval"] = (
                    metadata["weights_after_fit_sha256"] == final_weights
                    and metadata["state_fingerprint_after_fit"]
                    == final_snapshot
                )
                if not metadata["weights_frozen_during_eval"]:
                    infrastructure_errors.append({
                        "seed": str(seed),
                        "control": control,
                        "task_id": "seed-summary",
                        "error": "model weights changed across evaluation",
                    })
            elif control == "seeded_random":
                metadata["weights_after_eval_sha256"] = (
                    "see-per-row-state-fingerprints"
                )
                metadata["state_fingerprint_after_eval"] = (
                    "see-per-row-state-fingerprints"
                )
                metadata["weights_frozen_during_eval"] = None
            else:
                metadata["weights_after_eval_sha256"] = "none"
                metadata["weights_frozen_during_eval"] = True
    after = {
        "environment": _environment(),
        "source_sha256": _source_manifest(),
    }
    source_frozen = before["source_sha256"] == after["source_sha256"]
    expected_keys = {
        (seed, control, task["id"], budget)
        for seed in seeds
        for control in CONTROL_NAMES
        for task in eval_tasks
        for budget in budgets
    }
    expected_rows = len(expected_keys)
    grid_valid = _grid_valid(expected_keys, row_count_seen, completed_keys)
    if not grid_valid:
        infrastructure_errors.append({
            "seed": "all", "control": "all", "task_id": "run",
            "error": "run grid is incomplete or contains duplicates",
        })
    if not source_frozen:
        infrastructure_errors.append({
            "seed": "all", "control": "all", "task_id": "run",
            "error": "source or frozen input bytes changed during run",
        })
    before_environment = dict(before["environment"])
    after_environment = dict(after["environment"])
    before_environment.pop("git", None)
    after_environment.pop("git", None)
    if before_environment != after_environment:
        infrastructure_errors.append({
            "seed": "all", "control": "all", "task_id": "run",
            "error": "runtime environment changed during run",
        })
    training_report = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "seeds": training_records,
    }
    family_rows = []
    family_fractions: dict[tuple[str, int, str], list[float]] = defaultdict(
        list,
    )
    overall_seed: dict[tuple[str, int, int], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0, 0.0],
    )
    for (control, budget, family, seed), values in sorted(
            summary_counts.items()):
        successes, count, scans, wall, snn_steps = values
        fraction = successes / count if count else 0.0
        family_fractions[(control, budget, family)].append(fraction)
        family_rows.append({
            "control": control, "budget": budget, "family": family,
            "seed": seed, "successes": int(successes),
            "tasks": int(count), "success_fraction": fraction,
            "candidate_signatures": int(scans),
            "wall_seconds": wall, "snn_steps": int(snn_steps),
        })
        total = overall_seed[(control, budget, seed)]
        total[0] += successes
        total[1] += count
        total[2] += scans
        total[3] += wall
        total[4] += snn_steps
    family_means = []
    for (control, budget, family), fractions in sorted(
            family_fractions.items()):
        family_means.append({
            "control": control, "budget": budget, "family": family,
            "seed_fractions": fractions,
            "mean_seed_fraction": sum(fractions) / len(fractions),
        })
    overall_rows = []
    overall_means: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (control, budget, seed), values in sorted(overall_seed.items()):
        successes, count, scans, wall, snn_steps = values
        fraction = successes / count if count else 0.0
        overall_means[(control, budget)].append(fraction)
        overall_rows.append({
            "control": control, "budget": budget, "seed": seed,
            "successes": int(successes), "tasks": int(count),
            "success_fraction": fraction,
            "candidate_signatures": int(scans),
            "wall_seconds": wall, "snn_steps": int(snn_steps),
        })
    overall_summary = []
    for (control, budget), fractions in sorted(overall_means.items()):
        overall_summary.append({
            "control": control, "budget": budget,
            "seed_fractions": fractions,
            "mean_seed_fraction": sum(fractions) / len(fractions),
        })
    matched = []
    targets = protocol["decision_rule"]["matched_success_targets"]
    for control in CONTROL_NAMES:
        scores = {item["budget"]: item["mean_seed_fraction"]
                  for item in overall_summary
                  if item["control"] == control}
        for target in targets:
            reached = [budget for budget in budgets
                       if scores.get(budget, 0.0) >= target]
            matched.append({
                "control": control, "target": target,
                "lowest_budget": min(reached) if reached else None,
                "unreached_censored_at": (None if reached else
                                          max(budgets)),
            })
    target_budgets = [512, 2048]
    advantage_checks = []
    for budget in target_budgets:
        scores = {item["control"]: item["mean_seed_fraction"]
                  for item in overall_summary
                  if item["budget"] == budget}
        family_scores = {
            (item["family"], item["control"]):
            item["mean_seed_fraction"] for item in family_means
            if item["budget"] == budget
        }
        stronger_families = 0
        for family in protocol["corpus"]["families"]:
            trained = family_scores.get((family, "trained_snn"), 0.0)
            other = max((value for (name, control), value
                         in family_scores.items()
                         if name == family and control != "trained_snn"),
                        default=0.0)
            stronger_families += int(trained > other)
        best_other = max((value for control, value in scores.items()
                          if control != "trained_snn"), default=0.0)
        advantage_checks.append({
            "budget": budget,
            "trained_snn_mean_fraction": scores.get("trained_snn", 0.0),
            "best_other_mean_fraction": best_other,
            "overall_margin": scores.get("trained_snn", 0.0)
            - best_other,
            "families_where_trained_snn_is_higher": stronger_families,
        })
    rule_advantage = all(
        item["overall_margin"] >= 0.10
        and item["families_where_trained_snn_is_higher"] >= 2
        for item in advantage_checks
    )
    infrastructure_valid = (not infrastructure_errors and source_frozen
                            and grid_valid)
    _atomic_json(output / "training.json", training_report)
    aggregate_report = {
        "family_seed_rows": family_rows,
        "family_means_across_seeds": family_means,
        "overall_seed_rows": overall_rows,
        "overall_means_across_seeds": overall_summary,
        "matched_success_budgets": matched,
        "trained_snn_advantage": {
            "supported_by_frozen_rule": (rule_advantage
                                          and infrastructure_valid),
            "rule_would_pass": rule_advantage,
            "infrastructure_valid": infrastructure_valid,
            "checks": advantage_checks,
            "scope": protocol["decision_rule"]["claim_scope"],
        },
    }
    _atomic_json(output / "aggregate.json", aggregate_report)
    manifest = {
        "protocol_sha256": PROTOCOL_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "before": before,
        "after": after,
        "raw_jsonl_sha256": sha256_bytes(
            (output / "raw.jsonl").read_bytes()),
        "training_json_sha256": sha256_bytes(
            (output / "training.json").read_bytes()),
        "aggregate_json_sha256": sha256_bytes(
            (output / "aggregate.json").read_bytes()),
        "row_count": row_count_seen,
        "expected_row_count": expected_rows,
        "grid_valid": grid_valid,
        "infrastructure_valid": infrastructure_valid,
        "source_frozen": source_frozen,
        "infrastructure_errors": infrastructure_errors,
    }
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Run a complete benchmark to a new output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/benchmark-pre-eval-v1",
        help="new directory for the raw run and manifests",
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output_preexisting = output.exists()
    try:
        manifest = _run(output, pre_evaluation_search)
    except Exception as exc:
        if output.is_dir() and not output_preexisting:
            failure = {
                "row_type": "infrastructure_failure",
                "error": type(exc).__name__ + ": " + str(exc),
            }
            raw = output / "raw.jsonl"
            if raw.exists():
                with raw.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, sort_keys=True) + "\n")
            manifest = {
                "infrastructure_valid": False,
                "run_error": failure["error"],
                "raw_jsonl_sha256": (sha256_bytes(raw.read_bytes())
                                     if raw.exists() else None),
            }
            _atomic_json(output / "manifest.json", manifest)
            print(json.dumps(manifest, sort_keys=True), file=sys.stderr)
            return 2
        parser.error(str(exc))
    print(json.dumps(manifest, sort_keys=True))
    return 0 if manifest["infrastructure_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
