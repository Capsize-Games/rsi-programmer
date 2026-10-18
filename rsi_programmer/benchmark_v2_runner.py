"""Frozen input validation and execution for benchmark v2."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

from scripts import benchmark_pre_eval as v1
from rsi_programmer.benchmark_v2_aggregate import aggregate_rows
from rsi_programmer.benchmark_v2_training import build_training_groups
from rsi_programmer.benchmark_policy_random import SeededRandomPolicy
from rsi_programmer.probe import interpret

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "fixtures/benchmark_pre_eval_v2.json"
CORPUS_PATH = ROOT / "fixtures/benchmark_pre_eval_corpus_v2.json"
GENERATOR_PATHS = (
    ROOT / "scripts/generate_benchmark_pre_eval_v2.py",
    ROOT / "rsi_programmer/benchmark_v2_corpus.py",
)
PROTOCOL_SHA256 = (
    "cf83dc077dc7960d4e576d898d842484655ac2c61d0e36128da6cda5ae8e7015"
)
CORPUS_SHA256 = (
    "a8fd6add90b34da732da8dc96f2c0fafcabc2f017a17cdef4b31303eec4ba921"
)
CONTROL_NAMES = (
    "streaming", "heuristic", "seeded_random", "untrained_snn",
    "trained_snn", "mlp", "recurrent",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_frozen_inputs(
        protocol_path: Path = PROTOCOL_PATH,
        corpus_path: Path = CORPUS_PATH,
        generator_paths: tuple[Path, ...] = GENERATOR_PATHS,
        protocol_hash: str = PROTOCOL_SHA256,
        corpus_hash: str = CORPUS_SHA256,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify frozen bytes and generator provenance before parsing."""
    protocol_bytes = protocol_path.read_bytes()
    corpus_bytes = corpus_path.read_bytes()
    if sha256_bytes(protocol_bytes) != protocol_hash:
        raise ValueError("frozen v2 protocol hash mismatch")
    if sha256_bytes(corpus_bytes) != corpus_hash:
        raise ValueError("frozen v2 corpus hash mismatch")
    protocol = json.loads(protocol_bytes)
    corpus = json.loads(corpus_bytes)
    expected_generators = protocol["corpus"]["generator_source"]
    for path in generator_paths:
        try:
            label = str(path.relative_to(ROOT))
        except ValueError:
            label = path.name
        if sha256_bytes(path.read_bytes()) != expected_generators[label]:
            raise ValueError(f"frozen generator hash mismatch: {label}")
    if protocol["corpus"]["sha256"] != corpus_hash:
        raise ValueError("protocol does not pin the v2 corpus")
    validate_protocol(protocol)
    validate_corpus(corpus, protocol)
    return protocol, corpus


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Reject drift from the owner-approved v2 dimensions and controls."""
    if protocol.get("protocol_id") != "benchmark-pre-eval-v2":
        raise ValueError("unexpected v2 protocol ID")
    if tuple(protocol.get("controls", ())) != CONTROL_NAMES:
        raise ValueError("control set or order differs from v2 contract")
    training = protocol["training"]
    if training["seeds"] != [0, 1, 2] or training["epochs"] != 40:
        raise ValueError("training seeds or epochs differ from v2 contract")
    if training["learning_rate"] != 0.05:
        raise ValueError("learning rate differs from v2 contract")
    if training["positive_cap"] != 16 or training["negative_cap"] != 256:
        raise ValueError("training group caps differ from v2 contract")
    if training["sampling_seed"] != 82078:
        raise ValueError("training sampling seed differs from v2 contract")
    evaluation = protocol["evaluation"]
    if evaluation["tree_bound"] != 5:
        raise ValueError("tree bound differs from v2 contract")
    if evaluation["candidate_scan_budgets"] != [512, 2048, 8192, 32768]:
        raise ValueError("candidate budgets differ from v2 contract")
    decision = protocol["decision_rule"]
    if decision["primary_budgets"] != [2048, 8192]:
        raise ValueError("primary budgets differ from v2 contract")
    if decision["minimum_margin"] != 0.10:
        raise ValueError("minimum margin differs from v2 contract")
    if decision["minimum_families_higher"] != 2:
        raise ValueError("family count differs from v2 contract")
    if decision["matched_success_targets"] != [0.25, 0.5, 0.75, 1.0]:
        raise ValueError("matched-success targets differ from v2 contract")
    uncertainty = protocol["uncertainty"]
    if (uncertainty["draws"] != 2000 or uncertainty["seed"] != 82079
            or uncertainty["confidence_level"] != 0.95):
        raise ValueError("paired-bootstrap settings differ from v2 contract")
    corpus = protocol["corpus"]
    counts = corpus["counts"]
    if (counts["train_tasks"] != 24 or counts["eval_tasks"] != 48
            or counts["train_per_family"] != 8
            or counts["tasks_per_eval_family"] != 16):
        raise ValueError("corpus task counts differ from v2 contract")
    if corpus["families"] != ["balanced", "repeated_subtree", "skewed"]:
        raise ValueError("corpus families differ from v2 contract")
    if corpus["rows_per_task"] != {"observed": 12, "held": 12}:
        raise ValueError("corpus row counts differ from v2 contract")
    if (protocol["feature_encoding"]["name"] != "layer-ordinal-v1"
            or protocol["feature_encoding"]["dimensions"] != 12):
        raise ValueError("feature encoding differs from v2 contract")


def validate_corpus(corpus: dict[str, Any],
                    protocol: dict[str, Any]) -> None:
    """Check v2 counts, syntax witnesses, exact functions, and split rows."""
    splits = corpus.get("splits", {})
    train = splits.get("train", [])
    evaluation = splits.get("eval", [])
    expected = protocol["corpus"]["counts"]
    if len(train) != expected["train_tasks"]:
        raise ValueError("wrong number of v2 training tasks")
    if len(evaluation) != expected["eval_tasks"]:
        raise ValueError("wrong number of v2 evaluation tasks")
    ids: set[str] = set()
    structures: set[str] = set()
    functions: set[tuple[tuple[int, int, int], ...]] = set()
    coordinates: set[tuple[int, int]] = set()
    counts: dict[str, dict[str, int]] = {
        "train": {family: 0 for family in protocol["corpus"]["families"]},
        "eval": {family: 0 for family in protocol["corpus"]["families"]},
    }
    for split, tasks in (("train", train), ("eval", evaluation)):
        for task in tasks:
            task_id = task.get("id")
            if not isinstance(task_id, str) or task_id in ids:
                raise ValueError("v2 task IDs must be unique strings")
            ids.add(task_id)
            generator = task.get("generator")
            if not isinstance(generator, dict):
                raise ValueError("v2 task lacks generator witness")
            program = generator.get("program")
            key = _expression_key(program)
            poly = _expression_polynomial(program)
            family = task.get("family")
            if family not in counts[split]:
                raise ValueError("unknown v2 structural family")
            counts[split][family] += 1
            if key != generator.get("program_key"):
                raise ValueError("v2 program key differs from syntax witness")
            if key in structures:
                raise ValueError("v2 generating structures are duplicated")
            structures.add(key)
            canonical = tuple(tuple(term) for term in
                              generator.get("canonical_polynomial", []))
            if poly != canonical:
                raise ValueError("v2 canonical polynomial is incorrect")
            if poly in functions:
                raise ValueError("v2 functions are not globally unique")
            functions.add(poly)
            if generator.get("tree_ops") != _tree_ops(program):
                raise ValueError("v2 witness operation count is incorrect")
            operation_count = _tree_ops(program)
            if operation_count not in {4, 5}:
                raise ValueError("v2 witness is outside the frozen size range")
            if not _matches_family(program, family):
                raise ValueError("v2 witness does not match its family")
            if not _meets_difficulty_floor(poly):
                raise ValueError("v2 witness fails the algebraic floor")
            rows_per_split = protocol["corpus"]["rows_per_task"]
            for key_name, count_name in (("train", "observed"),
                                          ("held_out", "held")):
                rows = task.get(key_name)
                if (not isinstance(rows, list)
                        or len(rows) != rows_per_split[count_name]):
                    raise ValueError("wrong v2 row count")
                coords: set[tuple[int, int]] = set()
                for row in rows:
                    if (not isinstance(row, list) or len(row) != 3
                            or any(type(value) is not int for value in row)):
                        raise ValueError("v2 rows must be integer triples")
                    pair = (row[0], row[1])
                    if pair in coords:
                        raise ValueError("duplicate inputs within v2 task")
                    coords.add(pair)
                    if pair in coordinates:
                        raise ValueError(
                            "v2 input coordinates overlap globally")
                    coordinates.add(pair)
                    if _evaluate_polynomial(poly, *pair) != row[2]:
                        raise ValueError("v2 row differs from exact witness")
            if {tuple(row[:2]) for row in task["train"]} & {
                    tuple(row[:2]) for row in task["held_out"]}:
                raise ValueError("observed and held-out rows overlap")
            if task.get("max_tree_ops") != protocol["evaluation"][
                    "tree_bound"]:
                raise ValueError("task tree bound differs from protocol")
    if any(count != expected["train_per_family"]
           for count in counts["train"].values()):
        raise ValueError("v2 training family counts differ from protocol")
    if any(count != expected["tasks_per_eval_family"]
           for count in counts["eval"].values()):
        raise ValueError("v2 evaluation family counts differ from protocol")


def _expression_key(node: Any) -> str:
    if not isinstance(node, dict):
        raise ValueError("malformed v2 expression")
    if node.get("op") == "input":
        if node.get("name") not in {"x", "y"}:
            raise ValueError("unknown v2 input")
        return str(node["name"])
    op = node.get("op")
    if op not in {"add", "sub", "mul"}:
        raise ValueError("unknown v2 expression operator")
    return f"{op}({_expression_key(node.get('left'))}," \
        f"{_expression_key(node.get('right'))})"


Polynomial = dict[tuple[int, int], int]


def _expression_polynomial(node: Any) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(node, dict):
        raise ValueError("malformed v2 expression")
    if node.get("op") == "input":
        name = node.get("name")
        if name not in {"x", "y"}:
            raise ValueError("unknown v2 input")
        return (((1, 0, 1),) if name == "x" else ((0, 1, 1),))
    op = node.get("op")
    left = dict(((x, y), coefficient) for x, y, coefficient in
                _expression_polynomial(node.get("left")))
    right = dict(((x, y), coefficient) for x, y, coefficient in
                 _expression_polynomial(node.get("right")))
    result: Polynomial = dict(left)
    if op in {"add", "sub"}:
        factor = 1 if op == "add" else -1
        for monomial, coefficient in right.items():
            result[monomial] = result.get(monomial, 0) + factor * coefficient
    elif op == "mul":
        result = {}
        for (lx, ly), lv in left.items():
            for (rx, ry), rv in right.items():
                key = (lx + rx, ly + ry)
                result[key] = result.get(key, 0) + lv * rv
    else:
        raise ValueError("unknown v2 expression operator")
    return tuple((x, y, value) for (x, y), value in
                 sorted(result.items()) if value)


def _evaluate_polynomial(poly: tuple[tuple[int, int, int], ...],
                         x: int, y: int) -> int:
    return sum(coefficient * x ** xp * y ** yp
               for xp, yp, coefficient in poly)


def _tree_ops(node: dict[str, Any]) -> int:
    if node.get("op") == "input":
        return 0
    return 1 + _tree_ops(node["left"]) + _tree_ops(node["right"])


def _matches_family(node: dict[str, Any], family: str) -> bool:
    if node.get("op") == "input":
        return False
    left = node["left"]
    right = node["right"]
    if family == "balanced":
        return _tree_ops(left) == _tree_ops(right)
    if family == "repeated_subtree":
        seen: set[str] = set()

        def repeated(current: dict[str, Any]) -> bool:
            if current.get("op") == "input":
                return False
            key = _expression_key(current)
            if key in seen:
                return True
            seen.add(key)
            return repeated(current["left"]) or repeated(current["right"])

        return repeated(node)
    if family == "skewed":
        return abs(_tree_ops(left) - _tree_ops(right)) >= 2
    return False


def _meets_difficulty_floor(
        poly: tuple[tuple[int, int, int], ...]) -> bool:
    degree = max((x + y for x, y, _ in poly), default=-1)
    norm = sum(abs(coefficient) for _, _, coefficient in poly)
    return (degree >= 2 and (degree >= 5
            or (degree == 4 and norm > 1)
            or (degree == 3 and norm > 2)
            or (degree == 2 and norm > 4)))


def graph_polynomial(graph: dict[str, Any]
                     ) -> tuple[tuple[int, int, int], ...]:
    """Expand a validated returned DAG into an exact integer polynomial."""
    v1.validate_graph(graph)
    values: list[tuple[tuple[int, int, int], ...]] = []

    def resolve(ref: int | str) -> tuple[tuple[int, int, int], ...]:
        if isinstance(ref, str):
            return ((1, 0, 1),) if ref == "x" else ((0, 1, 1),)
        return values[ref]

    for node in graph["nodes"]:
        if node["op"] == "const":
            value = int(node["value"])
            poly = () if value == 0 else ((0, 0, value),)
        else:
            left = dict(((x, y), c) for x, y, c in
                        resolve(node["args"][0]))
            right = dict(((x, y), c) for x, y, c in
                         resolve(node["args"][1]))
            op = node["op"]
            result = dict(left)
            if op in {"add", "sub"}:
                factor = 1 if op == "add" else -1
                for monomial, coefficient in right.items():
                    result[monomial] = (result.get(monomial, 0)
                                        + factor * coefficient)
            elif op == "mul":
                result = {}
                for (lx, ly), lv in left.items():
                    for (rx, ry), rv in right.items():
                        key = (lx + rx, ly + ry)
                        result[key] = result.get(key, 0) + lv * rv
            else:
                raise ValueError("returned DAG is outside polynomial DSL")
            poly = tuple((x, y, c) for (x, y), c in
                         sorted(result.items()) if c)
        values.append(poly)
    return resolve(graph["output"])


def validate_graph_exact(graph: Any, task: dict[str, Any]
                         ) -> dict[str, Any]:
    """Check graph structure, observed/held rows, and exact function match."""
    report = {
        "train_valid": False, "held_out_valid": False,
        "exact_function_match": False, "validation_calls": 0,
        "node_row_operations": 0, "elapsed_seconds": 0.0, "error": "",
    }
    if graph is None:
        report["error"] = "no-graph"
        return report
    started = time.perf_counter()
    try:
        report["validation_calls"] += 1
        expected_poly = _expression_polynomial(
            task["generator"]["program"],
        )
        actual_poly = graph_polynomial(graph)
        report["exact_function_match"] = actual_poly == expected_poly
        node_count = len(graph["nodes"])
        for split in ("train", "held_out"):
            valid = True
            for x, y, expected in task[split]:
                report["validation_calls"] += 1
                report["node_row_operations"] += node_count
                if interpret(graph, x, y) != expected:
                    valid = False
            report["train_valid" if split == "train" else
                   "held_out_valid"] = valid
        if not report["train_valid"]:
            report["error"] = "observed-row-mismatch"
        elif not report["held_out_valid"]:
            report["error"] = "held-out-row-mismatch"
        elif not report["exact_function_match"]:
            report["error"] = "exact-polynomial-mismatch"
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
    report["elapsed_seconds"] = time.perf_counter() - started
    return report


def _random_seed(seed: int, task_id: str) -> int:
    payload = f"{seed}:{task_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _grid_valid(expected: set[tuple[int, str, str, int]],
                completed: list[tuple[int, str, str, int]]) -> bool:
    return len(completed) == len(expected) and set(completed) == expected


def _source_manifest(protocol_path: Path = PROTOCOL_PATH,
                     corpus_path: Path = CORPUS_PATH) -> dict[str, str]:
    paths = set((ROOT / "rsi_programmer").rglob("*.py"))
    paths.update((ROOT / "scripts").glob("benchmark_pre_eval*.py"))
    paths.update(GENERATOR_PATHS)
    paths.update({protocol_path, corpus_path, ROOT / "requirements.txt"})
    lock = ROOT / "uv.lock"
    if lock.exists():
        paths.add(lock)
    import spikeforge

    spikeforge_root = Path(spikeforge.__file__).resolve().parent
    paths.update(spikeforge_root.rglob("*.py"))
    return {
        (str(path.relative_to(ROOT)) if ROOT in path.parents else str(path)):
        sha256_bytes(path.read_bytes())
        for path in sorted(paths)
    }


def _environment() -> dict[str, Any]:
    import spikeforge

    path = Path(spikeforge.__file__).resolve()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "spikeforge_path": str(path),
        "spikeforge_module_sha256": sha256_bytes(path.read_bytes()),
        "git": _git_state(),
    }


def _git_state() -> dict[str, Any]:
    def command(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                                capture_output=True, text=True)
        return result.stdout.strip()

    return {"revision": command("rev-parse", "HEAD"),
            "status_porcelain": command("status", "--porcelain")}


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def run_benchmark(
        output: Path,
        search_fn: Callable[..., dict[str, Any]],
        protocol: dict[str, Any] | None = None,
        corpus: dict[str, Any] | None = None,
        source_manifest_fn: Callable[[], dict[str, str]] | None = None,
        environment_fn: Callable[[], dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
    """Execute the frozen matrix, retaining each attempted cell as JSONL."""
    if protocol is None or corpus is None:
        protocol, corpus = load_frozen_inputs()
    validate_protocol(protocol)
    validate_corpus(corpus, protocol)
    source_fn = source_manifest_fn or _source_manifest
    env_fn = environment_fn or _environment
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "raw.jsonl"
    raw_path.touch()
    before = {"environment": env_fn(), "source_sha256": source_fn()}
    groups, setup = build_training_groups(
        corpus["splits"]["train"],
        positive_cap=protocol["training"]["positive_cap"],
        negative_cap=protocol["training"]["negative_cap"],
        sampling_seed=protocol["training"]["sampling_seed"],
    )
    training_records = []
    errors: list[dict[str, str]] = [
        {"seed": "all", "control": "training", "task_id": group["task_id"],
         "error": group["error"]}
        for group in setup["groups"] if group["status"] != "ok"
    ]

    def save_training_checkpoint(status: str) -> None:
        _atomic_json(output / "training.json", {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": PROTOCOL_SHA256,
            "corpus_sha256": CORPUS_SHA256,
            "training_groups": setup,
            "status": status,
            "seeds": training_records,
        })

    save_training_checkpoint("training_and_evaluation_in_progress")
    records: list[dict[str, Any]] = []
    for seed in protocol["training"]["seeds"]:
        models, controls, init_seconds = v1._new_models(
            seed, groups, protocol["training"]["epochs"],
        )
        training_records.append({
            "seed": seed, "model_initialization_seconds": init_seconds,
            "controls": controls,
        })
        save_training_checkpoint("evaluation_in_progress")
        with raw_path.open("a", encoding="utf-8") as handle:
            for task in corpus["splits"]["eval"]:
                for budget in protocol["evaluation"][
                        "candidate_scan_budgets"]:
                    for control in CONTROL_NAMES:
                        model = _control_for(control, models, seed, task["id"])
                        state_before = ("streaming-order-none" if model is None
                                       else v1._state_hash(model))
                        weights_before = ("none" if model is None else
                                          v1._weights_hash(model))
                        started = time.perf_counter()
                        error = ""
                        try:
                            result = v1._search_once(
                                model, task, budget, task["id"], search_fn,
                            )
                        except Exception as exc:
                            result = {"solved": False, "graph": None}
                            error = type(exc).__name__ + ": " + str(exc)
                            errors.append({
                                "seed": str(seed), "control": control,
                                "task_id": task["id"], "error": error,
                            })
                        validation = validate_graph_exact(
                            result.get("graph"), task,
                        )
                        accepted = bool(
                            result.get("solved", False)
                            and validation["train_valid"]
                            and validation["held_out_valid"]
                            and validation["exact_function_match"]
                        )
                        state_after = (None if model is None else
                                       v1._state_hash(model))
                        weights_after = (None if model is None else
                                         v1._weights_hash(model))
                        frozen = (state_after == state_before
                                  if model is not None else True)
                        if not frozen:
                            errors.append({
                                "seed": str(seed), "control": control,
                                "task_id": task["id"],
                                "error": "policy state changed during eval",
                            })
                        row = {
                            "protocol_id": protocol["protocol_id"],
                            "corpus_sha256": CORPUS_SHA256,
                            "seed": seed, "control": control,
                            "task_id": task["id"],
                            "family": task["family"], "budget": budget,
                            "bound": protocol["evaluation"]["tree_bound"],
                            "solved": bool(result.get("solved", False)),
                            "accepted": accepted,
                            "validation": validation,
                            "error": error,
                            "wall_seconds_including_wrapper": (
                                time.perf_counter() - started),
                            "search": {key: value for key, value in
                                       result.items() if key != "graph"},
                            "returned_graph": result.get("graph"),
                            "state_fingerprint_before_eval": state_before,
                            "state_fingerprint_after_eval": state_after,
                            "weights_before_eval_sha256": weights_before,
                            "weights_after_eval_sha256": weights_after,
                            "state_frozen_during_eval": frozen,
                        }
                        handle.write(json.dumps(
                            row, sort_keys=True, allow_nan=False,
                        ) + "\n")
                        handle.flush()
                        records.append(row)
        for name, model in models.items():
            metadata = training_records[-1]["controls"][name]
            state_after_eval = v1._state_hash(model)
            weights_after_eval = v1._weights_hash(model)
            metadata["state_fingerprint_after_eval"] = state_after_eval
            metadata["weights_after_eval_sha256"] = weights_after_eval
            metadata["weights_frozen_during_eval"] = (
                state_after_eval == metadata[
                    "state_fingerprint_after_fit"]
                and weights_after_eval == metadata[
                    "weights_after_fit_sha256"]
            )
            if not metadata["weights_frozen_during_eval"]:
                errors.append({
                    "seed": str(seed), "control": name,
                    "task_id": "seed-summary",
                    "error": "model weights changed across evaluation",
                })
        save_training_checkpoint("seed_complete")
    after = {"environment": env_fn(), "source_sha256": source_fn()}
    source_frozen = before["source_sha256"] == after["source_sha256"]
    if not source_frozen:
        errors.append({"seed": "all", "control": "all",
                       "task_id": "run",
                       "error": "source or frozen input bytes changed"})
    expected_keys = {
        (seed, control, task["id"], budget)
        for seed in protocol["training"]["seeds"]
        for control in CONTROL_NAMES
        for task in corpus["splits"]["eval"]
        for budget in protocol["evaluation"]["candidate_scan_budgets"]
    }
    completed = [(row["seed"], row["control"], row["task_id"],
                  row["budget"]) for row in records]
    grid_valid = _grid_valid(expected_keys, completed)
    if not grid_valid:
        errors.append({"seed": "all", "control": "all",
                       "task_id": "run",
                       "error": "run grid is incomplete or duplicated"})
    environment_before = dict(before["environment"])
    environment_after = dict(after["environment"])
    environment_before.pop("git", None)
    environment_after.pop("git", None)
    if environment_before != environment_after:
        errors.append({"seed": "all", "control": "all",
                       "task_id": "run", "error":
                       "runtime environment changed during run"})
    infrastructure_valid = not errors and grid_valid and source_frozen
    training_report = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "training_groups": setup,
        "status": "complete",
        "seeds": training_records,
    }
    aggregate = aggregate_rows(records, protocol)
    aggregate["infrastructure_valid"] = infrastructure_valid
    aggregate["trained_snn_advantage"]["supported_by_frozen_rule"] = (
        aggregate["trained_snn_advantage"]["rule_would_pass"]
        and infrastructure_valid
    )
    _atomic_json(output / "training.json", training_report)
    _atomic_json(output / "aggregate.json", aggregate)
    manifest = {
        "protocol_sha256": PROTOCOL_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "before": before, "after": after,
        "raw_jsonl_sha256": sha256_bytes(raw_path.read_bytes()),
        "training_json_sha256": sha256_bytes(
            (output / "training.json").read_bytes()),
        "aggregate_json_sha256": sha256_bytes(
            (output / "aggregate.json").read_bytes()),
        "row_count": len(records),
        "expected_row_count": len(expected_keys),
        "grid_valid": grid_valid,
        "source_frozen": source_frozen,
        "infrastructure_valid": infrastructure_valid,
        "infrastructure_errors": errors,
    }
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def _control_for(name: str, models: dict[str, Any], seed: int,
                 task_id: str) -> Any:
    if name == "streaming":
        return None
    if name == "seeded_random":
        return SeededRandomPolicy(_random_seed(seed, task_id))
    return models[name]
