#!/usr/bin/env python3
"""Measure ranker positions independently of production layer builders.

Supports the plain add/sub/mul fixtures only. Random baselines use the
exact expected earliest solver position, not sampled permutations.
Outputs retain every task/seed/checkpoint, fixture hashes, and source hash.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from rsi_programmer import probe, search
from rsi_programmer.policy import FEATURE_ENCODING, SearchPolicy, encode

OPS = ("add", "sub", "mul")
Vector = tuple[int, ...]
Member = tuple[Vector, Vector, int]


def _source_hash() -> str:
    digest = hashlib.sha256()
    paths = sorted((ROOT / "rsi_programmer").glob("*.py"))
    paths.append(Path(__file__).resolve())
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _calculate(op: str, left: Vector, right: Vector) -> Vector:
    if op == "add":
        return tuple(a + b for a, b in zip(left, right))
    if op == "sub":
        return tuple(a - b for a, b in zip(left, right))
    return tuple(a * b for a, b in zip(left, right))


def _build(task: dict[str, Any]) -> dict[str, Any]:
    """Compose train/held vectors without any production enumeration."""
    if task.get("features"):
        raise ValueError("measurement supports plain arithmetic fixtures")
    rows, held = task["train"], task["held_out"]
    target = tuple(row[2] for row in rows)
    held_target = tuple(row[2] for row in held)
    levels: dict[int, list[Member]] = {
        0: [(tuple(row[col] for row in rows),
             tuple(row[col] for row in held), 0) for col in (0, 1)]}
    offset = 0
    for size in range(1, task["max_tree_ops"] + 1):
        entries: list[Any] = []
        seen: set[Any] = set()
        levels[size] = []
        for left_size in range(size):
            for left in levels[left_size]:
                for right in levels[size - left_size - 1]:
                    for op in OPS:
                        train = _calculate(op, left[0], right[0])
                        hold = _calculate(op, left[1], right[1])
                        entries.append((op, left, right, train, hold))
                        if (op, train) not in seen:
                            seen.add((op, train))
                            levels[size].append((train, hold, size))
        solvers = {i for i, entry in enumerate(entries)
                   if entry[3] == target}
        if solvers:
            both = {i for i in solvers
                    if entries[i][4] == held_target}
            return {"task": task["id"], "size": size,
                    "offset": offset, "entries": entries,
                    "solvers": solvers, "held_solvers": both,
                    "target": target, "bound": task["max_tree_ops"]}
        offset += len(entries)
    raise ValueError("no training solution: " + task["id"])


def _features(data: dict[str, Any]) -> list[list[float]]:
    """Independently encode per-layer candidate ordinals."""
    target = data["target"]
    count, low, high = len(target), min(target), max(target)
    width, bound = len(data["entries"]), data["bound"]
    out = []
    for i, (op, left, right, _, _) in enumerate(data["entries"]):
        row = [float(op == name) for name in OPS]
        row += [left[2] / bound, right[2] / bound, data["size"] / bound]
        for vector in (left[0], right[0]):
            row += [sum(a == b for a, b in zip(vector, target)) / count,
                    sum(low <= a <= high for a in vector) / count]
        row += [(i + 1) / width, 1 - (i + 1) / width]
        check = encode(op, left[0], right[0], target, left[2], right[2],
                       data["size"], bound, i + 1, width)
        if row != check:
            raise AssertionError("feature encoding differs")
        out.append(row)
    return out


def _position(scores: list[float], solvers: set[int]) -> int | None:
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    return next((k for k, i in enumerate(order, 1) if i in solvers), None)


def _streaming_work(task: dict[str, Any]) -> int:
    """Independent streaming comparator with the same lower-layer dedup."""
    rows = tuple(tuple(row[:2]) for row in task["train"])
    target = tuple(row[2] for row in task["train"])
    levels = {0: list(probe.terminals())}
    work = 0
    for size in range(1, task["max_tree_ops"] + 1):
        seen: set[Any] = set()
        levels[size] = []
        for left_size in range(size):
            for left in levels[left_size]:
                for right in levels[size - left_size - 1]:
                    for op in OPS:
                        expr = probe.Expr(op, left, right, size)
                        vector = probe.signature(expr, rows)
                        work += 1
                        if vector == target:
                            return work
                        if (op, vector) not in seen:
                            seen.add((op, vector))
                            levels[size].append(expr)
    return work


def _instrument(task: dict[str, Any],
                trained: SearchPolicy) -> dict[str, Any]:
    """Count production signature calls directly, excluding neural cost."""
    original = search.signature
    rows = tuple(tuple(row[:2]) for row in task["train"])
    target = tuple(row[2] for row in task["train"])

    class FixedOrder:
        def __init__(self, shuffled: bool = False) -> None:
            self.shuffled = shuffled

        def rank(self, features: list[list[float]],
                 *args: Any) -> list[int]:
            order = list(range(len(features)))
            if self.shuffled:
                random.Random(0).shuffle(order)
            return order

    runs = {}
    for name, policy in (("probe_order", FixedOrder()),
                         ("uniform_permutation_seed0", FixedOrder(True)),
                         ("untrained_seed0", SearchPolicy(seed=0)),
                         ("trained_last_seed_last_epoch", trained)):
        counts: Counter[str] = Counter()

        def tracked(expr: probe.Expr, inputs: Any) -> Vector:
            caller = sys._getframe(1).f_code.co_name
            kind = "candidate" if caller == "_intern" else "operand"
            counts[kind] += 1
            return original(expr, inputs)

        with patch.object(search, "signature", new=tracked):
            result = search.guided_search(
                policy, rows, target, task["max_tree_ops"], 10 ** 9)
        if counts["candidate"] != result["signature_evaluations"]:
            raise AssertionError("reported work differs from instrumentation")
        if counts["operand"] != result["operand_signature_evaluations"]:
            raise AssertionError("instrumented operand counter differs")
        runs[name] = {"candidate_signatures": counts["candidate"],
                      "operand_signatures": counts["operand"],
                      "examined": result["examined"]}
    if len({run["candidate_signatures"] for run in runs.values()}) != 1:
        raise AssertionError("eager signature work changed with ranker")
    baseline_calls = 0
    baseline_original = probe.signature

    def baseline_signature(expr: probe.Expr, inputs: Any) -> Vector:
        nonlocal baseline_calls
        if sys._getframe(1).f_code.co_name == "enumerate_programs":
            baseline_calls += 1
        return baseline_original(expr, inputs)

    with patch.object(probe, "signature", new=baseline_signature):
        _, tried, _ = probe.enumerate_programs(
            rows, target, task["max_tree_ops"])
    if tried != baseline_calls:
        raise AssertionError("enumerator root signature counter differs")
    return {"task": task["id"], "guided_runs": runs,
            "eager_enumerator_candidate_signatures": baseline_calls,
            "streaming_early_stop_candidate_signatures": _streaming_work(task)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", nargs="+", type=int, default=[5, 40, 50])
    parser.add_argument("--fixtures", nargs="+", default=[
        "fixtures/rsi_shadow_tasks.json", "fixtures/rsi_novel_tasks.json",
        "fixtures/rsi_ood_plain.json"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    epochs = sorted(set(args.epochs))
    if not epochs or epochs[0] <= 0:
        parser.error("epochs must be positive")
    torch.set_num_threads(1)
    before = _source_hash()
    families = {path: json.loads((ROOT / path).read_text())["tasks"]
                for path in args.fixtures}
    built = {path: [_build(task) for task in tasks]
             for path, tasks in families.items()}
    shadow = str(ROOT / "fixtures/rsi_shadow_tasks.json")
    groups = SearchPolicy().trace_groups(shadow)
    records = []
    for seed in args.seeds:
        ranker = SearchPolicy(seed=seed)
        previous = 0
        for epoch in [0] + epochs:
            ranker.fit(groups, epochs=epoch - previous)
            previous = epoch
            for path, data in built.items():
                for item in data:
                    scores = ranker.scores(_features(item))
                    if not all(map(math.isfinite, scores)):
                        raise ValueError("nonfinite scores: " + item["task"])
                    total = len(scores)
                    records.append({
                        "fixture": path, "task": item["task"],
                        "seed": seed, "epochs": epoch,
                        "layer_candidates": total,
                        "train_solvers": len(item["solvers"]),
                        "held_solvers": len(item["held_solvers"]),
                        "probe_position": min(item["solvers"]) + 1,
                        "random_expected": (total + 1) /
                        (len(item["solvers"]) + 1),
                        "position": _position(scores, item["solvers"]),
                        "held_position": _position(
                            scores, item["held_solvers"])})
            print("seed", seed, "epochs", epoch, flush=True)
    summary = []
    for path in families:
        for epoch in [0] + epochs:
            rows = [row for row in records
                    if row["fixture"] == path and row["epochs"] == epoch]
            summary.append({"fixture": path, "epochs": epoch,
                            "tasks": len(families[path]),
                            "seeds": args.seeds,
                            **{key: sum(row[key] for row in rows) / len(rows)
                               for key in ("position", "probe_position",
                                           "random_expected")}})
    work = [{"fixture": path, **_instrument(task, ranker)}
            for path, tasks in families.items() for task in tasks]
    if _source_hash() != before:
        raise RuntimeError("source changed during measurement")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "positions.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for name, value in (("summary", summary), ("work", work)):
        (args.out / (name + ".json")).write_text(json.dumps(value, indent=2))
    train_tables = {json.dumps(task["train"], sort_keys=True)
                    for task in json.loads(Path(shadow).read_text())["tasks"]}
    manifest = {"source_sha256": before, "feature_encoding": FEATURE_ENCODING,
                "seeds": args.seeds, "epochs": epochs,
                "training_fixture": str(Path(shadow).relative_to(ROOT)),
                "torch_threads": 1,
                "training_labels": "final-layer train-output equality",
                "torch_version": torch.__version__,
                "random_position_baseline": "exact expectation (N+1)/(K+1)",
                "signature_unit": "root vectors; excludes recursive calls",
                "streaming_comparator": "early-stop plain arithmetic search",
                "train_table_overlap_with_shadow": {
                    path: sum(json.dumps(task["train"], sort_keys=True)
                              in train_tables for task in tasks)
                    for path, tasks in families.items()},
                "fixtures_sha256": {
                    path: hashlib.sha256(
                        (ROOT / path).read_bytes()).hexdigest()
                    for path in families}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
