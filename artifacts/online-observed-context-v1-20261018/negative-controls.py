"""Run semantic and provenance negative controls on a context bundle."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

ROOT = Path(sys.argv[1]).resolve()
BUNDLE = Path(sys.argv[2]).resolve()
SOURCE_ROOT = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(ROOT))
from rsi_programmer.online_benchmark import load_corpus
from rsi_programmer.online_context_benchmark import load_protocol
from rsi_programmer.online_observed_context import ObservedContextPolicy
from scripts.audit_online_observed_context import (
    _audit_capacity,
    _audit_summary,
    _validate_row,
    audit_bundle,
)
from scripts.audit_online_graph import _sha256


def expect_reject(name: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except (ValueError, KeyError, StopIteration) as exc:
        print(f"{name}: rejected: {type(exc).__name__}: {exc}")
    else:
        raise AssertionError(f"{name}: corruption was accepted")


def read_first_row(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.loads(next(stream))


def prepare_first_row(row: dict[str, Any]) -> tuple[
    dict[str, Any], dict[str, Any], ObservedContextPolicy, dict[str, Any]
]:
    protocol = load_protocol()
    order = protocol["orders"][0]
    corpus = load_corpus()
    task = next(task for task in corpus["tasks"]
                if task["id"] == order["task_ids"][0])
    policy = ObservedContextPolicy(seed=protocol["model_seeds"][0])
    before = policy.metrics()
    policy.set_observed_rows(tuple(tuple(values)
                                   for values in task["observed"]))
    return order, task, policy, before


manifest = json.loads((BUNDLE / "manifest.json").read_text())
protocol = load_protocol()
first = read_first_row(BUNDLE / "episodes.jsonl.gz")
order, task, policy, before = prepare_first_row(first)
_validate_row(first, order, task, 0, protocol["budgets"][0], 0,
              policy, before)
print("unaltered first row: accepted", flush=True)
mutations: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("observed input hash", lambda row: row.__setitem__(
        "context_input_sha256", "0" * 64)),
    ("projection hash", lambda row: row.__setitem__(
        "context_projection_sha256", "0" * 64)),
    ("context work", lambda row: row["context_work"].__setitem__(
        "projection_mac", row["context_work"]["projection_mac"] - 1)),
    ("cold probability", lambda row: row[
        "diagnostic_before"]["probabilities"].__setitem__(
            0, row["diagnostic_before"]["probabilities"][0] + 0.05)),
    ("context CPU", lambda row: row.__setitem__("context_cpu_seconds", -1.0)),
)
for label, mutate in mutations:
    bad_row = copy.deepcopy(first)
    mutate(bad_row)
    order, task, policy, before = prepare_first_row(bad_row)
    expect_reject(label, lambda: _validate_row(
        bad_row, order, task, 0, protocol["budgets"][0], 0,
        policy, before))

with tempfile.TemporaryDirectory(prefix="observed-context-negative-") as temp:
    root = Path(temp)
    capacity_dir = root / "capacity"
    capacity_dir.mkdir()
    capacity = json.loads((BUNDLE / "capacity.json").read_text())
    capacity["rows"][0]["rates"][0][0] += 0.125
    capacity_path = capacity_dir / "capacity.json"
    capacity_path.write_text(json.dumps(capacity))
    bad_manifest = copy.deepcopy(manifest)
    bad_manifest["capacity_sha256"] = _sha256(capacity_path.read_bytes())
    (capacity_dir / "manifest.json").write_text(
        json.dumps(bad_manifest))
    expect_reject("capacity side study", lambda: _audit_capacity(
        capacity_dir, bad_manifest, load_corpus()["tasks"]))

    compact: list[dict[str, Any]] = []
    with gzip.open(BUNDLE / "episodes.jsonl.gz", "rt",
                   encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            compact.append({key: value for key, value in row.items()
                            if key != "attempts"})
    bad_summary = json.loads((BUNDLE / "summary.json").read_text())
    bad_summary["per_order"][0]["per_seed"][0]["total_cpu_seconds"] += 1.0
    summary_dir = root / "summary"
    summary_dir.mkdir()
    (summary_dir / "summary.json").write_text(json.dumps(bad_summary))
    orders = protocol["orders"]
    analysis_rows = [{**row,
                      "reset_cpu_seconds": row["reset_cpu_seconds"]
                      + row["context_cpu_seconds"]}
                     for row in compact]
    expect_reject("charged summary CPU", lambda: _audit_summary(
        summary_dir, manifest, analysis_rows, orders, protocol))

    missing_dir = root / "missing-first-order"
    missing_dir.mkdir()
    for name in ("manifest.json", "summary.json", "capacity.json"):
        shutil.copyfile(BUNDLE / name, missing_dir / name)
    raw_out = missing_dir / "episodes.jsonl.gz"
    skip_rows = (len(protocol["budgets"]) * len(protocol["model_seeds"])
                 * len(protocol["arms"]) * len(orders[0]["task_ids"]))
    with gzip.open(BUNDLE / "episodes.jsonl.gz", "rt",
                   encoding="utf-8") as source, \
            gzip.open(raw_out, "wt", encoding="utf-8") as target:
        for index, line in enumerate(source):
            if index >= skip_rows:
                target.write(line)
    missing_manifest = json.loads((missing_dir / "manifest.json").read_text())
    missing_manifest["raw_sha256"] = _sha256(raw_out.read_bytes())
    (missing_dir / "manifest.json").write_text(
        json.dumps(missing_manifest))
    expect_reject("missing first order", lambda: audit_bundle(missing_dir, repo_root=SOURCE_ROOT))
