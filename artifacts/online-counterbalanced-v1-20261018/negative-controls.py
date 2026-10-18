"""Run semantic and provenance negative controls on a completed bundle."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
BUNDLE = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT))
from rsi_programmer import online_benchmark as base
from scripts.audit_online_counterbalanced import (
    SCHEDULE, _audit_summary, _validate_counterbalanced_row, audit_bundle,
)
from scripts.audit_online_graph import _policy, verify_source_hashes


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def expect_reject(name: str, action: object) -> None:
    try:
        action()  # type: ignore[operator]
    except (ValueError, KeyError, StopIteration) as exc:
        print(f"{name}: rejected: {type(exc).__name__}: {exc}")
    else:
        raise AssertionError(f"{name}: corruption was accepted")


manifest = json.loads((BUNDLE / "manifest.json").read_text())
summary = json.loads((BUNDLE / "summary.json").read_text())
protocol = json.loads(SCHEDULE.read_text())["protocol"]
corpus = json.loads((ROOT / "fixtures/online_graph_v1.json").read_text())
orders = protocol["orders"]
tasks = {task["id"]: task for task in corpus["tasks"]}
first_order = orders[0]
first_arm = base.ARMS[0]
first_budget = base.BUDGETS[0]
first_seed = base.SEEDS[0]
first_task = first_order["task_ids"][0]
compact: list[dict[str, object]] = []
first: dict[str, object] | None = None
raw_path = BUNDLE / "episodes.jsonl.gz"
with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        if first is None:
            first = row
        compact.append({key: value for key, value in row.items()
                        if key != "attempts"})
if first is None:
    raise ValueError("empty episode file")

for label, mutate in (
    ("diagnostic entropy", lambda row: row["diagnostic_before"].__setitem__(
        "entropy_nats", row["diagnostic_before"]["entropy_nats"] + 0.5)),
    ("reward", lambda row: row["attempts"][0].__setitem__(
        "reward", row["attempts"][0]["reward"] + 0.25)),
    ("first phase position", lambda row: row.__setitem__("phase_position", 2)),
    ("incoming RNG offset", lambda row: row.__setitem__(
        "rng_draw_offset_before", row["rng_draw_offset_before"] + 1)),
):
    corrupt = copy.deepcopy(first)
    mutate(corrupt)
    expect_reject(label, lambda corrupt=corrupt: _validate_counterbalanced_row(
        corrupt, first_order, tasks[first_task], 0, first_budget, 0,
        _policy(first_arm, first_seed)))

with tempfile.TemporaryDirectory(prefix="counterbalanced-negative-") as temp:
    temp_root = Path(temp)
    missing_source = copy.deepcopy(manifest["source_files"])
    missing_source.pop(next(iter(missing_source)))
    expect_reject("omitted source hash", lambda: verify_source_hashes(
        missing_source, ROOT))

    bad_summary = copy.deepcopy(summary)
    bad_summary["per_order"][0]["per_seed"][0]["total_cpu_seconds"] += 1.0
    summary_dir = temp_root / "summary"
    summary_dir.mkdir()
    (summary_dir / "summary.json").write_text(json.dumps(bad_summary))
    expect_reject("summary CPU", lambda: _audit_summary(
        summary_dir, manifest, compact, orders, protocol))

    missing_dir = temp_root / "missing-row"
    missing_dir.mkdir()
    shutil.copyfile(BUNDLE / "manifest.json", missing_dir / "manifest.json")
    shutil.copyfile(BUNDLE / "summary.json", missing_dir / "summary.json")
    truncated_path = missing_dir / "episodes.jsonl.gz"
    pending: str | None = None
    with gzip.open(raw_path, "rt", encoding="utf-8") as source, \
            gzip.open(truncated_path, "wt", encoding="utf-8") as target:
        for line in source:
            if pending is not None:
                target.write(pending)
            pending = line
    changed_manifest = json.loads((missing_dir / "manifest.json").read_text())
    changed_manifest["raw_sha256"] = digest(truncated_path.read_bytes())
    (missing_dir / "manifest.json").write_text(json.dumps(changed_manifest))
    expect_reject("missing final row", lambda: audit_bundle(missing_dir, ROOT))
