#!/usr/bin/env python3
"""Exercise auditor rejection paths against copied, completed bundles.

The supplied bundle is read-only. Each control mutates a temporary copy and
refreshes file hashes so rejection comes from semantic auditing, not stale
bundle checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
AUDITOR_PATH = HERE / "bounded_audit.py"
SPEC = importlib.util.spec_from_file_location("bounded_audit", AUDITOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load bounded auditor")
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def refresh_hashes(bundle: Path, *names: str) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = manifest.setdefault("files_sha256", {})
    for name in names:
        hashes[name] = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
    write_json(manifest_path, manifest)


def raw_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def save_raw(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def mutate_duplicate_raw_key(bundle: Path) -> None:
    path = bundle / "raw.jsonl"
    with path.open("a") as handle:
        handle.write(path.read_text().splitlines()[0] + "\n")
    refresh_hashes(bundle, "raw.jsonl")


def mutate_raw_sequence(bundle: Path) -> None:
    path = bundle / "raw.jsonl"
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise RuntimeError("completed bundle has too few raw rows to reorder")
    lines[0], lines[1] = lines[1], lines[0]
    path.write_text("\n".join(lines) + "\n")
    refresh_hashes(bundle, "raw.jsonl")


def mutate_forged_ratio(bundle: Path) -> None:
    path = bundle / "aggregate.json"
    aggregate = json.loads(path.read_text())
    ratios = aggregate["paired_cost_ratios"]
    ratios[0]["bounded_over_full_cached_cpu_ratio"] += 0.25
    write_json(path, aggregate)
    refresh_hashes(bundle, "aggregate.json")


def mutate_wrong_accepted_graph(bundle: Path) -> None:
    path = bundle / "raw.jsonl"
    rows = raw_rows(path)
    target = next((row for row in rows if row.get("accepted") is True), None)
    if target is None:
        raise RuntimeError(
            "completed bundle has no accepted row for graph control"
        )
    target["returned_graph"] = {
        "input_types": {"x": "Int", "y": "Int"},
        "nodes": [{"op": "add", "type": "Int", "args": ["x", "y"]}],
        "output": 0,
    }
    save_raw(path, rows)
    refresh_hashes(bundle, "raw.jsonl")


def mutate_training_charge(bundle: Path) -> None:
    path = bundle / "training.json"
    training = json.loads(path.read_text())
    field = "training_group_cpu_ns"
    if field not in training:
        raise RuntimeError(f"training record missing {field}")
    training[field] += 1
    write_json(path, training)
    refresh_hashes(bundle, "training.json")


CONTROLS = {
    "duplicate_raw_key": (mutate_duplicate_raw_key, "duplicate raw key"),
    "reordered_raw_rows": (
        mutate_raw_sequence,
        "raw row order differs from frozen schedule",
    ),
    "forged_ratio_plus_0_25": (
        mutate_forged_ratio,
        "aggregate paired_cost_ratios[",
    ),
    "wrong_accepted_graph": (
        mutate_wrong_accepted_graph,
        "accepted flag inconsistent",
    ),
    "forged_training_charge": (
        mutate_training_charge,
        "training_amortized_cpu_ns_per_task_by_N",
    ),
}


def run_control(source: Path, name: str) -> dict[str, Any]:
    mutate, expected_error = CONTROLS[name]
    with tempfile.TemporaryDirectory(prefix=f"bounded-audit-{name}-") as tmp:
        copy = Path(tmp) / "bundle"
        shutil.copytree(source, copy)
        mutate(copy)
        result = AUDITOR.audit(copy)
    errors = result.get("errors", [])
    if not any(expected_error in error for error in errors):
        raise AssertionError(
            f"{name}: expected rejection containing {expected_error!r}; "
            f"got {errors!r}"
        )
    return {
        "control": name,
        "expected_rejection": expected_error,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--control", choices=(*CONTROLS, "all"), default="all")
    args = parser.parse_args()
    if not args.bundle.is_dir():
        parser.error("bundle directory does not exist")
    names = list(CONTROLS) if args.control == "all" else [args.control]
    result = {
        "report_type": "bounded-auditor-negative-controls",
        "source_bundle": str(args.bundle.resolve()),
        "source_bundle_modified": False,
        "results": [run_control(args.bundle, name) for name in names],
        "all_passed": True,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
