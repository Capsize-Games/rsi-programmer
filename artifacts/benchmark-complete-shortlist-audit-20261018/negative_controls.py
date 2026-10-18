#!/usr/bin/env python3
"""Semantic corruption controls for the independent shortlist auditor.

Mutations occur only in temporary copies. Bundle file hashes are refreshed so
these controls test the auditor's semantic checks rather than stale hashes.
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
AUDITOR_PATH = HERE / "complete_shortlist_audit.py"
SPEC = importlib.util.spec_from_file_location(
    "complete_shortlist_audit", AUDITOR_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load independent auditor")
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def refresh_hashes(bundle: Path, *names: str) -> None:
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    for name in names:
        manifest.setdefault("files_sha256", {})[name] = hashlib.sha256(
            (bundle / name).read_bytes()
        ).hexdigest()
    write_json(path, manifest)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def duplicate_key(bundle: Path) -> None:
    path = bundle / "raw.jsonl"
    with path.open("a") as handle:
        handle.write(path.read_text().splitlines()[0] + "\n")
    refresh_hashes(bundle, "raw.jsonl")


def reorder_rows(bundle: Path) -> None:
    path = bundle / "raw.jsonl"
    lines = path.read_text().splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    path.write_text("\n".join(lines) + "\n")
    refresh_hashes(bundle, "raw.jsonl")


def wrong_accepted_graph(bundle: Path) -> None:
    path = bundle / "raw.jsonl"
    rows = read_rows(path)
    row = rows[0]
    row["returned_graph"] = {
        "input_types": {"x": "Int", "y": "Int"},
        "nodes": [{"op": "add", "type": "Int", "args": ["x", "y"]}],
        "output": 0,
    }
    row["accepted"] = True
    row["solved"] = True
    row.setdefault("validation", {}).update(
        {
            "train_valid": True,
            "held_out_valid": True,
            "exact_function_match": True,
        }
    )
    write_rows(path, rows)
    refresh_hashes(bundle, "raw.jsonl")


def forged_paired_ratio(bundle: Path) -> None:
    path = bundle / "aggregate.json"
    value = json.loads(path.read_text())
    key = "complete_shortlist_over_full_cached_cpu_ratio"
    value["paired_cost_ratios"][0][key] += 0.25
    write_json(path, value)
    refresh_hashes(bundle, "aggregate.json")


def forged_selection_charge(bundle: Path) -> None:
    path = bundle / "training.json"
    value = json.loads(path.read_text())
    value["selection"]["total_cpu_ns"] += 1_000_000
    write_json(path, value)
    refresh_hashes(bundle, "training.json")


def forged_counter_partition(bundle: Path) -> None:
    path = bundle / "raw.jsonl"
    rows = read_rows(path)
    row = next(r for r in rows if r.get("engine") == "complete_shortlist")
    row["search"]["cheap_residual_arithmetic_operations"] += 1
    row["work"]["cheap_residual_arithmetic_operations"] += 1
    write_rows(path, rows)
    refresh_hashes(bundle, "raw.jsonl")


CONTROLS = {
    "duplicate_raw_key": (duplicate_key, "duplicate raw key"),
    "reordered_raw_rows": (reorder_rows, "raw row order differs"),
    "wrong_accepted_graph": (
        wrong_accepted_graph,
        "accepted flag inconsistent",
    ),
    "forged_paired_ratio_plus_0_25": (
        forged_paired_ratio,
        "paired_cost_ratios[",
    ),
    "forged_selection_charge": (
        forged_selection_charge,
        "training selection total_cpu_ns",
    ),
    "forged_counter_partition": (
        forged_counter_partition,
        "cheap_residual_arithmetic_operations identity mismatch",
    ),
}


def run_control(bundle: Path, name: str) -> dict[str, Any]:
    mutate, expected = CONTROLS[name]
    with tempfile.TemporaryDirectory(prefix=f"shortlist-audit-{name}-") as tmp:
        copy = Path(tmp) / "bundle"
        shutil.copytree(bundle, copy)
        mutate(copy)
        result = AUDITOR.audit(copy)
    errors = result.get("errors", [])
    if not any(expected in error for error in errors):
        raise AssertionError(f"{name}: expected {expected!r}; got {errors!r}")
    return {"control": name, "expected_rejection": expected, "passed": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--control", choices=(*CONTROLS, "all"), default="all")
    args = parser.parse_args()
    names = list(CONTROLS) if args.control == "all" else [args.control]
    results = [run_control(args.bundle, name) for name in names]
    output = {
        "report_type": "complete-shortlist-auditor-negative-controls",
        "source_bundle": str(args.bundle.resolve()),
        "source_bundle_modified": False,
        "results": results,
        "all_passed": all(r["passed"] for r in results),
    }
    encoded = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
