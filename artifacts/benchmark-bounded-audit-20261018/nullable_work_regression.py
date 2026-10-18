#!/usr/bin/env python3
"""Regression checks for nullable work totals and bounded peak strictness."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "bounded_audit", HERE / "bounded_audit.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load bounded auditor")
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def refresh_raw_hash(bundle: Path) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    raw = (bundle / "raw.jsonl").read_bytes()
    manifest["files_sha256"]["raw.jsonl"] = hashlib.sha256(raw).hexdigest()
    write_json(manifest_path, manifest)


def check_totals() -> dict[str, Any]:
    mixed = AUDITOR.work_sum(
        [
            {"work": {"peak": 4, "zero": 0, "count": 3}},
            {"work": {"peak": None, "zero": 0, "count": 8}},
            {"work": {"peak": 7, "zero": 0, "count": 2}},
        ]
    )
    if mixed != {"count": 13, "peak": None, "zero": 0}:
        raise AssertionError(f"nullable numeric work totals wrong: {mixed!r}")
    return {"mixed_null": None, "known_zero": 0, "numeric_sum": 13}


def check_bounded_peak(bundle: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="bounded-null-peak-") as tmp:
        copy = Path(tmp) / "bundle"
        shutil.copytree(bundle, copy)
        raw_path = copy / "raw.jsonl"
        rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
        target = next(row for row in rows if row.get("engine") == "bounded")
        target["search"]["peak_live_descriptors"] = None
        target["work"]["peak_live_descriptors"] = None
        raw_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
        refresh_raw_hash(copy)
        result = AUDITOR.audit(copy)
    errors = result.get("errors", [])
    expected = "bounded descriptor peak missing or invalid"
    if not any(expected in error for error in errors):
        raise AssertionError(
            f"null bounded peak was not rejected as invalid: {errors!r}"
        )
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if not args.bundle.is_dir():
        parser.error("bundle directory does not exist")
    result = {
        "report_type": "nullable-work-auditor-regression",
        "source_bundle": str(args.bundle.resolve()),
        "source_bundle_modified": False,
        "work_totals": check_totals(),
        "bounded_null_peak_rejection": check_bounded_peak(args.bundle),
        "passed": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
