#!/usr/bin/env python3
"""Exercise the independent result auditor with isolated corrupt bundles."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import result_audit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_BUNDLE = ROOT / "artifacts/benchmark-pre-eval-v2-20261018"
PROTOCOL = result_audit.input_audit.load_json(
    result_audit.input_audit.PROTOCOL_PATH)
CORPUS = result_audit.input_audit.load_json(
    result_audit.input_audit.CORPUS_PATH)


def digest(path: Path) -> str:
    """Return SHA256 of one temporary artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    """Read a temporary JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Write canonical-format pretty JSON used by the result bundle."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def update_manifest_hash(bundle: Path, field: str,
                         target_name: str) -> None:
    """Refresh the output hash that the runner normally freezes."""
    manifest_path = bundle / "manifest.json"
    manifest = read_json(manifest_path)
    manifest[field] = digest(bundle / target_name)
    write_json(manifest_path, manifest)


def bundle_digest(bundle: Path) -> str:
    """Hash relative file names and contents for immutability evidence."""
    state = hashlib.sha256()
    for path in sorted(bundle.rglob("*")):
        if path.is_file():
            state.update(str(path.relative_to(bundle)).encode("utf-8"))
            state.update(b"\0")
            state.update(path.read_bytes())
            state.update(b"\0")
    return state.hexdigest()


def mutate_rows(bundle: Path,
                mutation: Callable[[list[dict[str, Any]]], None]) -> None:
    """Mutate JSONL rows and refresh the frozen raw-output digest."""
    path = bundle / "raw.jsonl"
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line]
    mutation(rows)
    path.write_text("".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        for row in rows), encoding="utf-8")
    update_manifest_hash(bundle, "raw_jsonl_sha256", "raw.jsonl")


def run_case(name: str, mutation: Callable[[Path], None],
             expected_text: str, parent: Path) -> dict[str, Any]:
    """Audit one corrupted temporary bundle and check the expected failure."""
    bundle = parent / name
    shutil.copytree(SOURCE_BUNDLE, bundle)
    mutation(bundle)
    errors, details = result_audit.verify_raw(bundle, PROTOCOL, CORPUS)
    matched = [error for error in errors if expected_text in error]
    return {
        "name": name,
        "rejected": bool(errors),
        "expected_reason_found": bool(matched),
        "expected_reason": expected_text,
        "matching_errors": matched,
        "error_count": len(errors),
        "details": details,
    }


def main() -> int:
    started = time.perf_counter()
    source_hash_before = bundle_digest(SOURCE_BUNDLE)

    def duplicate_cell(bundle: Path) -> None:
        def mutate(rows: list[dict[str, Any]]) -> None:
            rows[1]["seed"] = rows[0]["seed"]
            rows[1]["control"] = rows[0]["control"]
            rows[1]["task_id"] = rows[0]["task_id"]
            rows[1]["budget"] = rows[0]["budget"]
        mutate_rows(bundle, mutate)

    def wrong_graph(bundle: Path) -> None:
        def mutate(rows: list[dict[str, Any]]) -> None:
            row = next(row for row in rows if row["accepted"])
            row["returned_graph"] = {
                "input_types": {"x": "Int", "y": "Int"},
                "nodes": [], "output": "x",
            }
        mutate_rows(bundle, mutate)

    def bad_training_ordinal(bundle: Path) -> None:
        path = bundle / "training.json"
        training = read_json(path)
        feature = training["training_groups"]["groups"][0][
            "selected_candidates"][0]["feature_vector"]
        feature[-2] += 0.125
        write_json(path, training)
        update_manifest_hash(bundle, "training_json_sha256", "training.json")

    def bad_aggregate(bundle: Path) -> None:
        path = bundle / "aggregate.json"
        aggregate = read_json(path)
        aggregate["overall_means_across_seeds"][0][
            "mean_seed_fraction"] += 0.125
        write_json(path, aggregate)
        update_manifest_hash(bundle, "aggregate_json_sha256", "aggregate.json")

    def forged_fitted_state(bundle: Path) -> None:
        def mutate(rows: list[dict[str, Any]]) -> None:
            row = next(row for row in rows
                       if row["control"] == "trained_snn")
            row["state_fingerprint_before_eval"] = "forged-consistent-state"
            row["state_fingerprint_after_eval"] = "forged-consistent-state"
        mutate_rows(bundle, mutate)

    cases = [
        ("duplicate-cell", duplicate_cell, "duplicated matrix key"),
        ("wrong-accepted-graph", wrong_graph,
         "accepted flag bypasses independent gates"),
        ("tampered-training-feature", bad_training_ordinal,
         "selected_candidates differs"),
        ("tampered-aggregate", bad_aggregate,
         "aggregate or paired bootstrap differs"),
        ("forged-model-state", forged_fitted_state,
         "state differs from fitted model"),
    ]
    with tempfile.TemporaryDirectory(prefix="rsi-v2-audit-selfcheck-") as tmp:
        parent = Path(tmp)
        results = [run_case(name, mutation, reason, parent)
                   for name, mutation, reason in cases]
    source_hash_after = bundle_digest(SOURCE_BUNDLE)
    report = {
        "ok": all(case["rejected"] and case["expected_reason_found"]
                  for case in results),
        "command": "python3 " + str(HERE / "selfcheck.py"),
        "source_bundle_sha256_before": source_hash_before,
        "source_bundle_sha256_after": source_hash_after,
        "source_bundle_unchanged": source_hash_before == source_hash_after,
        "cases": results,
        "duration_seconds": time.perf_counter() - started,
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
