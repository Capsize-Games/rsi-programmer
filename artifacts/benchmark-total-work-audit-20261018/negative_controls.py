#!/usr/bin/env python3
"""Create tampered bundle copies and require the independent auditor to reject them."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from total_work_audit import graph_polynomial, polynomial


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def refresh_manifest(bundle: Path, member: str) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    field = {
        "raw.jsonl": "raw_jsonl_sha256",
        "aggregate.json": "aggregate_json_sha256",
        "training.json": "training_json_sha256",
    }[member]
    manifest[field] = digest(bundle / member)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def make_cases(bundle: Path, temporary_root: Path) -> dict[str, Path]:
    cases = {
        "duplicate_row_key": temporary_root / "duplicate-row-key",
        "forged_cpu_aggregate_ratio": temporary_root
        / "forged-cpu-aggregate-ratio",
        "wrong_accepted_graph": temporary_root / "wrong-accepted-graph",
        "uncharged_training_cost": temporary_root / "uncharged-training-cost",
    }
    for case in cases.values():
        case.mkdir(parents=True, exist_ok=False)
        for name in (
            "raw.jsonl",
            "manifest.json",
            "training.json",
            "aggregate.json",
        ):
            shutil.copy2(bundle / name, case / name)

    duplicate = load_rows(cases["duplicate_row_key"] / "raw.jsonl")
    duplicate.append(duplicate[0].copy())
    write_rows(cases["duplicate_row_key"] / "raw.jsonl", duplicate)
    refresh_manifest(cases["duplicate_row_key"], "raw.jsonl")

    ratio_bundle = cases["forged_cpu_aggregate_ratio"]
    aggregate_path = ratio_bundle / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text())
    if not aggregate.get("paired_cost_ratios"):
        raise ValueError("aggregate has no paired cost ratios")
    ratio = aggregate["paired_cost_ratios"][0]
    current = ratio.get("optimized_over_baseline_cpu_ratio")
    ratio["optimized_over_baseline_cpu_ratio"] = (
        0.123456789 if current != 0.123456789 else 0.987654321
    )
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    refresh_manifest(ratio_bundle, "aggregate.json")

    graph_bundle = cases["wrong_accepted_graph"]
    rows = load_rows(graph_bundle / "raw.jsonl")
    selected = next(
        (
            row
            for row in rows
            if row.get("accepted") and row.get("returned_graph") is not None
        ),
        None,
    )
    if selected is None:
        raise ValueError(
            "no accepted graph is available for the graph control"
        )
    corpus_path = Path(__file__).resolve().parents[2] / (
        "fixtures/benchmark_pre_eval_corpus_v2.json"
    )
    corpus = json.loads(corpus_path.read_text())
    task = next(
        task
        for task in corpus["splits"]["eval"]
        if task["id"] == selected["task_id"]
    )
    expected = polynomial(task["generator"]["program"])
    candidates = [
        {"input_types": {"x": "Int", "y": "Int"}, "nodes": [], "output": name}
        for name in ("x", "y")
    ] + [
        {
            "input_types": {"x": "Int", "y": "Int"},
            "nodes": [{"op": op, "type": "Int", "args": ["x", "y"]}],
            "output": 0,
        }
        for op in ("add", "sub", "mul")
    ]
    selected["returned_graph"] = next(
        graph for graph in candidates if graph_polynomial(graph) != expected
    )
    write_rows(graph_bundle / "raw.jsonl", rows)
    refresh_manifest(graph_bundle, "raw.jsonl")

    training_bundle = cases["uncharged_training_cost"]
    aggregate_path = training_bundle / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text())
    learned = next(
        (
            item
            for item in aggregate.get("amortization", [])
            if item.get("control") in {"trained_snn", "mlp", "recurrent"}
        ),
        None,
    )
    if learned is None:
        raise ValueError("aggregate has no learned-control amortization row")
    learned["training_cpu_ns_mean_seed_including_full_setup"] = 0
    learned["training_cpu_ns_per_task"] = 0
    learned["amortized_cpu_ns_per_task"] = learned["mean_attempt_cpu_ns"]
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    refresh_manifest(training_bundle, "aggregate.json")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--auditor",
        type=Path,
        default=Path(__file__).with_name("total_work_audit.py"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {
        name: digest(args.bundle / name)
        for name in (
            "raw.jsonl",
            "manifest.json",
            "training.json",
            "aggregate.json",
        )
    }
    results = {}
    expected_reasons = {
        "duplicate_row_key": "duplicate row key at line",
        "forged_cpu_aggregate_ratio": "aggregate paired CPU ratios differ",
        "wrong_accepted_graph": "returned graph has wrong exact polynomial",
        "uncharged_training_cost": "training_amortization",
    }
    with tempfile.TemporaryDirectory(
        prefix=".negative-controls-", dir=args.output
    ) as temporary:
        cases = make_cases(args.bundle, Path(temporary))
        for name, case_bundle in cases.items():
            report = args.output / f"{name}-audit.json"
            command = [
                sys.executable,
                str(args.auditor),
                str(case_bundle),
                "--output",
                str(report),
            ]
            start_ns = time.monotonic_ns()
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            elapsed_ns = time.monotonic_ns() - start_ns
            if completed.returncode == 0 or not report.is_file():
                raise AssertionError(
                    f"auditor failed to reject {name}: {completed.stdout} {completed.stderr}"
                )
            payload = json.loads(report.read_text())
            if payload.get("valid") is not False or not payload.get("errors"):
                raise AssertionError(
                    f"{name} produced no recorded audit failure"
                )
            expected_reason = expected_reasons[name]
            reasons = payload.get("errors", [])
            details = json.dumps(
                payload.get("aggregate_summary_mismatches", [])
            )
            if (
                not any(expected_reason in error for error in reasons)
                and expected_reason not in details
            ):
                raise AssertionError(
                    f"{name} was rejected for the wrong reason: {reasons}"
                )
            results[name] = {
                "rejected_for_expected_reason": True,
                "expected_reason": expected_reason,
                "errors": reasons,
                "report": str(report.resolve()),
                "command": command,
                "exit_code": completed.returncode,
                "elapsed_wall_ns": elapsed_ns,
            }
    final_hashes = {name: digest(args.bundle / name) for name in source_hashes}
    if final_hashes != source_hashes:
        raise AssertionError(
            "source bundle changed while running negative controls"
        )
    receipt = args.output / "negative-control-results.json"
    receipt.write_text(
        json.dumps(
            {
                "source_bundle_before_sha256": source_hashes,
                "source_bundle_after_sha256": final_hashes,
                "cases": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "all_rejected_for_expected_reason": len(results) == 4,
                "receipt": str(receipt.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
