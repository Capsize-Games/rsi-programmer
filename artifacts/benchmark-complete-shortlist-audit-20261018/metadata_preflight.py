#!/usr/bin/env python3
"""Metadata-only preflight for the frozen shortlist measurement plan.

This does not open the evaluation bundle or import project code. It verifies
frozen protocol/selection pins and derives the declared 11,808 timed and 960
memory-row counts before evaluation starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "fixtures/benchmark_complete_shortlist_v1.json"
PROTOCOL_SHA256 = (
    "189fa147a11040c6a05ef173522a6146285de616996724adff49d17b5f66be0a"
)
SELECTION_PROTOCOL = ROOT / "fixtures/benchmark_shortlist_selection_v1.json"
SELECTION_PROTOCOL_SHA256 = (
    "64eb64fb2b029be4736f9170df8fa4d2956c954d85e8133952f7981a416ac616"
)
CORPUS = ROOT / "fixtures/benchmark_pre_eval_corpus_v2.json"
CORPUS_SHA256 = (
    "a8fd6add90b34da732da8dc96f2c0fafcabc2f017a17cdef4b31303eec4ba921"
)
SELECTION_MANIFEST = (
    ROOT / "artifacts/benchmark-shortlist-selection-run-20261018/manifest.json"
)
SELECTION_MANIFEST_SHA256 = (
    "922d906513e041f09941d35e3538e4df9e6055200fc601d94121d222e1616f39"
)
SELECTION_RECEIPT = (
    ROOT / "artifacts/benchmark-shortlist-validation-20261018/selection.json"
)
SELECTION_RECEIPT_SHA256 = (
    "ec893207349cb5e181914e54d81dc2c7b4de35aa4eaf4f571783e77770ea4173"
)
SELECTION_REPORT = ROOT / (
    "artifacts/benchmark-complete-shortlist-audit-20261018/"
    "selection-audit.json"
)
SELECTION_REPORT_SHA256 = (
    "37a1c12abde26147e592aacd56cb865bfb3e7cf1eef8e2e70a35fb07a0261b9b"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"{label} missing or hash mismatch")


def preflight() -> dict[str, Any]:
    check(PROTOCOL, PROTOCOL_SHA256, "main protocol")
    check(SELECTION_PROTOCOL, SELECTION_PROTOCOL_SHA256, "selection protocol")
    check(CORPUS, CORPUS_SHA256, "evaluation corpus")
    check(SELECTION_MANIFEST, SELECTION_MANIFEST_SHA256, "selection manifest")
    check(SELECTION_RECEIPT, SELECTION_RECEIPT_SHA256, "selection receipt")
    protocol = json.loads(PROTOCOL.read_bytes())
    manifest = json.loads(SELECTION_MANIFEST.read_bytes())
    receipt = json.loads(SELECTION_RECEIPT.read_bytes())
    evidence = protocol["selection_evidence"]
    if (
        protocol.get("status") != "frozen-before-evaluation-measurement"
        or evidence.get("status") != "complete"
        or evidence.get("selected")
        != {
            key: protocol["method"][key]
            for key in ("window_size", "shortlist_size", "cheap_row_index")
        }
        or evidence.get("actual_attempts") != manifest.get("actual_attempts")
        or manifest.get("matrix_valid") is not True
        or manifest.get("source_unchanged") is not True
        or receipt.get("exit_code") != 0
        or receipt.get("source_unchanged") is not True
    ):
        raise ValueError("protocol or selector readiness metadata is invalid")
    for name, expected in manifest.get("files_sha256", {}).items():
        member = SELECTION_MANIFEST.parent / name
        check(member, expected, "selection artifact " + name)
    ev = protocol["evaluation"]
    timed = (
        len(protocol["training"]["seeds"])
        * ev["task_count"]
        * ev["repeats"]
        * (
            len(ev["streaming_budgets"])
            + len(ev["ranked_budgets"])
            + len(ev["ranked_controls"])
            * len(ev["ranked_engines"])
            * len(ev["ranked_budgets"])
        )
    )
    configs = 2 + len(ev["ranked_controls"]) * len(ev["ranked_engines"])
    memory = len(protocol["memory"]["modes"]) * ev["task_count"] * configs
    if (
        timed != 11808
        or memory != 960
        or timed != ev["matrix_rows"]
        or memory != protocol["memory"]["matrix_rows"]
    ):
        raise ValueError("derived matrix counts differ from frozen protocol")
    check(
        SELECTION_REPORT,
        SELECTION_REPORT_SHA256,
        "independent selector audit report",
    )
    selector_report = json.loads(SELECTION_REPORT.read_bytes())
    if (
        selector_report.get("passed") is not True
        or selector_report.get("main_protocol_sha256") != PROTOCOL_SHA256
        or selector_report.get("selected")
        != {
            key: protocol["method"][key]
            for key in ("window_size", "shortlist_size", "cheap_row_index")
        }
    ):
        raise ValueError("independent selector audit is absent or mismatched")
    return {
        "preflight_passed": True,
        "main_protocol_sha256": PROTOCOL_SHA256,
        "selection_protocol_sha256": SELECTION_PROTOCOL_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "selection_manifest_sha256": SELECTION_MANIFEST_SHA256,
        "selection_receipt_sha256": SELECTION_RECEIPT_SHA256,
        "selected": protocol["method"],
        "expected_timed_rows": timed,
        "expected_memory_rows": memory,
        "evaluation_bundle_read": False,
        "evaluation_measurement_started_or_inferred": False,
        "selector_audit_report": str(SELECTION_REPORT.relative_to(ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = preflight()
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
