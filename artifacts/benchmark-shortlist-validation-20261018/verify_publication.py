"""Check published evidence and final source identity before tagging."""

from __future__ import annotations

import datetime
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from run_stage import snapshot

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
BUNDLE = ART / "benchmark-complete-shortlist-v1-20261018"
AUDIT = ART / "benchmark-complete-shortlist-audit-20261018"
PLOTS = ART / "benchmark-complete-shortlist-plots-20261018"
REPORT = ART / "benchmark-complete-shortlist-report-20261018"
PROGRESS = ART / "benchmark-complete-shortlist-progress-20261018"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    stages = [read(HERE / (name + ".json")) for name in (
        "focused-verified", "fast-verified", "measurement")]
    baseline = stages[0]["files_before"]
    for stage in stages:
        assert stage["exit_code"] == 0 and stage["source_unchanged"]
        assert stage["files_before"] == stage["files_after"] == baseline
    assert snapshot() == baseline
    manifest = read(BUNDLE / "manifest.json")
    assert manifest["infrastructure_valid"]
    for name, expected in manifest["files_sha256"].items():
        assert digest(BUNDLE / name) == expected, name
    audit = read(AUDIT / "main-audit-final-pass.json")
    assert audit["audit_passed"] and not audit["errors"]
    assert not audit["decision_supported_on_cost"]
    assert not audit["snn_efficiency_advantage_supported"]
    negatives = read(AUDIT / "negative-controls-final.json")
    assert negatives["all_passed"] and len(negatives["results"]) == 6
    assert all(row["passed"] for row in negatives["results"])
    assert not negatives["source_bundle_modified"]
    receipt = read(AUDIT / "final-audit-receipt.json")
    for name, field in (
        ("complete_shortlist_audit.py", "audit_script_sha256"),
        ("main-audit-final-pass.json", "corrected_audit_report_sha256"),
        ("negative_controls.py", "negative_controls_script_sha256"),
        ("negative-controls-final.json", "negative_control_report_sha256"),
    ):
        assert digest(AUDIT / name) == receipt[field], name
    assert digest(BUNDLE / "manifest.json") == receipt[
        "bundle_manifest_sha256"]
    plot = read(PLOTS / "execution.json")
    assert plot["status"] == "complete"
    assert digest(PLOTS / "plot_costs.py") == plot["script_sha256"]
    for name, expected in plot["input_sha256"].items():
        assert digest(BUNDLE / name) == expected, name
    for name, expected in plot["outputs"].items():
        assert digest(PLOTS / name) == expected, name
    report = read(REPORT / "run_receipt.json")
    assert report["exit_code"] == 0
    assert digest(REPORT / "report.py") == report["script_sha256"]
    for name, expected in report["bundle_sha256"].items():
        assert digest(BUNDLE / name) == expected, name
    for name, expected in report["outputs_sha256"].items():
        assert digest(REPORT / name) == expected, name
    memory = gzip.decompress(
        (PROGRESS / "memory-phase.jsonl.gz").read_bytes())
    assert memory == (BUNDLE / "memory.jsonl").read_bytes()
    prefix = gzip.decompress(
        (PROGRESS / "seed0-timed-prefix.jsonl.gz").read_bytes())
    assert (BUNDLE / "raw.jsonl").read_bytes().startswith(prefix)
    result = {
        "passed": True,
        "command": "flock /tmp/rsi-programmer-builder.lock "
        ".venv/bin/python artifacts/benchmark-shortlist-validation-20261018/"
        "verify_publication.py",
        "finished_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "script_sha256": digest(Path(__file__)),
        "source_files_identical_across_tests_measurement_and_now": len(
            baseline),
        "timed_rows": manifest["row_count"],
        "memory_rows": manifest["memory_row_count"],
        "audit_passed": True,
        "negative_controls_passed": 6,
        "plot_and_report_hashes_valid": True,
        "published_prefixes_match_final_bytes": True,
    }
    (HERE / "publication-check.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
