"""Bind the completed sensing result to unchanged verified source bytes."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    """Hash an exact file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Verify source equivalence, retained logs and sealed result hashes."""
    root = Path(__file__).resolve().parents[2]
    evidence = Path(__file__).resolve().parent
    bundle = root / "artifacts/sensing-fan-v1-20261018"
    plots = root / "artifacts/sensing-fan-plots-20261018"
    output = evidence / "publication-checks.json"
    if output.exists():
        raise FileExistsError("preserve prior publication evidence")
    manifest = json.loads((bundle / "manifest.json").read_text())
    source = manifest["source_files"]
    assert len(source) == 218
    implementation = "643be256363f2f23f9c51f280fb38d262a6f992d"
    for name, digest in source.items():
        assert sha(root / name) == digest, name
        archived = subprocess.check_output(
            ["git", "show", f"{implementation}:{name}"], cwd=root)
        assert hashlib.sha256(archived).hexdigest() == digest, name
    receipts = {}
    for label in ("affected-canonical-snapshot-final", "fast-auditor-final",
                  "frozen-benchmark", "independent-replay",
                  "corruption-controls"):
        path = evidence / f"{label}-receipt.json"
        receipt = json.loads(path.read_text())
        assert receipt["exit_code"] == 0 and receipt["source_unchanged"]
        assert receipt["source_files"] == source
        receipts[label] = sha(path)
    for name, digest in manifest["files"].items():
        assert sha(bundle / name) == digest, name
    for key in ("csv", "archive"):
        assert sha(root / manifest["dataset"][key]) == (
            manifest["dataset"][key + "_sha256"])
    for path in evidence.glob("*-receipt.json"):
        receipt = json.loads(path.read_text())
        if "log_sha256" not in receipt:
            continue
        label = path.name.removesuffix("-receipt.json")
        compressed = evidence / f"{label}.log.gz"
        raw = gzip.decompress(compressed.read_bytes())
        assert hashlib.sha256(raw).hexdigest() == receipt["log_sha256"]
    corrupt = json.loads((evidence / "corruption-checks.json").read_text())
    assert corrupt["all_rejected"] and len(corrupt["cases"]) == 8
    assert all(row["artifact_hash_rebound"] for row in corrupt["cases"])
    assert corrupt["original_manifest_sha256"] == sha(bundle / "manifest.json")
    audit = json.loads((evidence / "independent-audit.json").read_text())
    assert audit["summary_verified"] and audit["stream_rows"] == 924
    data = json.loads((plots / "plotted-aggregates.json").read_text())
    assert data["source_hashes"]["summary_json"] == sha(
        bundle / "summary.json")
    assert data["source_hashes"]["manifest_json"] == sha(
        bundle / "manifest.json")
    assert data["source_hashes"]["plot_script_sha256"] == sha(
        evidence / "plot_results.py")
    ci = json.loads((evidence / "ci-fast-evidence.json").read_text())
    for job in ci["completed_jobs"]:
        assert job["source"] == implementation
        assert job["conclusion"] == "success"
        assert job["test_runs"][0]["tests"] == 521
        raw = gzip.decompress((evidence / job["compressed_log"]).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == job["log_sha256"]
        assert not re.search(
            rb"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}", raw)
    paths = (list((root / "rsi_programmer").glob("sensing_*.py"))
             + list((root / "scripts").glob("*sensing*.py"))
             + list((root / "tests").glob("test_sensing*.py"))
             + list(evidence.glob("*.py")))
    for path in paths:
        text = path.read_text()
        tree = ast.parse(text)
        assert all(len(line) <= 79 for line in text.splitlines()), path
        assert sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree)) <= 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args.posonlyargs + node.args.args
                args += node.args.kwonlyargs
                args += [a for a in (node.args.vararg, node.args.kwarg) if a]
                assert node.returns is not None, path
                assert all(a.annotation is not None for a in args
                           if a.arg not in ("self", "cls")), path
    files = list(bundle.iterdir()) + list(plots.iterdir())
    files += [path for path in evidence.iterdir() if path.is_file()]
    result: dict[str, Any] = {
        "benchmark_source": manifest["git_head"],
        "implementation_source": implementation,
        "measured_source_files": len(source),
        "current_and_archived_sources_match_final_receipts": True,
        "sealed_results_and_dataset_unchanged": True,
        "all_log_receipts_verified": True,
        "independent_replay_and_eight_corruptions_verified": True,
        "plot_source_and_values_verified": True,
        "static_conventions_files": len(paths),
        "receipts": receipts,
        "full_ci": "pending; both fast jobs passed",
        "files": {str(path.relative_to(root)): sha(path)
                  for path in sorted(files)},
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("files", "receipts")}, indent=2))


if __name__ == "__main__":
    main()
