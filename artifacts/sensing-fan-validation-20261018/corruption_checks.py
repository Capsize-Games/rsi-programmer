"""Reject semantic corruptions of copies of the frozen result bundle."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from scripts.audit_sensing import audit_bundle


def main() -> None:
    """Rebind altered file hashes so rejection requires semantic replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve previous corruption evidence")
    results = []
    cases = ("prediction_next", "score", "model_hash_after", "alarm",
             "value", "source_slices", "fractional_work", "trained_hash")
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="rsi-sensing-corrupt-") as tmp:
            bundle = Path(tmp) / "bundle"
            shutil.copytree(args.bundle, bundle)
            name = ("training.json" if case == "trained_hash"
                    else "streams.jsonl.gz")
            if case == "trained_hash":
                path = bundle / name
                rows = json.loads(path.read_text())
                rows[0]["trained_weight_hash"] = "0" * 64
                path.write_text(json.dumps(rows) + "\n")
            else:
                path = bundle / name
                with gzip.open(path, "rt") as stream:
                    rows = [json.loads(line) for line in stream]
                row = rows[0]
                if case == "fractional_work":
                    row["work"]["updates"] += 0.0000000000001
                elif case == "source_slices":
                    row["source_slices"][0]["start"] += 10
                elif case == "model_hash_after":
                    row["windows"][6][case] = "0" * 64
                elif case == "alarm":
                    row["windows"][6][case] = not row["windows"][6][case]
                else:
                    row["windows"][6][case] += 0.01
                with gzip.open(path, "wt") as stream:
                    for row in rows:
                        stream.write(json.dumps(row) + "\n")
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"][name] = hashlib.sha256(
                (bundle / name).read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest) + "\n")
            try:
                audit_bundle(bundle)
            except ValueError as error:
                result: dict[str, Any] = {
                    "case": case, "rejected": True, "reason": str(error),
                    "artifact_hash_rebound": True,
                }
                if "artifact hash mismatch" in str(error):
                    raise AssertionError("expected a semantic rejection")
                results.append(result)
                print(json.dumps(result), flush=True)
            else:
                raise AssertionError(f"auditor accepted {case}")
    payload = {
        "original_manifest_sha256": hashlib.sha256(
            (args.bundle / "manifest.json").read_bytes()).hexdigest(),
        "cases": results, "all_rejected": True,
        "original_bundle_modified": False,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
