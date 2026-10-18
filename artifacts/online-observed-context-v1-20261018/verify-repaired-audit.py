"""Verify frozen measurements using the corrected protocol validator.

The measured source stays immutable. Verify it in an archive, and separately
prove that current execution code differs only in the auditor and its test.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT, ARCHIVE, BUNDLE = (Path(value).resolve() for value in sys.argv[1:4])
sys.path.insert(0, str(ROOT))
from rsi_programmer.online_benchmark import source_manifest
from scripts.audit_online_graph import verify_source_hashes
from scripts.audit_online_observed_context import audit_bundle

manifest = json.loads((BUNDLE / "manifest.json").read_text())
original = manifest["source_files"]
verify_source_hashes(original, ARCHIVE)
current = source_manifest()
if set(original) != set(current):
    raise ValueError("corrected checkout changed the source path set")
changes = {name: {"measured_sha256": original[name],
                  "corrected_sha256": current[name]}
           for name in original if current[name] != original[name]}
expected = {"scripts/audit_online_observed_context.py",
            "tests/test_audit_observed_context.py"}
if set(changes) != expected:
    raise ValueError("changes extend beyond the auditor and its test")
result = audit_bundle(BUNDLE, repo_root=ARCHIVE)
receipt = {
    "status": "passed",
    "measured_source": manifest["git_head"],
    "measured_manifest_sha256": hashlib.sha256(
        (BUNDLE / "manifest.json").read_bytes()).hexdigest(),
    "raw_sha256": manifest["raw_sha256"],
    "archived_source_verified": True,
    "current_execution_sources_match_measurement": True,
    "source_path_count": len(original),
    "analysis_only_changes": changes,
    "reason": "Validator expected an obsolete metadata key; corrected to "
              "the frozen answer-metadata flag and opaque-ID guard contract",
    "audit": result,
}
print(json.dumps(receipt, indent=2, sort_keys=True))
