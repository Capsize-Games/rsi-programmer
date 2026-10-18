"""Verify measured source and recount saved outcome/update evidence."""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def digest(payload: bytes) -> str:
    """Return a SHA-256 digest of exact bytes."""
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    """Read saved records only; never instantiate or train a model."""
    root = Path(__file__).resolve().parents[2]
    bundle = root / "artifacts" / "online-plastic-encoder-v1-20261018"
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest["source_files"].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError(f"measured source changed: {name}")
        archived = subprocess.check_output(
            ["git", "show", f"{manifest['git_head']}:{name}"], cwd=root)
        if digest(archived) != expected:
            raise ValueError(f"archive differs: {name}")
    raw = bundle / "episodes.jsonl.gz"
    if digest(raw.read_bytes()) != manifest["episodes_raw_sha256"]:
        raise ValueError("episode SHA differs from manifest")
    counts: dict[str, Any] = {
        "episode_rows": 0, "observed_solutions": 0,
        "observed_solutions_rejected": 0, "plastic_primary_rows": 0,
        "plastic_primary_encoder_hash_changed_rows": 0,
        "plastic_primary_encoder_update_calls": 0,
        "plastic_primary_changed_synapse_updates": 0,
    }
    with gzip.open(raw, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            counts["episode_rows"] += 1
            if row["observed_solved"]:
                counts["observed_solutions"] += 1
                if not all(row[key] for key in (
                        "accepted", "held_correct", "polynomial_correct")):
                    counts["observed_solutions_rejected"] += 1
            if row["arm"] == "plastic_snn" and row["budget"] == 144:
                counts["plastic_primary_rows"] += 1
                changed = (row["encoder_hash_before"]
                           != row["encoder_hash_after"])
                counts["plastic_primary_encoder_hash_changed_rows"] += (
                    int(changed))
                for key, counter in (
                    ("plastic_primary_encoder_update_calls",
                     "plastic_encoder_updates"),
                    ("plastic_primary_changed_synapse_updates",
                     "plastic_encoder_changed_synapses"),
                ):
                    counts[key] += row["policy_work"][counter]
    if (counts["episode_rows"] != 13824
            or counts["plastic_primary_rows"] != 864
            or counts["observed_solutions_rejected"] != 0):
        raise ValueError("unexpected outcome matrix")
    receipt = {
        "source_head": manifest["git_head"],
        "measured_source_files_verified": len(manifest["source_files"]),
        "episode_sha256": manifest["episodes_raw_sha256"],
        "interpretation": "Saved output recount; no new learning run. "
        "Hash changes and update counts do not establish useful learning.",
        "counts": counts,
    }
    output = Path(__file__).with_name("source-and-outcome-proof.json")
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
