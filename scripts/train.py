#!/usr/bin/env python3
"""Train the RSI search-control policy (canonical reproduction)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from rsi_programmer.author import train_author
from rsi_programmer.policy import SearchPolicy


def _args() -> argparse.Namespace:
    """Seed, epochs, fixture, and output locations for one run."""
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=str(
        here / "fixtures" / "rsi_shadow_tasks.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--name", default="rsi-search-policy-w1")
    parser.add_argument("--out", default=str(here / "weights"))
    return parser.parse_args()


def _manifest(name: str, policy: SearchPolicy,
              fixture: str, seed: int, epochs: int,
              rows: int) -> dict[str, Any]:
    """Weights spec plus training provenance for one run."""
    snap = policy.snapshot()
    state = policy.weights()
    return {
        "name": name,
        "spec": snap["spec"],
        "params": sum(v.numel() for v in state.values()),
        "training": {
            "fixture": fixture,
            "epochs": epochs,
            "seed": seed,
            "trained_rows": rows,
            "snapshot_sha256": _snapshot_hash(snap["weights"]),
        },
        "license": "LicenseRef-Capsize-Proprietary",
        "license_file": "LICENSE",
    }


def _snapshot_hash(weights: dict[str, Any]) -> str:
    """Canonical sha256 over the JSON snapshot weights."""
    text = json.dumps(weights, sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    """Train, save weights and manifest, and print the digest."""
    args = _args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = SearchPolicy(seed=args.seed)
    rows = train_author(policy, args.fixture, epochs=args.epochs)
    blob_path = out / (args.name + ".pt")
    torch.save(policy.weights(), str(blob_path))
    manifest = _manifest(args.name, policy, args.fixture,
                         args.seed, args.epochs, rows)
    (out / (args.name + ".manifest.json")).write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    blob = blob_path.read_bytes()
    print("weights sha256:", hashlib.sha256(blob).hexdigest())
    print("size_bytes:", len(blob))
    print("snapshot:", manifest["training"]["snapshot_sha256"])


if __name__ == "__main__":
    main()
