#!/usr/bin/env python3
"""Run the frozen complete-fallback shortlist and extended comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi_programmer.benchmark_shortlist import (
    PROTOCOL_PATH,
    expected_keys,
    load_protocol,
    run,
)
from rsi_programmer.benchmark_shortlist_summary import configurations
from rsi_programmer.benchmark_v2_runner import (
    _atomic_json,
    load_frozen_inputs,
    sha256_bytes,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/benchmark-complete-shortlist-v1",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    existed = args.output.exists()
    try:
        if args.check_only:
            protocol = load_protocol(args.protocol)
            _, corpus = load_frozen_inputs()
            memory_cells = (
                len(protocol["memory"]["modes"])
                * len(corpus["splits"]["eval"])
                * len(configurations(protocol))
            )
            print(
                json.dumps(
                    {
                        "preflight_valid": True,
                        "timed_cells": len(expected_keys(protocol, corpus)),
                        "memory_cells": memory_cells,
                    }
                )
            )
            return 0
        manifest = run(args.output, args.protocol)
    except Exception as exc:
        manifest = {
            "infrastructure_valid": False,
            "run_error": type(exc).__name__ + ": " + str(exc),
        }
        if not existed and not args.check_only:
            args.output.mkdir(parents=True, exist_ok=True)
            manifest["files_sha256"] = {
                name: sha256_bytes((args.output / name).read_bytes())
                for name in ("raw.jsonl", "memory.jsonl", "training.json")
                if (args.output / name).exists()
            }
            _atomic_json(args.output / "failure.json", manifest)
        print(json.dumps(manifest, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0 if manifest["infrastructure_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
