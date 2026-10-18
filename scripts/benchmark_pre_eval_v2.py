#!/usr/bin/env python3
"""Run the frozen pre-evaluation search benchmark v2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi_programmer.benchmark_v2_runner import (
    ROOT,
    _atomic_json,
    run_benchmark,
    sha256_bytes,
)
from rsi_programmer.pre_eval import pre_evaluation_search


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/benchmark-pre-eval-v2",
        help="new directory for raw rows and benchmark reports",
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    existed = output.exists()
    try:
        manifest = run_benchmark(output, pre_evaluation_search)
    except Exception as exc:
        if output.is_dir() and not existed:
            raw = output / "raw.jsonl"
            failure: dict[str, Any] = {
                "row_type": "infrastructure_failure",
                "error": type(exc).__name__ + ": " + str(exc),
            }
            if raw.exists():
                with raw.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, sort_keys=True) + "\n")
            manifest = {
                "infrastructure_valid": False,
                "run_error": failure["error"],
                "raw_jsonl_sha256": (sha256_bytes(raw.read_bytes())
                                     if raw.exists() else None),
            }
            _atomic_json(output / "manifest.json", manifest)
            print(json.dumps(manifest, sort_keys=True), file=sys.stderr)
            return 2
        parser.error(str(exc))
    print(json.dumps(manifest, sort_keys=True))
    return 0 if manifest["infrastructure_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
