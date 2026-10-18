#!/usr/bin/env python3
"""Run the frozen paired total-work benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi_programmer.benchmark_total_work import run_total_work
from rsi_programmer.benchmark_v2_runner import _atomic_json, sha256_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path,
        default=ROOT / "fixtures/benchmark_total_work_v1.json",
        help="frozen total-work protocol JSON",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/benchmark-total-work-v1",
        help="new output directory for raw rows and summaries",
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output_existed = output.exists()
    try:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        manifest = run_total_work(output, protocol)
    except Exception as exc:
        failure: dict[str, Any] = {
            "row_type": "infrastructure_failure",
            "error": type(exc).__name__ + ": " + str(exc),
        }
        raw_path = output / "raw.jsonl"
        if not output_existed:
            output.mkdir(parents=True, exist_ok=True)
            if raw_path.exists():
                with raw_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, sort_keys=True) + "\n")
            manifest = {
                "infrastructure_valid": False,
                "run_error": failure["error"],
                "raw_jsonl_sha256": (
                    sha256_bytes(raw_path.read_bytes())
                    if raw_path.exists() else None),
                "training_checkpoint_sha256": (
                    sha256_bytes((output / "training.json").read_bytes())
                    if (output / "training.json").exists() else None),
            }
            _atomic_json(output / "manifest.json", manifest)
        else:
            manifest = {"infrastructure_valid": False,
                        "run_error": failure["error"]}
        print(json.dumps(manifest, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0 if manifest["infrastructure_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
