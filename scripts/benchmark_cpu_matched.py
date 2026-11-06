#!/usr/bin/env python3
"""Run the frozen CPU-matched learned-ranking benchmark (Pursue 2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi_programmer.benchmark_cpu_matched import run_benchmark


def main(argv: list[str] | None = None) -> int:
    """Run the full matrix, or a two-task smoke subset with --smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for raw rows and reports")
    parser.add_argument("--smoke", action="store_true",
                        help="one seed and two tasks only; not evidence")
    args = parser.parse_args(argv)
    manifest = run_benchmark(args.output.resolve(), smoke=args.smoke)
    print(json.dumps(manifest["aggregate"]["verdict"], indent=2,
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
