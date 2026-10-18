"""Run the fixed online graph-learning pilot; use python -m scripts...."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsi_programmer.online_benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = run_benchmark(args.output)
    print(json.dumps(summary["gates"], sort_keys=True))


if __name__ == "__main__":
    main()
