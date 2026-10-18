"""Run the frozen first fan-vibration sensing comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsi_programmer.sensing_benchmark import run_benchmark


def main() -> None:
    """Save a new measurement directory without replacing prior evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_benchmark(args.output)
    print(json.dumps({key: result[key] for key in (
        "git_head", "protocol_sha256", "stream_rows", "windows",
        "full_process_cpu_seconds", "wall_seconds")}, sort_keys=True))


if __name__ == "__main__":
    main()
