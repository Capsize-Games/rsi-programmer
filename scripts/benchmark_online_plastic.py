"""Run the frozen online plastic-encoder comparison."""
from __future__ import annotations

import argparse
from pathlib import Path

from rsi_programmer.online_plastic_benchmark import run


def main() -> int:
    """Parse the output directory and start the complete fixed matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path,
        help="new result directory; existing directories are rejected",
    )
    arguments = parser.parse_args()
    summary = run(arguments.output)
    print(summary["all_order_gates"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
