"""Run the frozen observed-example context comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsi_programmer.online_context_benchmark import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps(result["all_order_gates"], sort_keys=True))


if __name__ == "__main__":
    main()
