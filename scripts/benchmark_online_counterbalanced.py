"""Run the frozen counterbalanced graph-learning diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsi_programmer.online_counterbalanced import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps(result["diagnosis"], sort_keys=True))


if __name__ == "__main__":
    main()
