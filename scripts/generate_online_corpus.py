#!/usr/bin/env python3
"""Write the frozen synthetic online-graph corpus."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi_programmer.online_corpus import CORPUS_SEED
from rsi_programmer.online_corpus import write_online_corpus


def main() -> int:
    """Generate a JSON corpus and print its content digest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("fixtures/online_graph_v1.json"))
    parser.add_argument("--seed", type=int, default=CORPUS_SEED)
    args = parser.parse_args()
    digest = write_online_corpus(args.output, args.seed)
    print(f"wrote {args.output} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
