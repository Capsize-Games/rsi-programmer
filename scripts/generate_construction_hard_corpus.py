"""Create the hard construction bank at exact sampled op counts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rsi_programmer.construction_hard_corpus import (
    MAX_ATTEMPTS,
    QUOTAS,
    SEED,
    V1_CORPUS,
    generate_corpus,
)


def build_parser() -> argparse.ArgumentParser:
    """Define the hard-bank generator command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train", type=int, default=QUOTAS["train"])
    parser.add_argument("--dev", type=int, default=QUOTAS["dev"])
    parser.add_argument("--eval", type=int, default=QUOTAS["eval"])
    parser.add_argument("--v1-corpus", type=Path, default=V1_CORPUS)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    return parser


def main() -> int:
    """Generate the corpus, refusing to overwrite existing output."""
    parser = build_parser()
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing {args.output}")
    quotas = {"train": args.train, "dev": args.dev, "eval": args.eval}
    corpus = generate_corpus(seed=args.seed, quotas=quotas,
                             v1_path=args.v1_corpus,
                             max_attempts=args.max_attempts)
    data = (json.dumps(corpus, sort_keys=True, separators=(",", ":"))
            + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"sha256": hashlib.sha256(data).hexdigest(),
                      "bytes": len(data), "tasks": {
                          split: len(tasks)
                          for split, tasks in corpus["splits"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
