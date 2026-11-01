"""Create the construction pilot's new frozen corpus without fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rsi_programmer.construction_corpus import generate_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = generate_corpus()
    data = (json.dumps(corpus, sort_keys=True, separators=(",", ":"))
            + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"sha256": hashlib.sha256(data).hexdigest(),
                      "bytes": len(data), "tasks": {
                          split: len(tasks) for split, tasks
                          in corpus["splits"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
