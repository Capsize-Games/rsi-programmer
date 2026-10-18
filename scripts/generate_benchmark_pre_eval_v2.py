#!/usr/bin/env python3
"""Generate the frozen-before-measurement v2 benchmark corpus."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_MODULE_PATH = (Path(__file__).resolve().parents[1] /
                "rsi_programmer" / "benchmark_v2_corpus.py")
_SPEC = importlib.util.spec_from_file_location(
    "benchmark_v2_corpus_standalone", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("fixtures/benchmark_pre_eval_corpus_v2.json"),
        help="destination JSON path (default: v2 fixture)",
    )
    args = parser.parse_args()
    corpus = _MODULE.generate_corpus()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_MODULE.canonical_json(corpus), encoding="utf-8")


if __name__ == "__main__":
    main()
