#!/usr/bin/env python3
"""Generate the CPU-matched benchmark corpus with fresh seeds.

Uses the exact v2 enumerator code path with fresh family, split and
input seeds, then renames IDs and verifies exact-polynomial
disjointness against the v1 and v2 banks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "rsi_programmer" / "benchmark_v2_corpus.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_v2_corpus_standalone", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CORPUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORPUS)

BASE_FAMILY_SEEDS = {"repeated_subtree": 83071, "balanced": 83072,
                     "skewed": 83073}
BASE_SPLIT_SEED = 83074
BASE_INPUT_SEED = 83075
CORPUS_ID = "benchmark-cpu-matched-v1"
V1_PATH = ROOT / "fixtures" / "benchmark_pre_eval_corpus_v1.json"
V2_PATH = ROOT / "fixtures" / "benchmark_pre_eval_corpus_v2.json"


def _polys(path: Path) -> set[Any]:
    """Collect canonical polynomials from a frozen bank fixture."""
    corpus = json.loads(path.read_text(encoding="utf-8"))
    found: set[Any] = set()
    for tasks in corpus["splits"].values():
        for task in tasks:
            poly = CORPUS.polynomial(task["generator"]["program"])
            found.add(poly)
    return found


def _rename(corpus: dict[str, Any]) -> dict[str, Any]:
    """Give the new bank its own corpus and task IDs."""
    corpus["corpus_id"] = CORPUS_ID
    for split in ("train", "eval"):
        for index, task in enumerate(corpus["splits"][split]):
            task["id"] = f"cm_{split}_{index + 1:02d}"
    return corpus


def main() -> None:
    """Generate, disjointness-check and write the new corpus."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "fixtures"
                        / "benchmark_cpu_matched_corpus_v1.json")
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    family_seeds = {key: value + args.seed_offset
                    for key, value in BASE_FAMILY_SEEDS.items()}
    split_seed = BASE_SPLIT_SEED + args.seed_offset
    input_seed = BASE_INPUT_SEED + args.seed_offset
    CORPUS.FAMILY_SEEDS.update(family_seeds)
    CORPUS.SPLIT_SEED = split_seed
    CORPUS.INPUT_SEED = input_seed
    old = _polys(V1_PATH) | _polys(V2_PATH)
    pools, rejects = CORPUS.candidate_pools()
    removed = 0
    for family in pools:
        kept = [node for node in pools[family]
                if CORPUS.polynomial(node) not in old]
        removed += len(pools[family]) - len(kept)
        pools[family] = kept
    rejects = dict(rejects)
    rejects["prior_bank_functions"] = removed
    CORPUS.candidate_pools = lambda: (pools, rejects)
    corpus = _rename(CORPUS.generate_corpus())
    new = {CORPUS.polynomial(task["generator"]["program"])
           for tasks in corpus["splits"].values() for task in tasks}
    overlap = old & new
    corpus["derivation"] = {
        "code_path": "same enumerator and selection as "
                     "benchmark-pre-eval-v2; the only difference is "
                     "removing exact polynomials already banked in "
                     "v1/v2 before selection",
        "seed_offset": args.seed_offset,
        "prior_bank_functions_removed": removed,
        "family_seeds": dict(family_seeds),
        "split_seed": split_seed,
        "input_seed": input_seed,
        "checked_banks": [str(V1_PATH.name), str(V2_PATH.name)],
        "old_function_count": len(old),
        "new_function_count": len(new),
        "exact_polynomial_overlap": len(overlap),
    }
    if overlap:
        raise SystemExit(f"frozen bank overlaps old banks: {len(overlap)}")
    args.output.write_text(CORPUS.canonical_json(corpus),
                           encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output} sha256={digest}")


if __name__ == "__main__":
    main()
