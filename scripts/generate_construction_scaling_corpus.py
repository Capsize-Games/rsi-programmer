"""Create the shape-broad scaling bank with full exclusions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from rsi_programmer.construction_broad_corpus import (
    QUOTAS,
    SEED,
    generate_corpus,
)
from rsi_programmer.construction_hard_corpus import load_v1_identities

HARD_CORPUS = "fixtures/construction_hard_corpus_v1.json"
V1_CORPUS = "fixtures/construction_corpus_v1.json"
D1_DIAGNOSE = "artifacts/construction-d1-diagnostics-20261106/diagnose.py"
D2_DIAGNOSE = "artifacts/construction-d2-diagnostics-20261106/diagnose.py"


def _split_polys(tasks: list[dict[str, Any]]) -> set[Any]:
    """Collect polynomial identities from one task list."""
    return {tuple(tuple(term) for term in task["generator"]["polynomial"])
            for task in tasks}


def _fixture_polys(path: Path) -> set[Any]:
    """Collect polynomial identities from a frozen corpus file."""
    corpus = json.loads(path.read_text())
    return {poly for split in corpus["splits"].values()
            for poly in _split_polys(split)}


def _module_polys(name: str, path: str) -> set[Any]:
    """Import an artifact module and collect its dev identities."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    found: set[Any] = set()
    for item in module.build_sets():
        if isinstance(item, list):
            found |= _split_polys(item)
    return found


def build_exclusions(root: Path) -> tuple[set[Any], dict[str, int]]:
    """Assemble v1 + sealed-hard + D1/D2 dev identities exactly."""
    v1 = load_v1_identities(root / V1_CORPUS)
    hard = _fixture_polys(root / HARD_CORPUS)
    d1 = _module_polys("d1_diagnose", str(root / D1_DIAGNOSE))
    d2 = _module_polys("d2_diagnose", str(root / D2_DIAGNOSE))
    counts = {"v1": len(v1), "hard": len(hard),
              "d1_dev": len(d1), "d2_dev": len(d2)}
    return v1 | hard | d1 | d2, counts


def build_parser() -> argparse.ArgumentParser:
    """Define the scaling-bank generator command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(
        "fixtures/construction_scaling_corpus_v1.json"))
    parser.add_argument("--train-per", type=int,
                        default=QUOTAS["train"] // 3)
    parser.add_argument("--eval-per", type=int,
                        default=QUOTAS["eval"] // 3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def _write_corpus(out: Path, corpus: dict[str, Any]) -> dict[str, Any]:
    """Write the corpus file and report its digest and sizes."""
    data = (json.dumps(corpus, sort_keys=True, separators=(",", ":"))
            + "\n").encode()
    with out.open("xb") as stream:
        stream.write(data)
    return {"sha256": hashlib.sha256(data).hexdigest(),
            "train": len(corpus["splits"]["train"]),
            "eval": len(corpus["splits"]["eval"]),
            "pools": corpus["available_functions_by_stratum"]}


def main() -> int:
    """Build the exclusion set, then generate the scaling bank."""
    parser = build_parser()
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    excluded, counts = build_exclusions(root)
    print(json.dumps({"excluded_identities": len(excluded), **counts}))
    if args.verify_only:
        return 0
    if args.out.exists():
        parser.error(f"refusing to overwrite existing {args.out}")
    quotas = {"train": 3 * args.train_per, "eval": 3 * args.eval_per}
    corpus = generate_corpus(seed=args.seed, quotas=quotas,
                             excluded=excluded)
    print(json.dumps(_write_corpus(args.out, corpus)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
