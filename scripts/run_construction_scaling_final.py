"""One-shot sealed data-scaling run: imitation vs streaming, equal CPU."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from rsi_programmer.construction_scaling_final import (
    require_scaling_budget,
    run_scaling_final,
    verify_scaling_bundle,
)

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    """Collect the frozen protocol path and the output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=(
        ROOT / "fixtures/construction_scaling_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_bundle(args: argparse.Namespace) -> dict[str, Any]:
    """Load the protocol, then verify the sealed bank it points at."""
    protocol: dict[str, Any] = json.loads(args.protocol.read_text())
    corpus = Path(protocol["corpus_path"])
    if not corpus.is_absolute():
        corpus = ROOT / corpus
    bundle = verify_scaling_bundle(corpus, args.protocol)
    require_scaling_budget(bundle["protocol"])
    return bundle


def main() -> int:
    """Verify the sealed bank, run once, and write the verdict."""
    args = _parse_args()
    bundle = _load_bundle(args)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.process_time()
    report = run_scaling_final(bundle["corpus"], bundle["protocol"])
    report["worker_cpu_seconds"] = time.process_time() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2)
                                             + "\n")
    decision = report["decision"]
    credited = {"learned_credited_mean":
                decision["learned_credited_mean"],
                "streaming_credited": decision["streaming_credited"],
                "untrained_credited_mean":
                decision["untrained_credited_mean"]}
    print(json.dumps({"verdict": "PASS" if report["verdict"] else "FAIL",
                      "credited": credited}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
