"""One-shot sealed hard-bank run: feedback vs baselines, equal CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from rsi_programmer.construction_final import verify_generator_sources
from rsi_programmer.construction_hard_final import (
    EVAL_SPLIT,
    TRAIN_SPLIT,
    require_hard_budget,
    run_hard_seed,
    summarize_hard,
    verify_hard_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("streaming", "untrained", "imitation", "feedback")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-receipt", type=Path, required=True,
                        help="Dev report pinning the frozen recipe.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=(
        ROOT / "fixtures/construction_hard_corpus_v1.json"))
    parser.add_argument("--protocol", type=Path, default=(
        ROOT / "fixtures/construction_hard_v1.json"))
    return parser.parse_args()


def _load_bundle(args: argparse.Namespace) -> dict[str, Any]:
    if not args.frozen_receipt.is_file():
        raise ValueError("missing frozen recipe receipt")
    bundle = verify_hard_bundle(args.corpus, args.protocol)
    bundle["generator_sources"] = verify_generator_sources(
        ROOT, bundle["protocol"])
    bundle["receipt_sha256"] = hashlib.sha256(
        args.frozen_receipt.read_bytes()).hexdigest()
    require_hard_budget(bundle["protocol"])
    return bundle


def _split_tasks(bundle: dict[str, Any]) -> tuple[list[dict[str, Any]],
                                                  list[dict[str, Any]]]:
    """Return train/eval rows; eval rows never feed training here."""
    splits = bundle["corpus"]["splits"]
    return list(splits[TRAIN_SPLIT]), list(splits[EVAL_SPLIT])


def _collect(seeds: list[dict[str, Any]]
             ) -> tuple[dict[str, dict[int, list[dict[str, Any]]]],
                        dict[str, dict[int, dict[str, float]]]]:
    """Group per-seed rows and fit/rollout charges by arm."""
    arms_rows: dict[str, dict[int, list[dict[str, Any]]]] = {
        arm: {} for arm in ARMS}
    costs: dict[str, dict[int, dict[str, float]]] = {
        arm: {} for arm in ARMS}
    for row in seeds:
        for arm in ARMS:
            arms_rows[arm][row["seed"]] = row["arms"][arm]["rows"]
            costs[arm][row["seed"]] = {
                "fit_cpu": row["fits"][arm]["fit_cpu"],
                "rollout_cpu": row["fits"][arm]["rollout_cpu"]}
    return arms_rows, costs


def _assemble(bundle: dict[str, Any], args: argparse.Namespace,
              seeds: list[dict[str, Any]],
              started: float) -> dict[str, Any]:
    """Combine per-seed results into the one-shot verdict report."""
    protocol = bundle["protocol"]
    arms_rows, costs = _collect(seeds)
    summary = summarize_hard(
        arms_rows, costs, protocol["decision_rules"]["kill_criterion"])
    return {"scope": ("One-shot sealed hard-bank run. Eval rows "
                      "never train; rollout generation is charged."),
            "verdict": summary["verdict"]["passed"],
            "decision": summary["verdict"],
            "means": summary["means"], "strata": summary["strata"],
            "equal_cpu_vs_streaming": summary["equal_cpu_vs_streaming"],
            "seeds": seeds,
            "corpus_sha256": protocol["corpus_sha256"],
            "generator_sources": bundle["generator_sources"],
            "receipt_sha256": bundle["receipt_sha256"],
            "worker_cpu_seconds": time.process_time() - started}


def main() -> int:
    args = _parse_args()
    bundle = _load_bundle(args)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    protocol = bundle["protocol"]
    train, final = _split_tasks(bundle)
    started = time.process_time()
    seeds = [run_hard_seed(train, final, protocol, seed)
             for seed in protocol["evaluation"]["model_seeds"]]
    report = _assemble(bundle, args, seeds, started)
    (args.output / "report.json").write_text(json.dumps(report, indent=2)
                                             + "\n")
    print(json.dumps({"verdict": "PASS" if report["verdict"] else "FAIL",
                      "means": report["means"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
