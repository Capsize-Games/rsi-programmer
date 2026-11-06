"""One-shot sealed construction-v1 final run; never a tuning pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import torch

from rsi_programmer.construction_final import (
    FINAL_SPLIT,
    cost_gate,
    evaluate_final,
    learning_gate,
    verify_frozen_corpus,
    verify_generator_sources,
)
from rsi_programmer.construction_policy import ConstructionPolicy
from rsi_programmer.construction_training import fit_constructor

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-receipt", type=Path, required=True,
                        help="Dev report pinning the frozen recipe.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--corpus", type=Path, default=(
        ROOT / "fixtures/construction_corpus_v1.json"))
    parser.add_argument("--protocol", type=Path, default=(
        ROOT / "fixtures/construction_v1.json"))
    return parser.parse_args()


def _load_bundle(args: argparse.Namespace) -> dict[str, Any]:
    if not args.frozen_receipt.is_file():
        raise ValueError("missing frozen recipe receipt")
    if not args.epochs >= 1:
        raise ValueError("epochs must be positive")
    bundle = verify_frozen_corpus(args.corpus, args.protocol)
    bundle["generator_sources"] = verify_generator_sources(
        ROOT, bundle["protocol"])
    bundle["receipt_sha256"] = hashlib.sha256(
        args.frozen_receipt.read_bytes()).hexdigest()
    return bundle


def _sweep(tasks: list[dict[str, Any]], method: str,
           model: ConstructionPolicy | None,
           caps: list[int]) -> list[dict[str, Any]]:
    """Evaluate rising caps, stopping at first full coverage."""
    sweep: list[dict[str, Any]] = []
    for cap in caps:
        summary, rows = evaluate_final(tasks, method, model, cap)
        sweep.append({"cap": cap, **summary, "rows": rows})
        print(json.dumps({"method": method, "cap": cap, **summary}),
              flush=True)
        if summary["accepted"] == summary["tasks"]:
            break
    return sweep


def _run_seed(train: list[dict[str, Any]], final: list[dict[str, Any]],
              caps: list[int], seed: int, epochs: int) -> dict[str, Any]:
    """Fit once on train, then infer on final only at the fixed caps."""
    model = ConstructionPolicy(seed=seed, backend="ann")
    untrained = ConstructionPolicy(seed=seed, backend="ann")
    fit = fit_constructor(model, train, epochs)
    print(json.dumps({"seed": seed, "fit": fit}), flush=True)
    primary, _ = evaluate_final(final, "beam", model, caps[1],
                                beam_seed=seed)
    base, _ = evaluate_final(final, "beam", untrained, caps[1],
                             beam_seed=seed)
    chance, _ = evaluate_final(final, "beam", None, caps[1],
                               beam_seed=seed)
    trained = _sweep(final, "beam", model, caps)
    return {"seed": seed, "fit": fit,
            "primary_cap": {"cap": caps[1], "trained": primary,
                            "untrained": base, "random": chance},
            "trained_sweep": trained}


def _rates(seeds: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    """Acceptance rates per seed at the primary cap."""
    return {row["seed"]: {
        arm: row["primary_cap"][arm]["accepted"]
        / row["primary_cap"][arm]["tasks"]
        for arm in ("trained", "untrained", "random")} for row in seeds}


def _costs(seeds: list[dict[str, Any]],
           stream: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Cost verdict per seed against the shared streaming sweep."""
    tasks = len(seeds[0]["trained_sweep"][0]["rows"])
    return {row["seed"]: cost_gate(
        row["trained_sweep"], stream,
        row["fit"]["complete_cpu_seconds"], tasks) for row in seeds}


def _assemble(bundle: dict[str, Any], protocol: dict[str, Any],
              args: argparse.Namespace, seeds: list[dict[str, Any]],
              stream: list[dict[str, Any]], started: float) -> dict[str, Any]:
    """Combine per-seed results into the one-shot verdict report."""
    learning = learning_gate(_rates(seeds))
    costs = _costs(seeds, stream)
    return {"scope": "One-shot sealed final. No tuning; one run only.",
            "verdict": learning["passed"] and all(
                cost["passed"] for cost in costs.values()),
            "learning_gate": learning, "cost_gates": costs,
            "model_seeds": seeds, "streaming_sweep": stream,
            "corpus_sha256": protocol["corpus_sha256"],
            "generator_sources": bundle["generator_sources"],
            "receipt_sha256": bundle["receipt_sha256"],
            "epochs": args.epochs,
            "worker_cpu_seconds": time.process_time() - started,
            "peak_worker_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss,
            "runtime": {"python": sys.version.split()[0]}}


def _tasks(bundle: dict[str, Any],
           args: argparse.Namespace) -> tuple[list[dict[str, Any]],
                                             list[dict[str, Any]]]:
    """Split train/final tasks, guarding the sealed final size."""
    final = [task for task in bundle["corpus"]["splits"][FINAL_SPLIT]]
    if len(final) != 244 and args.corpus == (
            ROOT / "fixtures/construction_corpus_v1.json"):
        raise ValueError("sealed final split changed size")
    return [task for task in bundle["corpus"]["splits"]["train"]], final


def main() -> int:
    args = _parse_args()
    bundle = _load_bundle(args)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    protocol = bundle["protocol"]
    train, final = _tasks(bundle, args)
    caps = protocol["evaluation"]["execution_caps"]
    started = time.process_time()
    stream = _sweep(final, "streaming", None, caps)
    seeds = [_run_seed(train, final, caps, seed, args.epochs)
             for seed in protocol["evaluation"]["model_seeds"]]
    report = _assemble(bundle, protocol, args, seeds, stream, started)
    (args.output / "report.json").write_text(json.dumps(report, indent=2)
                                             + "\n")
    print(json.dumps({"verdict": "PASS" if report["verdict"] else "FAIL",
                      "learning_passed":
                          report["learning_gate"]["passed"]}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
