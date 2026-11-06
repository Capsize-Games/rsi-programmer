"""CPU-matched learned-ranking runner: identical CPU budgets, no SNN arm."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import torch

from scripts import benchmark_pre_eval as v1
from rsi_programmer import benchmark_v2_runner as v2runner
from rsi_programmer.benchmark_fit_work import fit_with_work
from rsi_programmer.benchmark_mlp import MLPControl
from rsi_programmer.benchmark_policy import Group
from rsi_programmer.benchmark_policy_heuristic import (
    ClassicalHeuristicPolicy,
)
from rsi_programmer.benchmark_policy_random import SeededRandomPolicy
from rsi_programmer.benchmark_recurrent import RecurrentControl
from rsi_programmer.benchmark_v2_training import build_training_groups
from rsi_programmer.cpu_budget import CpuBudget, CpuBudgetExceeded
from rsi_programmer.cpu_matched_frozen import (
    ARMS,
    CORPUS_SHA256,
    PROTOCOL_SHA256,
    ROOT,
    load_frozen_inputs,
    validate_protocol,
)
from rsi_programmer.cpu_matched_verdict import summarize
from rsi_programmer.pre_eval import pre_evaluation_search
from rsi_programmer.program_cost import CostClock


def _random_seed(seed: int, task_id: str) -> int:
    """Derive a stable per-task random stream from the seed."""
    value = f"{seed}:{task_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _fit(model: Any, groups: list[Group], epochs: int
         ) -> dict[str, Any]:
    """Fit one learned arm, charging its measured CPU cost."""
    clock = CostClock()
    steps = int(model.metadata().get("simulation_steps", 1))
    loss, work = fit_with_work(model, groups, epochs, steps)
    return {"loss": float(loss), "fit_work": work,
            "fit_cpu_seconds": clock.elapsed()["cpu_seconds"],
            "fingerprint_after_fit": model.fingerprint()}


def build_arms(seed: int, groups: list[Group], epochs: int
               ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build frozen arms; only the two learned arms pay fit CPU."""
    trained = {"trained_mlp": MLPControl(seed=seed),
               "trained_recurrent": RecurrentControl(seed=seed)}
    models: dict[str, Any] = {
        "streaming": None,
        "heuristic": ClassicalHeuristicPolicy(),
        "seeded_random": None,
        "untrained_mlp": MLPControl(seed=seed),
    }
    models.update(trained)
    fit: dict[str, Any] = {}
    for name, model in models.items():
        if name in trained:
            fit[name] = _fit(model, groups, epochs)
        else:
            fit[name] = {"loss": None, "fit_work": {},
                         "fit_cpu_seconds": 0.0,
                         "fingerprint_after_fit": (
                             None if model is None
                             else model.fingerprint())}
    return models, fit


def _policy_for(models: dict[str, Any], arm: str, seed: int,
                task_id: str) -> Any:
    """Resolve per-cell policies, seeding random streams per task."""
    if arm == "seeded_random":
        return SeededRandomPolicy(_random_seed(seed, task_id))
    return models[arm]


def run_cell(search_fn: Callable[..., dict[str, Any]], policy: Any,
             task: dict[str, Any], cell_budget: float,
             backstop: int) -> tuple[dict[str, Any], float, str]:
    """Search one cell under an identical CPU-second budget."""
    clock = CostClock()
    try:
        with CpuBudget(cell_budget):
            result = v1._search_once(policy, task, backstop,
                                     task["id"], search_fn)
        return result, clock.elapsed()["cpu_seconds"], ""
    except CpuBudgetExceeded as exc:
        result = {"solved": False, "graph": None}
        return result, clock.elapsed()["cpu_seconds"], (
            "cpu-budget-exceeded: " + str(exc))


def _atomic_json(path: Path, value: Any) -> None:
    """Write JSON atomically so partial runs never masquerade."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _git_revision() -> str:
    """Return the checked-out revision for the run manifest."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _cell_row(seed: int, arm: str, task: dict[str, Any],
              result: dict[str, Any], cpu: float, error: str,
              share: float) -> dict[str, Any]:
    """Validate one searched cell and build its evidence row."""
    validation = v2runner.validate_graph_exact(result.get("graph"),
                                               task)
    accepted = bool(result.get("solved", False)
                    and validation["train_valid"]
                    and validation["held_out_valid"]
                    and validation["exact_function_match"])
    return {"seed": seed, "arm": arm, "task_id": task["id"],
            "family": task["family"], "accepted": accepted,
            "error": error, "cpu_seconds": cpu,
            "charged_cpu_seconds": cpu + share,
            "returned_graph": result.get("graph"),
            "validation": validation}


def run_benchmark(
        output: Path,
        search_fn: Callable[..., dict[str, Any]
                             ] = pre_evaluation_search,
        protocol: dict[str, Any] | None = None,
        corpus: dict[str, Any] | None = None,
        smoke: bool = False,
        ) -> dict[str, Any]:
    """Run the frozen CPU-matched matrix and judge the kill rule."""
    if protocol is None or corpus is None:
        protocol, corpus = load_frozen_inputs()
    validate_protocol(protocol)
    v2runner.validate_corpus(corpus, protocol)
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "raw.jsonl"
    raw_path.touch()
    seeds = (protocol["training"]["seeds"][:1] if smoke
             else protocol["training"]["seeds"])
    eval_tasks = corpus["splits"]["eval"]
    tasks = eval_tasks[:2] if smoke else eval_tasks
    groups, _setup = build_training_groups(
        corpus["splits"]["train"],
        positive_cap=protocol["training"]["positive_cap"],
        negative_cap=protocol["training"]["negative_cap"],
        sampling_seed=protocol["training"]["sampling_seed"])
    cell_budget = protocol["cost"]["cell_budget_cpu_seconds"]
    backstop = protocol["evaluation"]["candidate_backstop"]
    volume = protocol["cost"]["amortization_volume_per_seed"]
    records: list[dict[str, Any]] = []
    fit_cpu: dict[str, dict[int, float]] = {arm: {} for arm in ARMS}
    fit_reports: dict[str, Any] = {}
    with raw_path.open("a", encoding="utf-8") as handle:
        for seed in seeds:
            models, fit = build_arms(
                seed, groups, protocol["training"]["epochs"])
            fit_reports[str(seed)] = fit
            for arm in ARMS:
                fit_cpu[arm][seed] = fit[arm]["fit_cpu_seconds"]
            for task in tasks:
                for arm in ARMS:
                    policy = _policy_for(models, arm, seed,
                                         task["id"])
                    result, cpu, error = run_cell(
                        search_fn, policy, task, cell_budget,
                        backstop)
                    share = fit_cpu[arm][seed] / volume
                    row = _cell_row(seed, arm, task, result, cpu,
                                    error, share)
                    handle.write(json.dumps(
                        row, sort_keys=True, allow_nan=False) + "\n")
                    handle.flush()
                    records.append(row)
    aggregate = summarize(records, fit_cpu, protocol)
    manifest = {
        "protocol_sha256": PROTOCOL_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "revision": _git_revision(),
        "smoke": smoke,
        "row_count": len(records),
        "aggregate": aggregate,
    }
    _atomic_json(output / "aggregate.json", aggregate)
    _atomic_json(output / "manifest.json", manifest)
    return manifest
