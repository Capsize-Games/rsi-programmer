"""Sealed data-scaling final: imitation at volume vs streaming."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from rsi_programmer.construction_beam import beam_construct
from rsi_programmer.construction_check import check_graph
from rsi_programmer.construction_policy import ConstructionPolicy
from rsi_programmer.construction_scaling_decision import (
    credited_solves,
    decide_scaling,
    summarize_scaling,
)
from rsi_programmer.construction_state import Rows
from rsi_programmer.construction_training import fit_constructor
from rsi_programmer.cpu_budget import CpuBudget, CpuBudgetExceeded
from rsi_programmer.pre_eval import pre_evaluation_search

__all__ = ["REQUIRED_KEYS", "credited_solves", "decide_scaling",
           "evaluate_arm", "require_scaling_budget",
           "run_scaling_final", "run_scaling_seed",
           "verify_scaling_bundle"]

EVAL_SPLIT, TRAIN_SPLIT = "eval", "train"
REQUIRED_KEYS = ("model_seeds", "per_task_cpu_seconds", "beam_width",
                 "bound", "budget", "imitation_epochs",
                 "declared_volume")
Policy = ConstructionPolicy | None
SCOPE = "Sealed data-scaling run; fit CPU charged before eval."


def _check_sources(root: Path, protocol: dict[str, Any]
                   ) -> dict[str, str]:
    """Refuse null or changed pinned generator sources."""
    expected = protocol["source_provenance"]["generator_sha256"]
    actual: dict[str, str] = {}
    for name, digest in expected.items():
        if digest is None or not (root / name).is_file():
            raise ValueError(f"bad generator source: {name}")
        seen = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if seen != digest:
            raise ValueError(f"pinned generator source changed: {name}")
        actual[name] = seen
    return actual


def _check_shapes(corpus: dict[str, Any], protocol: dict[str, Any]
                  ) -> None:
    """Match declared volume and quotas to the corpus split sizes."""
    evaluation = protocol.get("evaluation", {})
    for key in REQUIRED_KEYS:
        if key not in evaluation:
            raise ValueError(f"scaling protocol lacks {key}")
    splits = corpus.get("splits", {})
    quotas = protocol.get("quotas", {})
    for split, quota in ((TRAIN_SPLIT, "train"), (EVAL_SPLIT, "eval")):
        if len(splits.get(split, [])) != quotas.get(quota):
            raise ValueError("scaling quotas differ from corpus splits")
    if evaluation["declared_volume"] != quotas.get("eval"):
        raise ValueError("declared volume differs from eval split")


def verify_scaling_bundle(corpus_path: Path, protocol_path: Path
                          ) -> dict[str, Any]:
    """Hash-check the sealed scaling bank; require the frozen keys."""
    protocol: dict[str, Any] = json.loads(Path(protocol_path).read_text())
    evaluation = protocol.get("evaluation", {})
    digest = protocol.get("corpus_sha256")
    if not isinstance(digest, str):
        raise ValueError("scaling corpus hash must be frozen, not null")
    raw = Path(corpus_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("frozen scaling corpus changed")
    if evaluation.get("frozen_at") is None:
        raise ValueError("scaling evaluation must carry frozen_at")
    corpus = json.loads(raw)
    _check_shapes(corpus, protocol)
    sources = _check_sources(Path(__file__).resolve().parents[1],
                             protocol)
    return {"protocol": protocol, "corpus": corpus,
            "generator_sources": sources}


def require_scaling_budget(protocol: dict[str, Any]) -> float:
    """Refuse a null/unset per-task CPU budget; it must freeze first."""
    budget = protocol["evaluation"]["per_task_cpu_seconds"]
    if budget is None or not isinstance(budget, (int, float)):
        raise ValueError("per-task CPU budget must be frozen, not null")
    if not budget > 0.0:
        raise ValueError("per-task CPU budget must be positive")
    return float(budget)


def _search(arm: str, policy: Policy, observed: Rows, bound: int,
            budget: int, width: int, seed: int) -> dict[str, Any]:
    """Run one search; the streaming CPU cap is applied by the caller."""
    if arm == "streaming":
        pairs = tuple((x, y) for x, y, _ in observed)
        wants = tuple(want for _, _, want in observed)
        out = pre_evaluation_search(None, pairs, wants, bound, budget)
        return {"graph": out["graph"], "executions":
                out["signature_evaluations"], "error":
                None if out["graph"] is not None else out["verdict"]}
    out = beam_construct(policy, observed, bound=bound, budget=budget,
                         width=width, seed=seed)
    return {"graph": out["graph"], "executions": out["executions"],
            "error": out.get("error")}


def _capped_search(arm: str, policy: Policy, observed: Rows,
                   cpu: float, backstop: int, bound: int,
                   width: int, seed: int) -> dict[str, Any]:
    """Search one task; only streaming runs inside the CPU cap."""
    if arm != "streaming":
        return _search(arm, policy, observed, bound, backstop,
                       width, seed)
    try:
        with CpuBudget(cpu):
            return _search(arm, policy, observed, bound, backstop,
                           width, seed)
    except CpuBudgetExceeded:
        return {"graph": None, "executions": 0,
                "error": "cpu budget"}


def _checked_row(task: dict[str, Any], arm: str, policy: Policy,
                 cpu: float, backstop: int, bound: int, width: int,
                 seed: int) -> dict[str, Any]:
    """Time one task's search plus checking; charge the sum."""
    observed = tuple(tuple(row) for row in task["observed"])
    held = tuple(tuple(row) for row in task["held"])
    poly = tuple(tuple(t) for t in task["generator"]["polynomial"])
    started = time.process_time()
    out = _capped_search(arm, policy, observed, cpu, backstop,
                         bound, width, seed)
    check = check_graph(out["graph"], observed, held, poly, bound)
    return {"id": task["id"], "accepted": check["accepted"],
            "cpu_seconds": time.process_time() - started,
            "executions": out["executions"], "error": out["error"],
            "stratum": task["generator"].get("minimal_tree_ops")}


def evaluate_arm(tasks: list[dict[str, Any]], arm: str,
                 policy: Policy, cpu: float, backstop: int,
                 bound: int, width: int,
                 seed: int) -> tuple[dict[str, Any],
                                      list[dict[str, Any]]]:
    """Score one arm and seed; learned weights must not move."""
    if not tasks or any(t.get("split") != EVAL_SPLIT for t in tasks):
        raise ValueError("scaling inference needs eval-split tasks")
    before = policy.fingerprint() if policy is not None else None
    rows = [_checked_row(task, arm, policy, cpu, backstop, bound,
                         width, seed) for task in tasks]
    if policy is not None and policy.fingerprint() != before:
        raise ValueError("evaluation changed neural weights")
    summary = {"arm": arm, "seed": seed, "accepted": sum(
        1 for row in rows if row["accepted"]), "tasks": len(rows),
               "cpu_seconds": sum(row["cpu_seconds"] for row in rows),
               "executions": sum(row["executions"] for row in rows)}
    return summary, rows


def run_scaling_seed(train: list[dict[str, Any]],
                     tasks: list[dict[str, Any]],
                     protocol: dict[str, Any],
                     seed: int) -> dict[str, Any]:
    """Fit imitation fresh, then score learned/untrained at one seed."""
    evaluation = protocol["evaluation"]
    learned = ConstructionPolicy(seed=seed, backend="ann")
    fit = fit_constructor(learned, train,
                          evaluation["imitation_epochs"])
    plain = ConstructionPolicy(seed=seed, backend="ann")
    cpu = require_scaling_budget(protocol)
    scored = {}
    for arm, policy in (("learned", learned), ("untrained", plain)):
        summary, rows = evaluate_arm(
            tasks, arm, policy, cpu, evaluation["budget"],
            evaluation["bound"], evaluation["beam_width"], seed)
        scored[arm] = {"summary": summary, "rows": rows}
    return {"seed": seed, "arms": scored, "fit_cpu": {
        "learned": fit["complete_cpu_seconds"], "untrained": 0.0},
            "fits": {"learned": fit, "untrained": None}}


def _run_arms(train: list[dict[str, Any]],
              final: list[dict[str, Any]],
              protocol: dict[str, Any]) -> tuple[Any, ...]:
    """Fit and score every seed; streaming runs once outside seeds."""
    evaluation = protocol["evaluation"]
    seeds = [run_scaling_seed(train, final, protocol, seed)
             for seed in evaluation["model_seeds"]]
    cpu = require_scaling_budget(protocol)
    summary, rows = evaluate_arm(
        final, "streaming", None, cpu, evaluation["budget"],
        evaluation["bound"], evaluation["beam_width"], 0)
    report = summarize_scaling(
        {r["seed"]: r["arms"]["learned"]["rows"] for r in seeds},
        {r["seed"]: r["arms"]["untrained"]["rows"] for r in seeds},
        rows, {r["seed"]: r["fit_cpu"]["learned"] for r in seeds},
        protocol["decision_rules"]["kill_criterion"])
    return seeds, summary, rows, report


def run_scaling_final(corpus: dict[str, Any],
                      protocol: dict[str, Any]) -> dict[str, Any]:
    """Score every arm at the declared volume; render the verdict."""
    started = time.process_time()
    train = list(corpus["splits"][TRAIN_SPLIT])
    final = list(corpus["splits"][EVAL_SPLIT])
    seeds, summary, rows, report = _run_arms(train, final, protocol)
    return {"scope": SCOPE, "verdict": report["decision"]["passed"],
            "decision": report["decision"], "means": report["means"],
            "strata": report["strata"], "seeds": seeds,
            "streaming": {"summary": summary, "rows": rows},
            "corpus_sha256": protocol["corpus_sha256"],
            "generator_sources": protocol["source_provenance"][
                "generator_sha256"],
            "worker_cpu_seconds": time.process_time() - started}
