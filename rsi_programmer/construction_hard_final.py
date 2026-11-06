"""Sealed hard-bank final: feedback training against baselines.

Compares `streaming`, `untrained`, `imitation`, and `feedback` arms
on the sealed hard eval split at an identical per-task CPU budget.
Equal-total-CPU truncation reuses `compare_pair` on per-seed
`(accepted, inference_cpu + amortized_fit_share)` sequences. The
frozen W3 loop is consumed through `_feedback_fit` only, which
expects W3 return keys `rounds`, `total_rollout_cpu_seconds`,
`total_fit_cpu_seconds`, and `stopped_zero_acceptance`.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rsi_programmer.construction_beam import beam_construct
from rsi_programmer.construction_check import check_graph
from rsi_programmer.construction_final import verify_frozen_corpus
from rsi_programmer.construction_policy import ConstructionPolicy
from rsi_programmer.construction_state import Rows
from rsi_programmer.construction_training import fit_constructor
from rsi_programmer.cpu_budget import CpuBudget, CpuBudgetExceeded
from rsi_programmer.cpu_matched_verdict import compare_pair
from rsi_programmer.pre_eval import pre_evaluation_search

EVAL_SPLIT = "eval"
TRAIN_SPLIT = "train"
BACKSTOP = 200000
REQUIRED_KEYS = ("model_seeds", "per_task_cpu_seconds", "beam_width",
                 "bound", "imitation_epochs", "feedback")
FEEDBACK_KEYS = ("rounds", "epochs_per_round", "rollout_budget")


def _feedback_fit(policy: ConstructionPolicy,
                  tasks: list[dict[str, Any]],
                  feedback: dict[str, Any],
                  bound: int, width: int) -> dict[str, Any]:
    """Run the frozen W3 loop; the sole import site for W3 code."""
    from rsi_programmer.construction_feedback import fit_with_feedback
    return fit_with_feedback(policy, tasks, feedback["rounds"],
                             feedback["epochs_per_round"],
                             feedback["rollout_budget"], bound,
                             width=width)


def require_hard_budget(protocol: dict[str, Any]) -> float:
    """Refuse a null/unset per-task CPU budget; it must freeze first."""
    budget = protocol["evaluation"]["per_task_cpu_seconds"]
    if budget is None or not isinstance(budget, (int, float)):
        raise ValueError("per-task CPU budget must be frozen, not null")
    if not budget > 0.0:
        raise ValueError("per-task CPU budget must be positive")
    return float(budget)


def verify_hard_bundle(corpus_path: Path,
                       protocol_path: Path) -> dict[str, Any]:
    """Hash-check the hard bank and require the frozen W1/W4 keys."""
    bundle = verify_frozen_corpus(corpus_path, protocol_path)
    evaluation = bundle["protocol"]["evaluation"]
    checks = ((evaluation, REQUIRED_KEYS),
              (evaluation["feedback"], FEEDBACK_KEYS))
    for mapping, keys in checks:
        for key in keys:
            if key not in mapping:
                raise ValueError(f"hard protocol lacks {key}")
    rules = bundle["protocol"].get("decision_rules", {})
    if "kill_criterion" not in rules:
        raise ValueError("hard protocol lacks kill_criterion")
    return bundle


def _search(arm: str, policy: ConstructionPolicy | None,
            observed: Rows, bound: int, width: int,
            seed: int) -> dict[str, Any]:
    """Run one budgeted search; the CPU cap is applied by the caller."""
    if arm == "streaming":
        rows = tuple((x, y) for x, y, _ in observed)
        values = tuple(want for _, _, want in observed)
        out = pre_evaluation_search(None, rows, values, bound, BACKSTOP)
        found = out["graph"] is not None
        tried = out["signature_evaluations"]
        return {"graph": out["graph"], "executions": tried,
                "error": None if found else out["verdict"]}
    out = beam_construct(policy, observed, bound=bound,
                         budget=BACKSTOP, width=width, seed=seed)
    return {"graph": out["graph"], "executions": out["executions"],
            "error": out.get("error")}


def _checked_row(task: dict[str, Any], arm: str,
                 policy: ConstructionPolicy | None, budget: float,
                 bound: int, width: int, seed: int) -> dict[str, Any]:
    """Run one task inside the CPU budget, then check it cold."""
    observed = tuple(tuple(row) for row in task["observed"])
    held = tuple(tuple(row) for row in task["held"])
    poly = tuple(tuple(t) for t in task["generator"]["polynomial"])
    started = time.process_time()
    try:
        with CpuBudget(budget):
            out = _search(arm, policy, observed, bound, width, seed)
    except CpuBudgetExceeded:
        out = {"graph": None, "executions": 0, "error": "cpu budget"}
    check = check_graph(out["graph"], observed, held, poly, bound)
    return {"id": task["id"], "accepted": check["accepted"],
            "cpu_seconds": time.process_time() - started,
            "executions": out["executions"], "error": out["error"],
            "stratum": task["generator"].get("minimal_tree_ops")}


def evaluate_arm(tasks: list[dict[str, Any]], arm: str,
                 policy: ConstructionPolicy | None, budget: float,
                 bound: int, width: int, seed: int
                 ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score one arm/seed; learned weights must not move during eval."""
    if not tasks or any(task.get("split") != EVAL_SPLIT
                        for task in tasks):
        raise ValueError("hard inference needs eval-split tasks only")
    before = policy.fingerprint() if policy is not None else None
    rows = [_checked_row(task, arm, policy, budget, bound, width,
                         seed) for task in tasks]
    if policy is not None and policy.fingerprint() != before:
        raise ValueError("evaluation changed neural weights")
    summary = {"arm": arm, "seed": seed, "accepted": sum(
        1 for row in rows if row["accepted"]), "tasks": len(rows),
               "cpu_seconds": sum(row["cpu_seconds"] for row in rows),
               "executions": sum(row["executions"] for row in rows)}
    return summary, rows


def _charged_cells(rows: list[dict[str, Any]], extra_cpu: float,
                   volume: int) -> list[tuple[bool, float]]:
    """Pair outcomes with inference CPU plus the amortized fit share."""
    if volume < 1:
        raise ValueError("amortization volume must be positive")
    share = extra_cpu / volume
    return [(bool(row["accepted"]), row["cpu_seconds"] + share)
            for row in rows]


def _charged_all(arms_rows: dict[str, dict[int, list[dict[str, Any]]]],
                 costs: dict[str, dict[int, dict[str, float]]],
                 arm: str, volume: int) -> list[list[tuple[bool, float]]]:
    """Charge each inference row its amortized fit/rollout share."""
    return [_charged_cells(rows, costs[arm][seed]["fit_cpu"]
                           + costs[arm][seed]["rollout_cpu"], volume)
            for seed, rows in sorted(arms_rows[arm].items())]


def _mean_strata(seeds_rows: dict[int, list[dict[str, Any]]
                                   ]) -> dict[str, float]:
    """Average per-stratum solves across seeds for one arm."""
    totals: dict[str, float] = {}
    for rows in seeds_rows.values():
        for row in rows:
            if row["accepted"]:
                key = str(row["stratum"])
                totals[key] = totals.get(key, 0.0) + 1
    count = len(seeds_rows)
    return {key: value / count for key, value in totals.items()}


def _solve_means(arms_rows: dict[str, dict[int, list[dict[str, Any]]]]
                 ) -> dict[str, float]:
    """Average solved tasks per arm across seeds."""
    means = {}
    for arm, seeds in arms_rows.items():
        solves = [sum(1 for row in rows if row["accepted"])
                  for rows in seeds.values()]
        means[arm] = sum(solves) / len(solves)
    return means


def decide_hard(feedback_mean: float, untrained_mean: float,
                imitation_mean: float, comparison: dict[str, Any],
                kill_criterion: str) -> dict[str, Any]:
    """Pass only when feedback beats both arms and survives equal CPU."""
    beats_untrained = feedback_mean > untrained_mean
    beats_imitation = feedback_mean > imitation_mean
    survives_cpu = not comparison["kill"]
    passed = beats_untrained and beats_imitation and survives_cpu
    return {"passed": passed, "feedback_mean": feedback_mean,
            "untrained_mean": untrained_mean,
            "imitation_mean": imitation_mean,
            "equal_cpu": comparison, "survives_equal_cpu": survives_cpu,
            "kill_criterion": kill_criterion}


def summarize_hard(arms_rows: dict[str, dict[int, list[dict[str, Any]]]],
                   costs: dict[str, dict[int, dict[str, float]]],
                   kill_criterion: str) -> dict[str, Any]:
    """Aggregate means, strata, equal-CPU truncation, and the verdict."""
    first = next(iter(next(iter(arms_rows.values())).values()))
    means = _solve_means(arms_rows)
    comparison = compare_pair(
        _charged_all(arms_rows, costs, "feedback", len(first)),
        _charged_all(arms_rows, costs, "streaming", len(first)))
    verdict = decide_hard(means["feedback"], means["untrained"],
                          means["imitation"], comparison, kill_criterion)
    strata = {arm: _mean_strata(seeds)
              for arm, seeds in arms_rows.items()}
    return {"means": means, "strata": strata,
            "equal_cpu_vs_streaming": comparison, "verdict": verdict}


def _seed_costs(fit: dict[str, Any],
                loop: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Separate fit, rollout, and inference charges for one seed."""
    zero = {"fit_cpu": 0.0, "rollout_cpu": 0.0, "record": None}
    return {"streaming": dict(zero), "untrained": dict(zero),
            "imitation": {"fit_cpu": fit["complete_cpu_seconds"],
                          "rollout_cpu": 0.0, "record": fit},
            "feedback": {"fit_cpu": loop["total_fit_cpu_seconds"],
                         "rollout_cpu": loop["total_rollout_cpu_seconds"],
                         "record": loop}}


def _fit_seed(train: list[dict[str, Any]],
              protocol: dict[str, Any], seed: int) -> dict[str, Any]:
    """Fit imitation and feedback fresh for one seed; train only."""
    evaluation = protocol["evaluation"]
    policies = {name: ConstructionPolicy(seed=seed, backend="ann")
                for name in ("untrained", "imitation", "feedback")}
    fit = fit_constructor(policies["imitation"], train,
                          evaluation["imitation_epochs"])
    loop = _feedback_fit(policies["feedback"], train,
                         evaluation["feedback"], evaluation["bound"],
                         evaluation["beam_width"])
    return {"policies": policies, "fits": _seed_costs(fit, loop),
            "imitation_fit": fit, "feedback_loop": loop}


def run_hard_seed(train: list[dict[str, Any]], tasks: list[dict[str, Any]],
                  protocol: dict[str, Any], seed: int) -> dict[str, Any]:
    """Fit both learned arms fresh, then score every arm at one seed."""
    fitted = _fit_seed(train, protocol, seed)
    evaluation = protocol["evaluation"]
    budget = require_hard_budget(protocol)
    policies = dict(fitted["policies"], streaming=None)
    arms = {}
    for arm in ("streaming", "untrained", "imitation", "feedback"):
        summary, rows = evaluate_arm(
            tasks, arm, policies[arm], budget, evaluation["bound"],
            evaluation["beam_width"], seed)
        arms[arm] = {"summary": summary, "rows": rows}
    return {"seed": seed, "arms": arms, "fits": fitted["fits"],
            "imitation_fit": fitted["imitation_fit"],
            "feedback_loop": fitted["feedback_loop"]}
