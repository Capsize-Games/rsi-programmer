"""Sealed final gates for the construction-v1 pilot; no tuning here."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from rsi_programmer.construction_beam import beam_construct
from rsi_programmer.construction_check import check_graph
from rsi_programmer.construction_policy import ConstructionPolicy
from rsi_programmer.pre_eval import pre_evaluation_search

MARGIN = 0.10
COST_RATIO = 0.90


def verify_frozen_corpus(corpus_path: Path,
                         protocol_path: Path) -> dict[str, Any]:
    """Load the corpus only when its bytes match the frozen hash."""
    protocol: dict[str, Any] = json.loads(protocol_path.read_text())
    raw = corpus_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != protocol["corpus_sha256"]:
        raise ValueError("frozen construction corpus changed")
    corpus: dict[str, Any] = json.loads(raw)
    return {"protocol": protocol, "corpus": corpus}


def verify_generator_sources(root: Path,
                             protocol: dict[str, Any]) -> dict[str, str]:
    """Refuse when a pinned generator source differs from the protocol."""
    expected: dict[str, str] = protocol["source_provenance"][
        "generator_sha256"]
    actual: dict[str, str] = {}
    for name, digest in expected.items():
        seen = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if seen != digest:
            raise ValueError(f"pinned generator source changed: {name}")
        actual[name] = seen
    return actual


def require_final_only(tasks: list[dict[str, Any]]) -> None:
    """Refuse inference on anything but final-split tasks."""
    if not tasks or any(task.get("split") != "final"
                        for task in tasks):
        raise ValueError("final inference needs final-split tasks only")


def _check_task(task: dict[str, Any], method: str,
                model: ConstructionPolicy | None, budget: int,
                beam_seed: int) -> dict[str, Any]:
    """Run one final task, then check it without learner feedback."""
    observed = tuple(tuple(row) for row in task["observed"])
    if method == "streaming":
        out = pre_evaluation_search(
            None, tuple((x, y) for x, y, _ in observed),
            tuple(value for _, _, value in observed), 4, budget)
    else:
        out = beam_construct(model, observed, budget=budget,
                             width=8, seed=beam_seed)
    check = check_graph(
        out["graph"], observed,
        tuple(tuple(row) for row in task["held"]),
        tuple(tuple(term) for term in task["generator"]["polynomial"]))
    return {"id": task["id"], "accepted": check["accepted"],
            "graph": out["graph"], "actions": out.get("actions"),
            "executions": out.get("executions"),
            "error": out.get("error"), "checker": check}


def evaluate_final(tasks: list[dict[str, Any]], method: str,
                   model: ConstructionPolicy | None, budget: int,
                   beam_seed: int = 0) -> tuple[dict[str, Any],
                                                list[dict[str, Any]]]:
    """Count independently accepted final programs at one root cap."""
    require_final_only(tasks)
    before = model.fingerprint() if model is not None else None
    started = time.process_time()
    rows = [_check_task(task, method, model, budget, beam_seed)
            for task in tasks]
    if model is not None and model.fingerprint() != before:
        raise ValueError("evaluation changed neural weights")
    summary = {"accepted": sum(row["accepted"] for row in rows),
               "tasks": len(rows),
               "cpu_seconds": time.process_time() - started}
    return summary, rows


def learning_gate(rates: dict[int, dict[str, float]]) -> dict[str, Any]:
    """Pass when every trained seed beats both baselines by the margin."""
    seeds: dict[str, dict[str, float | bool]] = {}
    for seed, arm in rates.items():
        trained = arm["trained"] - arm["untrained"] >= MARGIN
        random = arm["trained"] - arm["random"] >= MARGIN
        seeds[str(seed)] = {"margin_untrained": arm["trained"]
                            - arm["untrained"],
                            "margin_random": arm["trained"] - arm["random"],
                            "passed": trained and random}
    return {"passed": all(info["passed"] for info in seeds.values()),
            "margin": MARGIN, "seeds": seeds}


def _first_full(sweep: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first full-coverage sweep entry, if any."""
    return next((row for row in sweep
                 if row["accepted"] == row["tasks"]), None)


def cost_gate(trained: list[dict[str, Any]],
              streaming: list[dict[str, Any]], fit_cpu: float,
              tasks: int) -> dict[str, Any]:
    """Pass on full coverage with amortized ANN cost below the ratio."""
    cover = {"trained": _first_full(trained),
             "streaming": _first_full(streaming)}
    if cover["trained"] is None or cover["streaming"] is None:
        return {"passed": False, "reason": "missing full coverage",
                "first_full_coverage": {
                    arm: (row["cap"] if row is not None else None)
                    for arm, row in cover.items()}}
    per_task = ((fit_cpu + cover["trained"]["cpu_seconds"]) / tasks,
                cover["streaming"]["cpu_seconds"] / tasks)
    ratio = per_task[0] / per_task[1] if per_task[1] > 0 else float("inf")
    return {"passed": ratio <= COST_RATIO, "ratio": ratio,
            "limit": COST_RATIO, "ann_per_task_cpu": per_task[0],
            "streaming_per_task_cpu": per_task[1],
            "first_full_coverage": {arm: row["cap"]
                                    for arm, row in cover.items()}}
