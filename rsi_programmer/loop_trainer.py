#!/usr/bin/env python3
"""Default loop trainer: fit on shadow, score shadow and v2.

Builds a SearchPolicy from the ranker-surface config fields
(hidden, steps, lr), fits briefly on frozen shadow traces,
runs the author joint pass with the configured joint fields,
then scores per-task results on shadow (train) and v2 (held).
A score is nonzero only when the independent oracle accepts
the result on both train and held rows; accepted tasks are
scored by canonical candidate-signature work. Since guided
search computes signatures eagerly, this work measure does
not establish a ranking improvement. Evolve/population fields
stay inert until those trainers plug in; margin and beta have
no policy-surface counterpart and are recorded but inert here.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from rsi_programmer.author import refine_operands
from rsi_programmer.evolve import guided_search
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.probe import CONSTS
from rsi_programmer.traces import TraceRecorder

ROOT = Path(__file__).resolve().parent.parent
SHADOW = str(ROOT / "fixtures" / "rsi_shadow_tasks.json")
HELD = str(ROOT / "fixtures" / "rsi_ood_v2.json")
EPOCHS = 5
BUDGET = 10 ** 9
ACTIVE_FIELDS = ("hidden", "steps", "lr", "joint_lambda",
                 "joint_epochs")


def _score(out: dict[str, Any], held: bool = True) -> float:
    """Task score: held-correct result weighted by signature work.

    A miss scores zero. A held-correct result is scored using
    log candidate-signature work against the configured scale.
    Older search outputs fall back to their tried counter.
    """
    if not out["solved"] or not held:
        return 0.0
    work = out.get("signature_evaluations", out["tried"])
    return 2.0 - math.log10(work + 1) / \
        math.log10(BUDGET)


def _consts_for(task: dict[str, Any]) -> tuple[int, ...]:
    """Constant terminals for tasks flagged needing them.

    Mirrors the oracle-side rule: CONSTS iff the task lists
    const among its features, else bare terminals.
    """
    if "const" in task.get("features", []):
        return CONSTS
    return ()


def _solves(policy: SearchPolicy, fixture: str) -> list[float]:
    """Per-task cost-aware scores after train and held checks."""
    rec = TraceRecorder(fixture)
    flags: list[float] = []
    for task in rec.load_family()["tasks"]:
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        out = guided_search(policy, rows, target, bound,
                            BUDGET, str(task["id"]),
                            oracle=lambda graph: rec.oracle_holds(
                                task, graph),
                            consts=_consts_for(task))
        held = bool(out["solved"] and out.get("graph") is not None
                    and rec.oracle_holds(task, out["graph"]))
        flags.append(_score(out, held))
    return flags


def train_scores(config: dict[str, Any], seed: int,
                 workdir: str) -> dict[str, list[float]]:
    """Fit on shadow traces and score shadow plus v2 tasks.

    The author joint pass receives the joint config fields.
    Their effect on the gate score is not guaranteed because
    accepted results use eager signature-work counts.
    """
    _ = workdir
    policy = SearchPolicy(seed=seed,
                          hidden=int(config.get("hidden", 16)),
                          steps=int(config.get("steps", 6)),
                          lr=float(config.get("lr", 0.05)))
    groups = policy.trace_groups(SHADOW)
    policy.fit(groups, EPOCHS)
    refine_operands(policy, groups, SHADOW,
                    lam=float(config.get("joint_lambda", 0.2)),
                    epochs=int(config.get("joint_epochs", 5)))
    return {"train": _solves(policy, SHADOW),
            "held": _solves(policy, HELD)}
