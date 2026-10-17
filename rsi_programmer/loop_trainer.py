#!/usr/bin/env python3
"""Default loop trainer: ranker fit on shadow, transfer on novel.

Builds a SearchPolicy from the ranker-surface config fields
(hidden, steps, lr), fits briefly on frozen shadow traces,
runs the author joint pass with the threaded joint fields,
and scores per-task solve flags on shadow (train) and novel
(held). Evolve/population fields stay inert until those
trainers plug in; margin and beta have no policy-surface
counterpart and are recorded but inert here too.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rsi_programmer.author import refine_operands
from rsi_programmer.evolve import guided_search
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.traces import TraceRecorder

ROOT = Path(__file__).resolve().parent.parent
SHADOW = str(ROOT / "fixtures" / "rsi_shadow_tasks.json")
NOVEL = str(ROOT / "fixtures" / "rsi_novel_tasks.json")
EPOCHS = 5
BUDGET = 10 ** 9


def _solves(policy: SearchPolicy, fixture: str) -> list[float]:
    """Per-task solve flags over one fixture family."""
    rec = TraceRecorder(fixture)
    flags: list[float] = []
    for task in rec.load_family()["tasks"]:
        rows = tuple((c[0], c[1]) for c in task["train"])
        target = tuple(c[2] for c in task["train"])
        bound = int(task.get("max_tree_ops", 4))
        out = guided_search(policy, rows, target, bound,
                            BUDGET, str(task["id"]))
        flags.append(1.0 if out["solved"] else 0.0)
    return flags


def train_scores(config: dict[str, Any], seed: int,
                 workdir: str) -> dict[str, list[float]]:
    """Fit on shadow, score transfer on novel; deterministic.

    The author joint pass threads the joint config fields so
    joint_lambda/joint_epochs edits move the policy and the
    gate can price them; the loop EPOCHS constant still caps
    the ranker fit.
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
            "held": _solves(policy, NOVEL)}
