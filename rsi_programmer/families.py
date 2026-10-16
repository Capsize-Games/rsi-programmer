#!/usr/bin/env python3
"""Fixture families and the curriculum bridge for generated tasks.

generate_family grows count deterministic tasks with mixed const and
branch requirements and freezes their splits; curriculum_params turns
a PromotionGate failure clustering into family parameters so one
closed loop runs from gate verdicts to fresh data.
"""
from __future__ import annotations

import json
import random
from typing import Any

from rsi_programmer.tasks import generate_task
from rsi_programmer.traces import TraceRecorder


def generate_family(seed: int, count: int, bound: int,
                    branch_every: int, prefix: str,
                    start: int = 0) -> dict[str, Any]:
    """Count frozen tasks; every branch_every-th needs select."""
    rng = random.Random(seed)
    seen: set[tuple] = set()
    tasks: list[dict[str, Any]] = []
    index = start
    while len(tasks) < count:
        need_branch = (len(tasks) % branch_every) == 1
        task = generate_task(rng, "%s_%02d" % (prefix, index),
                             bound, need_branch, True)
        key = (tuple(tuple(c) for c in task["train"]),
               tuple(tuple(c) for c in task["held_out"]))
        index += 1
        if key in seen:
            continue
        seen.add(key)
        task["split_sha256"] = TraceRecorder.split_hash(task)
        tasks.append(task)
    return {"max_tree_ops": bound, "tasks": tasks, "version": 1}


def write_family(path: str, family: dict[str, Any]) -> None:
    """Deterministic fixture bytes: sorted keys, indent two."""
    with open(path, "w") as handle:
        json.dump(family, handle, sort_keys=True, indent=2)
        handle.write("\n")


def curriculum_params(report: list[dict[str, Any]]
                      ) -> dict[str, Any]:
    """Turn a PromotionGate curriculum into family parameters."""
    kinds = {str(item.get("kind", "")): int(item.get("count", 0))
             for item in report}
    if any("forg" in kind or "safety" in kind or "violat" in kind
           for kind in kinds):
        return {"count": 0, "branch_every": 2,
                "reason": "gate issue, not data"}
    if any("ambigu" in kind or "tie" in kind or "mismatch" in kind
           for kind in kinds):
        return {"count": 8, "branch_every": 2,
                "reason": "operand-dense top-up"}
    total = sum(kinds.values())
    if total <= 0:
        return {"count": 4, "branch_every": 4,
                "reason": "healthy top-up"}
    return {"count": min(total, 12), "branch_every": 2,
            "reason": "failure-proportional top-up"}
