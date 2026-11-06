"""Checker-scored construction rollouts for expert iteration."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator

from rsi_programmer.construction_beam import beam_construct
from rsi_programmer.construction_check import check_graph
from rsi_programmer.construction_policy import ConstructionPolicy
from rsi_programmer.construction_state import (
    Action,
    ConstructionState,
    Graph,
    Rows,
)
from rsi_programmer.online_graph import Polynomial
from rsi_programmer.probe import interpret


@dataclass(frozen=True)
class Rollout:
    """One beam trajectory scored by the independent checker."""
    task_id: str
    actions: list[Action]
    accepted: bool
    observed_match_fraction: float
    executions: int
    graph: Graph | None


def _rows(task: dict[str, Any], key: str) -> Rows:
    """Read an integer row set without touching other splits."""
    return tuple(tuple(row) for row in task[key])


def _identity(task: dict[str, Any]) -> Polynomial:
    """Read the exact target polynomial for checker scoring."""
    return tuple(tuple(term) for term in
                task["generator"]["polynomial"])


def _match_fraction(graph: Graph | None, observed: Rows) -> float:
    """Share of observed rows the graph reproduces; None scores zero."""
    if graph is None or not observed:
        return 0.0
    matches = 0
    for x, y, want in observed:
        try:
            if interpret(graph, x, y) == want:
                matches += 1
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return matches / len(observed)


def _collect_one(policy: ConstructionPolicy | None,
                 task: dict[str, Any], bound: int, budget: int,
                 width: int, seed: int) -> Rollout:
    """Search one task and score the returned graph with the checker."""
    observed = _rows(task, "observed")
    outcome = beam_construct(policy, observed, bound, budget,
                             width, seed)
    graph = outcome["graph"]
    accepted = check_graph(graph, observed, _rows(task, "held"),
                           _identity(task), bound)["accepted"]
    return Rollout(task["id"], list(outcome["actions"]), accepted,
                   _match_fraction(graph, observed),
                   outcome["executions"], graph)


def collect_rollouts(policy: ConstructionPolicy | None,
                     tasks: list[dict[str, Any]], bound: int,
                     budget: int, width: int = 8,
                     seed: int = 0) -> tuple[list[Rollout], float]:
    """Score one beam rollout per training task; never read eval rows.

    Task ``index`` is searched with ``seed + index`` so the random
    control varies per task while staying deterministic. The float
    is process CPU seconds for the whole collection call.
    """
    if any(task.get("split") != "train" for task in tasks):
        raise ValueError("rollout collection needs training tasks")
    started = time.process_time()
    records = [_collect_one(policy, task, bound, budget, width,
                            seed + index)
               for index, task in enumerate(tasks)]
    return records, time.process_time() - started


def trajectory_examples(
        rollouts: list[Rollout],
        tasks_by_id: dict[str, dict[str, Any]]
) -> Iterator[tuple[ConstructionState, Action]]:
    """Replay accepted rollouts as pre-action pairs, teacher-shaped.

    Each yielded pair matches ``teacher_steps`` conventions: states
    are pre-action, and the final pair carries ``{"op": "stop"}``.
    A replay whose final graph differs from the rollout raises.
    """
    for rollout in rollouts:
        if not rollout.accepted:
            continue
        task = tasks_by_id[rollout.task_id]
        state = ConstructionState(_rows(task, "observed"),
                                  task["max_tree_ops"])
        for action in rollout.actions:
            yield state, action
            state = state.apply(action)
        if state.graph() != rollout.graph:
            raise ValueError("replay diverged for " + rollout.task_id)
