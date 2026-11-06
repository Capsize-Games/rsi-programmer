"""Expert-iteration construction learning from checker-accepted rollouts.

Frozen interface for the evaluation runner (worker W4):

* ``collect`` defaults to the real ``collect_rollouts`` from
  ``rsi_programmer.construction_rollouts``. It is called as
  ``collect(policy, tasks, bound, rollout_budget)`` and returns
  ``(records, collection_cpu_seconds)``. Each record carries
  ``task_id``, ``actions``, ``accepted``, ``observed_match_fraction``,
  ``executions`` and ``graph`` (``accepted`` is read from either a
  mapping key or an attribute).
* ``examples_of`` defaults to the real ``trajectory_examples`` from
  the same module. It is called as
  ``examples_of(accepted_records, tasks_by_id)`` and yields
  ``(state, action)`` pairs shaped exactly like ``teacher_steps``.
* Return keys are frozen: ``rounds`` (one record per round with
  ``round``, ``rollouts``, ``accepted``, ``examples``,
  ``rollout_cpu_seconds`` and ``fit``), ``total_rollout_cpu_seconds``,
  ``total_fit_cpu_seconds`` and ``stopped_zero_acceptance``. A round
  whose ``fit`` is ``None`` performed no refit (zero acceptance).
"""
from __future__ import annotations

from typing import Any

from rsi_programmer.construction_corpus import teacher_steps
from rsi_programmer.construction_policy import ConstructionPolicy
from rsi_programmer.construction_state import Action, ConstructionState
from rsi_programmer.construction_training import fit_on_examples

try:
    from rsi_programmer.construction_rollouts import (
        collect_rollouts, trajectory_examples,
    )
except ImportError:
    collect_rollouts, trajectory_examples = None, None

Example = tuple[ConstructionState, Action]


def _check_feedback(tasks: list[dict[str, Any]], rounds: int,
                    epochs: int, budget: int, bound: int,
                    collect: Any, examples_of: Any) -> None:
    if not tasks or any(task.get("split") != "train" for task in tasks):
        raise ValueError("only nonempty training tasks may supply labels")
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be positive")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs_per_round must be positive")
    if type(budget) is not int or budget < 0:
        raise ValueError("rollout_budget must be a nonnegative integer")
    if type(bound) is not int or not 1 <= bound <= 8:
        raise ValueError("bound must be an integer from 1 to 8")
    if collect is None or examples_of is None:
        raise ValueError("feedback needs collect/examples_of callables")


def _reset_to_init(policy: ConstructionPolicy) -> None:
    """Reload deterministic initial weights via a fresh same-config policy."""
    fresh = ConstructionPolicy(seed=policy.seed, backend=policy.backend,
                               hidden=policy.hidden, steps=policy.steps)
    policy.load_state_dict(fresh.state_dict())


def _fit_round(policy: ConstructionPolicy, examples: list[Example],
               epochs: int, batch_size: int, lr: float) -> dict[str, Any]:
    _reset_to_init(policy)
    return fit_on_examples(policy, examples, epochs, batch_size, lr)


def _round_record(number: int, rollouts: int, accepted: int,
                  examples: int, cpu: float,
                  fit: dict[str, Any] | None) -> dict[str, Any]:
    return {"round": number, "rollouts": rollouts, "accepted": accepted,
            "examples": examples, "rollout_cpu_seconds": cpu, "fit": fit}


def _accepted(records: list[Any]) -> list[Any]:
    kept = []
    for record in records:
        flag = (record.get("accepted") if isinstance(record, dict)
                else record.accepted)
        if flag:
            kept.append(record)
    return kept


def _opening_round(policy: ConstructionPolicy, teacher: list[Example],
                   epochs: int, batch_size: int, lr: float
                   ) -> dict[str, Any]:
    fit = _fit_round(policy, teacher, epochs, batch_size, lr)
    return _round_record(0, 0, 0, len(teacher), 0.0, fit)


def _later_round(policy: ConstructionPolicy, tasks: list[dict[str, Any]],
                 by_id: dict[str, Any], teacher: list[Example],
                 number: int, epochs: int, budget: int, bound: int,
                 batch_size: int, lr: float, collect: Any,
                 examples_of: Any, width: int = 8
                 ) -> tuple[dict[str, Any], float, bool]:
    records, cpu = collect(policy, tasks, bound, budget, width=width)
    accepted = _accepted(records)
    if not accepted:
        return _round_record(number, len(records), 0, len(teacher),
                             cpu, None), cpu, True
    extra = list(examples_of(accepted, by_id))
    fit = _fit_round(policy, teacher + extra, epochs, batch_size, lr)
    return _round_record(number, len(records), len(accepted),
                         len(teacher) + len(extra), cpu, fit), cpu, False


def _feedback_rounds(policy: ConstructionPolicy,
                     history: list[dict[str, Any]],
                     tasks: list[dict[str, Any]], by_id: dict[str, Any],
                     teacher: list[Example], rounds: int, epochs: int,
                     budget: int, bound: int, batch_size: int,
                     lr: float, collect: Any, examples_of: Any,
                     width: int = 8) -> tuple[float, bool]:
    total, stopped = 0.0, False
    for number in range(1, rounds):
        record, cpu, stop = _later_round(
            policy, tasks, by_id, teacher, number, epochs, budget,
            bound, batch_size, lr, collect, examples_of, width)
        history.append(record)
        total += cpu
        if stop:
            stopped = True
            break
    return total, stopped


def _fit_cpu_total(history: list[dict[str, Any]]) -> float:
    return sum(record["fit"]["complete_cpu_seconds"]
               for record in history if record["fit"] is not None)


def _feedback_receipt(history: list[dict[str, Any]], rollout_total: float,
                      stopped: bool) -> dict[str, Any]:
    return {"rounds": history,
            "total_rollout_cpu_seconds": rollout_total,
            "total_fit_cpu_seconds": _fit_cpu_total(history),
            "stopped_zero_acceptance": stopped}


def fit_with_feedback(policy: ConstructionPolicy,
                      tasks: list[dict[str, Any]], rounds: int,
                      epochs_per_round: int, rollout_budget: int,
                      bound: int, batch_size: int = 64, lr: float = 0.003,
                      width: int = 8,
                      collect: Any = collect_rollouts,
                      examples_of: Any = trajectory_examples,
                      ) -> dict[str, Any]:
    """Imitate, then refit from init on teacher plus accepted rollouts."""
    _check_feedback(tasks, rounds, epochs_per_round, rollout_budget,
                    bound, collect, examples_of)
    teacher = [example for task in tasks for example in teacher_steps(task)]
    by_id = {task["id"]: task for task in tasks}
    history = [_opening_round(policy, teacher, epochs_per_round,
                              batch_size, lr)]
    rollout_total, stopped = _feedback_rounds(
        policy, history, tasks, by_id, teacher, rounds,
        epochs_per_round, rollout_budget, bound, batch_size, lr,
        collect, examples_of, width)
    return _feedback_receipt(history, rollout_total, stopped)
