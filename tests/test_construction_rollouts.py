"""Checker-scored rollouts and teacher-shaped replay pairs."""
from __future__ import annotations

import unittest
from typing import Any

from rsi_programmer.construction_corpus import teacher_steps
from rsi_programmer.construction_policy import ConstructionPolicy
from rsi_programmer.construction_rollouts import (
    Rollout,
    collect_rollouts,
    trajectory_examples,
)
from rsi_programmer.construction_state import ConstructionState
from rsi_programmer.online_graph import graph_polynomial
from rsi_programmer.probe import Expr, lower_to_dag

PROGRAMS = {"add": lambda x, y: x + y, "mul": lambda x, y: x * y}


def _task(task_id: str, split: str, op: str) -> dict[str, Any]:
    """Build a one-operation witness task with tiny row sets."""
    program = lower_to_dag(Expr(op, Expr("input", "x"),
                                Expr("input", "y")))
    poly = graph_polynomial(program)
    rows = [[x, y, PROGRAMS[op](x, y)]
            for x in range(-1, 2) for y in range(-1, 2)]
    return {"id": task_id, "split": split, "observed": rows[:6],
            "held": rows[6:], "max_tree_ops": 4,
            "generator": {"minimal_tree_ops": 1, "program": program,
                          "polynomial": [list(term) for term in poly]}}


def _observed(task: dict[str, Any]) -> Any:
    """Pack list rows into the tuple form states require."""
    return tuple(tuple(row) for row in task["observed"])


def _accepted(task: dict[str, Any]) -> Rollout:
    """Search policy seeds until the untrained net is accepted."""
    for seed in range(10):
        policy = ConstructionPolicy(seed=seed, hidden=8)
        records, _ = collect_rollouts(policy, [task], 4, 64)
        if records[0].accepted:
            return records[0]
    raise AssertionError("untrained policy solved nothing")


class ConstructionRolloutsTests(unittest.TestCase):
    """Rollout scoring uses the checker; replay matches teachers."""

    def test_accepted_rollout_replays_its_graph_exactly(self) -> None:
        task = _task("train-add", "train", "add")
        rollout = _accepted(task)
        self.assertEqual(rollout.task_id, "train-add")
        self.assertEqual(rollout.actions[-1], {"op": "stop"})
        self.assertEqual(rollout.observed_match_fraction, 1.0)
        pairs = list(trajectory_examples([rollout],
                                         {task["id"]: task}))
        self.assertEqual([action for _, action in pairs],
                         rollout.actions)
        state = ConstructionState(_observed(task), 4)
        for _, action in pairs:
            state = state.apply(action)
        self.assertEqual(state.graph(), rollout.graph)

    def test_collection_reports_frozen_shape_and_cpu(self) -> None:
        tasks = [_task("train-add", "train", "add"),
                 _task("train-mul", "train", "mul")]
        policy = ConstructionPolicy(seed=0, hidden=8)
        records, cpu = collect_rollouts(policy, tasks, 4, 64)
        self.assertEqual([record.task_id for record in records],
                         ["train-add", "train-mul"])
        self.assertIsInstance(cpu, float)
        self.assertGreaterEqual(cpu, 0.0)
        for record in records:
            self.assertGreaterEqual(record.observed_match_fraction,
                                    0.0)
            self.assertLessEqual(record.observed_match_fraction, 1.0)
            self.assertGreaterEqual(record.executions, 0)
            if record.graph is None:
                self.assertEqual(record.observed_match_fraction, 0.0)

    def test_empty_budget_yields_nothing_to_replay(self) -> None:
        task = _task("train-add", "train", "add")
        records, _ = collect_rollouts(None, [task], 4, 0)
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].accepted)
        self.assertIsNone(records[0].graph)
        self.assertEqual(records[0].observed_match_fraction, 0.0)
        self.assertEqual(list(trajectory_examples(
            records, {task["id"]: task})), [])

    def test_non_train_tasks_are_refused(self) -> None:
        policy = ConstructionPolicy(seed=0, hidden=8)
        dev = _task("dev-add", "dev", "add")
        with self.assertRaises(ValueError):
            collect_rollouts(policy, [dev], 4, 64)
        with self.assertRaises(ValueError):
            collect_rollouts(policy, [_task("t", "train", "add"),
                                      dev], 4, 64)

    def test_replay_matches_teacher_stop_shape(self) -> None:
        task = _task("train-mul", "train", "mul")
        taught = teacher_steps(task)
        state = ConstructionState(_observed(task),
                                  task["max_tree_ops"])
        for _, action in taught:
            state = state.apply(action)
        rollout = Rollout(task["id"],
                          [action for _, action in taught], True,
                          1.0, len(taught) - 1, state.graph())
        replayed = list(trajectory_examples([rollout],
                                            {task["id"]: task}))
        self.assertEqual([action for _, action in replayed],
                         [action for _, action in taught])
        self.assertEqual([state.graph() for state, _ in replayed],
                         [state.graph() for state, _ in taught])
        self.assertEqual(replayed[-1][1], {"op": "stop"})
        self.assertFalse(replayed[-1][0].stopped)

    def test_diverged_replay_raises(self) -> None:
        task = _task("train-mul", "train", "mul")
        taught = teacher_steps(task)
        empty = {"input_types": {"x": "Int", "y": "Int"},
                 "nodes": [], "output": "x"}
        rollout = Rollout(task["id"],
                          [action for _, action in taught], True,
                          1.0, 1, empty)
        with self.assertRaises(ValueError):
            list(trajectory_examples([rollout], {task["id"]: task}))


if __name__ == "__main__":
    raise SystemExit(unittest.main())
