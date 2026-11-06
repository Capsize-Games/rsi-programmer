"""Feedback construction loop: delegation, determinism, bookkeeping."""
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from rsi_programmer import construction_training
from rsi_programmer.construction_corpus import teacher_steps
from rsi_programmer.construction_feedback import fit_with_feedback
from rsi_programmer.construction_policy import ConstructionPolicy
from rsi_programmer.construction_training import (
    fit_constructor, fit_on_examples,
)


def _task(task_id: str, op: str, targets: tuple[int, int, int]
          ) -> dict[str, Any]:
    rows = [[-2, 1, targets[0]], [0, 3, targets[1]],
            [2, -1, targets[2]]]
    return {"id": task_id, "split": "train", "observed": rows,
            "max_tree_ops": 4,
            "generator": {"program": {"nodes": [
                {"op": op, "type": "Int", "args": ["x", "y"]}]}}}


TASKS = [_task("add", "add", (-1, 3, 1)),
         _task("mul", "mul", (-2, 0, -2))]
TEACHER_COUNT = 4


def _teacher_examples() -> list[Any]:
    return [example for task in TASKS for example in teacher_steps(task)]


def _accepting_collect(policy: ConstructionPolicy,
                       tasks: list[dict[str, Any]], bound: int,
                       budget: int, width: int = 8
                       ) -> tuple[list[Any], float]:
    """Stub double: every task yields one accepted rollout record."""
    records = [{"task_id": task["id"], "actions": [],
                "accepted": True, "observed_match_fraction": 1.0,
                "executions": 1, "graph": None} for task in tasks]
    return records, 0.25


def _refusing_collect(policy: ConstructionPolicy,
                      tasks: list[dict[str, Any]], bound: int,
                      budget: int, width: int = 8
                      ) -> tuple[list[Any], float]:
    """Stub double: attribute-style records with no acceptance."""
    records = [SimpleNamespace(task_id=task["id"], actions=[],
                               accepted=False,
                               observed_match_fraction=0.0,
                               executions=1, graph=None)
               for task in tasks]
    return records, 0.5


def _replaying_examples(accepted: list[Any],
                        tasks_by_id: dict[str, Any]) -> list[Any]:
    """Stub double: accepted rollouts replay their teacher trace."""
    return [example for record in accepted for example in teacher_steps(
        tasks_by_id[record["task_id"]])]


class FeedbackTests(unittest.TestCase):
    """Check the W3 loop without the W2 rollout module."""

    def test_constructor_delegates_to_fit_on_examples(self) -> None:
        model = ConstructionPolicy(hidden=8)
        receipt = {"initial_hash": "a", "final_hash": "b",
                   "examples": TEACHER_COUNT, "epochs": 2,
                   "updates": 2, "final_loss": 0.5}
        with mock.patch.object(construction_training,
                               "fit_on_examples",
                               return_value=dict(receipt)) as fitted:
            result = fit_constructor(model, TASKS, epochs=2,
                                     batch_size=2, lr=0.01)
        passed = fitted.call_args.args
        self.assertIs(passed[0], model)
        self.assertEqual(len(passed[1]), TEACHER_COUNT)
        self.assertEqual(passed[2:], (2, 2, 0.01))
        self.assertEqual(result["tasks"], len(TASKS))
        self.assertEqual(result["teacher_steps"], TEACHER_COUNT)
        self.assertNotIn("examples", result)

    def test_constructor_matches_direct_fit_on_shared_keys(self) -> None:
        from_constructor = fit_constructor(
            ConstructionPolicy(seed=3, hidden=8), TASKS, epochs=1,
            batch_size=2)
        direct = fit_on_examples(ConstructionPolicy(seed=3, hidden=8),
                                 _teacher_examples(), epochs=1,
                                 batch_size=2)
        for key in ("initial_hash", "final_hash", "epochs", "updates",
                    "final_loss", "neural_work"):
            self.assertEqual(from_constructor[key], direct[key])
        self.assertEqual(from_constructor["teacher_steps"],
                         direct["examples"])

    def test_repeated_fits_are_deterministic(self) -> None:
        hashes = [fit_on_examples(ConstructionPolicy(seed=5, hidden=8),
                                  _teacher_examples(), epochs=1,
                                  batch_size=2)["final_hash"]
                  for _ in range(2)]
        self.assertEqual(hashes[0], hashes[1])

    def _feedback_result(self) -> tuple[str, dict[str, Any]]:
        model = ConstructionPolicy(seed=7, hidden=8)
        before = model.fingerprint()
        result = fit_with_feedback(
            model, TASKS, rounds=3, epochs_per_round=1,
            rollout_budget=4, bound=4, batch_size=2,
            collect=_accepting_collect,
            examples_of=_replaying_examples)
        return before, result

    def test_feedback_changes_fingerprint_and_sums_costs(self) -> None:
        before, result = self._feedback_result()
        self.assertEqual([record["round"] for record in result["rounds"]],
                         [0, 1, 2])
        opening = result["rounds"][0]
        self.assertEqual((opening["rollouts"], opening["accepted"],
                          opening["examples"]), (0, 0, TEACHER_COUNT))
        middle = result["rounds"][1]
        self.assertEqual((middle["accepted"], middle["examples"]), (2, 8))
        self.assertFalse(result["stopped_zero_acceptance"])
        self.assertAlmostEqual(result["total_rollout_cpu_seconds"], 0.5)
        fit_total = sum(record["fit"]["complete_cpu_seconds"]
                        for record in result["rounds"])
        self.assertAlmostEqual(result["total_fit_cpu_seconds"], fit_total)
        self.assertNotEqual(opening["fit"]["final_hash"], before)

    def test_feedback_freezes_w4_record_keys(self) -> None:
        _, result = self._feedback_result()
        for record in result["rounds"]:
            self.assertEqual(set(record), {"round", "rollouts",
                                           "accepted", "examples",
                                           "rollout_cpu_seconds", "fit"})
        self.assertEqual(set(result), {"rounds",
                                       "total_rollout_cpu_seconds",
                                       "total_fit_cpu_seconds",
                                       "stopped_zero_acceptance"})

    def test_feedback_resets_deterministically(self) -> None:
        runs = [fit_with_feedback(
            ConstructionPolicy(seed=9, hidden=8), TASKS, rounds=2,
            epochs_per_round=1, rollout_budget=4, bound=4,
            batch_size=2, collect=_accepting_collect,
            examples_of=_replaying_examples) for _ in range(2)]
        for first, second in zip(runs[0]["rounds"], runs[1]["rounds"]):
            self.assertEqual(first["fit"]["final_hash"],
                             second["fit"]["final_hash"])

    def test_zero_acceptance_stops_without_refit(self) -> None:
        model = ConstructionPolicy(seed=11, hidden=8)
        result = fit_with_feedback(
            model, TASKS, rounds=3, epochs_per_round=1,
            rollout_budget=4, bound=4, batch_size=2,
            collect=_refusing_collect,
            examples_of=_replaying_examples)
        self.assertTrue(result["stopped_zero_acceptance"])
        self.assertEqual(len(result["rounds"]), 2)
        stopped = result["rounds"][1]
        self.assertEqual((stopped["rollouts"], stopped["accepted"],
                          stopped["examples"]), (2, 0, TEACHER_COUNT))
        self.assertIsNone(stopped["fit"])
        self.assertAlmostEqual(result["total_rollout_cpu_seconds"], 0.5)
        only_fit = result["rounds"][0]["fit"]["complete_cpu_seconds"]
        self.assertAlmostEqual(result["total_fit_cpu_seconds"], only_fit)

    def test_feedback_real_collector_end_to_end(self) -> None:
        # Integration: production defaults must bind the real
        # collector/replay from construction_rollouts (worker W2),
        # not the stub doubles used above. The stub TASKS above
        # lack held rows/polynomials, so build witness tasks here.
        import rsi_programmer.construction_feedback as feedback
        from rsi_programmer.online_graph import graph_polynomial
        from rsi_programmer.probe import Expr, lower_to_dag
        self.assertIsNotNone(feedback.collect_rollouts)
        self.assertIsNotNone(feedback.trajectory_examples)
        tasks = []
        for op, fn in (("add", lambda x, y: x + y),
                       ("mul", lambda x, y: x * y)):
            program = lower_to_dag(Expr(op, Expr("input", "x"),
                                        Expr("input", "y")))
            rows = [[x, y, fn(x, y)] for x in range(-1, 2)
                    for y in range(-1, 2)]
            tasks.append({
                "id": op, "split": "train", "observed": rows[:6],
                "held": rows[6:], "max_tree_ops": 4,
                "generator": {
                    "program": program,
                    "polynomial": [list(t)
                                   for t in graph_polynomial(program)]}})
        model = ConstructionPolicy(seed=13, hidden=8)
        result = fit_with_feedback(
            model, tasks, rounds=2, epochs_per_round=1,
            rollout_budget=64, bound=4, batch_size=2)
        self.assertEqual(len(result["rounds"]), 2)
        later = result["rounds"][1]
        self.assertEqual(later["rollouts"], len(tasks))
        self.assertGreater(later["rollout_cpu_seconds"], 0.0)
        self.assertEqual(result["stopped_zero_acceptance"],
                         later["accepted"] == 0)

    def test_feedback_guards_reject_bad_inputs(self) -> None:
        poisoned = copy.deepcopy(TASKS)
        poisoned[0]["split"] = "dev"
        for tasks, rounds, collect in (
                (poisoned, 2, _accepting_collect),
                (TASKS, 0, _accepting_collect),
                (TASKS, 2, None)):
            with self.assertRaises(ValueError):
                fit_with_feedback(
                    ConstructionPolicy(hidden=8), tasks, rounds,
                    epochs_per_round=1, rollout_budget=4, bound=4,
                    collect=collect,
                    examples_of=_replaying_examples)


if __name__ == "__main__":
    unittest.main()
