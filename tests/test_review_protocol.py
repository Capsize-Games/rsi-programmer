"""Synthetic regressions for protocol and benchmark review findings."""

from __future__ import annotations

import unittest
import random
from unittest.mock import patch
from typing import Any

from rsi_programmer.author import author_scores
from rsi_programmer.headless import HeadlessLoop
from rsi_programmer.ood import ablate_author_bonus, behavior_hash
from rsi_programmer.policy import DIM, SearchPolicy
from rsi_programmer.ood import _probe_feats, _probe_layer
from rsi_programmer.probe import Expr
from rsi_programmer.rich import fitness_of, hard_neg, hard_pos, rollup
from rsi_programmer.evolve import PolicyLoop
from rsi_programmer.promote import PromotionGate
from rsi_programmer.scale import ScaleUp, _regression_flags
from rsi_programmer.tasks import generate_task


class ProtocolReviewTests(unittest.TestCase):
    def test_behavior_hash_covers_call_settings_and_policy_steps(
            self) -> None:
        policy = SearchPolicy(seed=2, hidden=4, steps=2)
        before = behavior_hash(policy, {"extra": [0.0, 0.0]})
        changed = behavior_hash(policy, {"extra": [0.1, 0.0]})
        self.assertNotEqual(before, changed)
        policy._steps = 3
        self.assertNotEqual(before,
                            behavior_hash(policy,
                                          {"extra": [0.0, 0.0]}))
        policy.set_rich_bonus([1.0, 0.0, 0.0, 0.0])
        self.assertNotEqual(before,
                            behavior_hash(policy,
                                          {"extra": [0.0, 0.0]}))

    def test_exact_flag_is_external_and_can_be_ablated(self) -> None:
        values = [0.0] * DIM + [0.0, 0.0, 1.0, 0.0]

        class NeuralRecorder:
            def scores(self, feats: list[list[float]]) -> list[float]:
                self.got = feats
                return [0.0]

        policy = NeuralRecorder()
        scored = author_scores(policy, [values], [0.0, 0.0, 2.0, 0.0])
        self.assertEqual(len(policy.got[0]), DIM)
        self.assertEqual(scored, [2.0])
        self.assertEqual(ablate_author_bonus(
            [1.0, 2.0, 3.0, 4.0], ("exact_flag",)),
            [1.0, 2.0, 0.0, 4.0])

    def test_headless_pairs_one_average_per_task(self) -> None:
        loop = object.__new__(HeadlessLoop)
        loop._trace = "synthetic"
        loop._max_gens = 2
        loop._compare = False
        captured = {}

        def gate(ref: list[float], cand: list[float], bad: int,
                 cost: int, wall: float) -> dict[str, Any]:
            captured["ref"] = ref
            captured["cand"] = cand
            return {"promote": False}

        loop._gate_decide = gate
        tasks = [{"id": "a"}, {"id": "b"}]
        baseline = {"solve_rate": .5, "cost": 10,
                    "cost_tried": 10,
                    "solved_by_task": {
                        "a": {"solved": False},
                        "b": {"solved": True}}}
        decisions = [
            {"task": "a", "solved": True, "sandbox": "accepted",
             "tried": 2, "signature_evaluations": 2, "path": "beam"},
            {"task": "a", "solved": False, "sandbox": "accepted",
             "tried": 3, "signature_evaluations": 3, "path": "beam"},
            {"task": "b", "solved": True, "sandbox": "accepted",
             "tried": 4, "signature_evaluations": 4, "path": "beam"},
            {"task": "b", "solved": True, "sandbox": "accepted",
             "tried": 5, "signature_evaluations": 5, "path": "beam"}]
        result = loop._summarize(0.0, baseline, tasks, decisions)
        self.assertEqual(captured["ref"], [0.0, 1.0])
        self.assertEqual(captured["cand"], [.5, 1.0])
        self.assertEqual(result["paired_task_count"], 2)
        self.assertEqual(result["cost_signature_evaluations"], 14)

    def test_headless_skips_oracle_for_rejected_graph(self) -> None:
        loop = object.__new__(HeadlessLoop)

        class RejectingBox:
            def run_candidate(self, graph: dict[str, Any],
                              pairs: list[list[int]],
                              claim: dict[str, Any] | None
                              ) -> dict[str, Any]:
                return {"status": "rejected", "violations": ["verifier"]}

            def lineage_hash(self, graph: dict[str, Any]) -> str:
                raise AssertionError("rejected graph must not be hashed")

        class UnusedRecorder:
            def oracle_holds(self, task: dict[str, Any],
                             graph: dict[str, Any]) -> bool:
                raise AssertionError("rejected graph must not reach oracle")

        loop._box = RejectingBox()
        loop._recorder = UnusedRecorder()
        verdict, holds, lineage, solved = loop._judge(
            {"nodes": [{"op": []}], "output": 0}, [], None, {})
        self.assertEqual(verdict["status"], "rejected")
        self.assertFalse(holds)
        self.assertEqual(lineage, "")
        self.assertFalse(solved)

    def test_headless_reports_oracle_exception(self) -> None:
        loop = object.__new__(HeadlessLoop)

        class AcceptingBox:
            def run_candidate(self, graph: dict[str, Any],
                              pairs: list[list[int]],
                              claim: dict[str, Any] | None
                              ) -> dict[str, Any]:
                return {"status": "accepted", "violations": []}

            def lineage_hash(self, graph: dict[str, Any]) -> str:
                return "hash"

        class BrokenRecorder:
            def oracle_holds(self, task: dict[str, Any],
                             graph: dict[str, Any]) -> bool:
                raise ValueError("bad graph")

        loop._box = AcceptingBox()
        loop._recorder = BrokenRecorder()
        verdict, holds, lineage, solved = loop._judge({}, [], None, {})
        self.assertEqual(verdict["status"], "oracle-error")
        self.assertIn("supervisor-oracle-error", verdict["violations"])
        self.assertFalse(holds)
        self.assertEqual(lineage, "")
        self.assertFalse(solved)

    def test_fallback_requires_held_out_oracle(self) -> None:
        loop = object.__new__(HeadlessLoop)
        loop._width = 1
        loop._compare = True

        class Recorder:
            def oracle_holds(self, task: dict[str, Any],
                             graph: dict[str, Any]) -> bool:
                return False

        def split(task: dict[str, Any]
                  ) -> tuple[tuple[Any, ...], tuple[Any, ...], int,
                             list[list[int]]]:
            return (), (), 1, []

        def append(record: dict[str, Any]) -> None:
            del record

        loop._recorder = Recorder()
        loop._split = split
        loop._append = append

        beam = {"solved": False, "graph": None, "tried": 1,
                "verdict": "miss"}
        fallback = {"solved": True, "graph": {"output": "x",
                    "nodes": []}, "tried": 1,
                    "signature_evaluations": 1, "verdict": "solved"}
        with patch("rsi_programmer.headless.beam_search",
                   return_value=beam), \
                patch("rsi_programmer.headless.guided_search",
                      return_value=fallback):
            result = loop._propose(SearchPolicy(seed=0), [],
                                   {"id": "t"}, 0, None, 10)
        self.assertTrue(result["fallback_train_solved"])
        self.assertFalse(result["fallback_oracle_holds"])
        self.assertFalse(result["fallback_solved"])

    def test_headless_insufficient_pairs_refuses_without_crashing(
            self) -> None:
        loop = object.__new__(HeadlessLoop)
        loop._gate = PromotionGate(min_pairs=3)
        result = loop._gate_decide([1.0], [1.0], 0, 0, 0.0)
        self.assertFalse(result["promote"])
        self.assertEqual(result["paired_task_count"], 1)
        self.assertIn("at least 3", result["reasons"][0])

    def test_scale_flags_each_family_regression(self) -> None:
        candidate = {"w1": {"solve_rate": 1.0},
                     "hard": {"solve_rate": .25}}
        baseline = {"w1": {"solve_rate": .75},
                    "hard": {"solve_rate": .5}}
        self.assertEqual(_regression_flags(candidate, baseline),
                         {"w1": False, "hard": True})

    def test_training_ordinals_are_local_not_answer_positions(
            self) -> None:
        task = {"train": [[1, 2, 3]], "max_tree_ops": 2}
        triples, total = _probe_layer()
        encoded = _probe_feats(task, triples, total)
        self.assertEqual(encoded[0][-2:], [1 / total,
                                           1 - 1 / total])
        self.assertEqual(encoded[-1][-2:], [1.0, 0.0])

        leaf_x = Expr("input", "x", None, 0)
        leaf_y = Expr("input", "y", None, 0)
        solution = Expr("add", leaf_x, leaf_y, 1)
        rows = ((1, 2),)
        target = (3,)
        early, _ = hard_pos(solution, rows, target, 2, 1)
        late, _ = hard_pos(solution, rows, target, 2, 999999)
        self.assertEqual(early[0][-2:], late[0][-2:])
        self.assertEqual(early[0][-2] + early[0][-1], 1.0)

    def test_rich_negative_groups_skip_exact_train_solvers(self) -> None:
        rows = ((1, 2),)
        target = (3,)
        negatives, _ = hard_neg(rows, target, 1)
        solution = Expr("add", Expr("input", "x", None, 0),
                        Expr("input", "y", None, 0), 1)
        positive, _ = hard_pos(solution, rows, target, 1)
        self.assertNotIn(positive[0], [features for features, _ in negatives])

    def test_rich_rollup_uses_canonical_cost_with_legacy_aliases(
            self) -> None:
        report = rollup([{"family": "w1", "solved": True,
                          "tried": 10, "signature_evaluations": 3,
                          "base_tried": 8,
                          "base_signature_evaluations": 2}], "w1")
        self.assertEqual(report["cost"], 3)
        self.assertEqual(report["baseline_cost"], 2)
        self.assertEqual(report["cost_tried"], 10)
        self.assertEqual(report["baseline_tried"], 8)

    def test_scale_family_cost_uses_canonical_counter(self) -> None:
        family = ScaleUp._fam([{"task": "t", "solved": True,
                                "tried": 10,
                                "signature_evaluations": 3}])
        self.assertEqual(family["cost"], 3)
        self.assertEqual(family["cost_tried"], 10)

    def test_rich_fitness_does_not_credit_held_failure(self) -> None:
        class FakePolicy:
            def set_rich_bonus(self, weights: list[float]) -> None:
                del weights

            def snapshot(self) -> dict[str, Any]:
                return {}

            def restore(self, snapshot: dict[str, Any]) -> None:
                del snapshot

            def perturb(self, sigma: float, seed: int) -> None:
                del sigma, seed

            def scores(self, feats: list[list[float]]) -> list[float]:
                return [0.0 for _ in feats]

        result = {"solved": True, "graph": {"nodes": [],
                  "output": "x"}, "tried": 1,
                  "signature_evaluations": 1}
        def held_oracle(graph: dict[str, Any]) -> bool:
            del graph
            return False

        jobs = [((), (), 1, 2, "task", held_oracle)]
        with patch("rsi_programmer.rich.SearchPolicy",
                   return_value=FakePolicy()), \
                patch("rsi_programmer.rich.guided_search",
                      return_value=dict(result)):
            selection = [[[0.0], [1.0]]]
            fitness = fitness_of(FakePolicy(), jobs, selection,
                                 [0.0] * 4, 2, 1)
        self.assertEqual(fitness["correctness"], 0.0)
        self.assertEqual(fitness["recovery"], 0.0)

    def test_generated_task_discloses_selection_filter(self) -> None:
        task = generate_task(random.Random(8), "tiny", 2, False, False,
                             search_filter=True)
        select = task["selection_filter"]
        self.assertTrue(select["enumerator_solved_within_bound"])
        self.assertTrue(select["enumerator_solution_held_out_confirmed"])
        self.assertGreater(select["enumerator_candidates_tried"], 0)

    def test_recovery_requires_held_out_oracle(self) -> None:
        class FakePolicy:
            def snapshot(self) -> dict[str, Any]:
                return {}

        class NoisyPolicy:
            def restore(self, snapshot: dict[str, Any]) -> None:
                return None

            def perturb(self, sigma: float, seed: int) -> None:
                return None

        class Recorder:
            def oracle_holds(self, task: dict[str, Any],
                             graph: dict[str, Any]) -> bool:
                return False

        loop = object.__new__(PolicyLoop)
        loop._seed = 3
        loop._budget = 10
        loop._base = {"task": {"signature_evaluations": 5}}
        loop._recorder = Recorder()
        task = {"id": "task", "train": [[1, 2, 3]],
                "held_out": [[2, 3, 5]]}
        found = {"solved": True, "graph": {"nodes": [], "output": "x"},
                 "signature_evaluations": 4}
        with patch("rsi_programmer.evolve.SearchPolicy",
                   return_value=NoisyPolicy()), \
                patch("rsi_programmer.evolve.guided_search",
                      return_value=found):
            self.assertEqual(loop._recovery(FakePolicy(), task, 4), 0.0)


if __name__ == "__main__":
    unittest.main()
