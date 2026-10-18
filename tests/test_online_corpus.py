"""Contracts for the frozen synthetic online-graph task corpus."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer import online_corpus


class OnlineCorpusTests(unittest.TestCase):
    """Verify reproducibility, reachability, and answer-key integrity."""

    def test_corpus_is_reproducible_and_has_frozen_phase_quotas(
            self) -> None:
        first = online_corpus.build_online_corpus()
        second = online_corpus.build_online_corpus()
        self.assertEqual(first, second)
        tasks = first["tasks"]
        self.assertEqual(len(tasks), 48)
        self.assertEqual(len({task["id"] for task in tasks}), 48)
        self.assertEqual(len({tuple(map(tuple, task["polynomial"]))
                              for task in tasks}), 48)
        expected = (
            {"add": 12, "sub": 2, "mul": 2},
            {"add": 2, "sub": 12, "mul": 2},
            {"add": 2, "sub": 2, "mul": 12},
        )
        for phase, quotas in enumerate(expected):
            phase_tasks = [task for task in tasks
                           if task["phase"] == phase]
            actual = {operator: sum(task["root"] == operator
                                    for task in phase_tasks)
                      for operator in online_corpus.OPERATORS}
            self.assertEqual(actual, quotas)
            self.assertEqual(len(phase_tasks), 16)

    def test_candidate_pools_and_primary_reachability(self) -> None:
        witnesses, positions, all_polynomials, count = (
            online_corpus._candidate_universe())
        self.assertEqual(count, 432)
        self.assertEqual(len(all_polynomials), 104)
        self.assertEqual(
            {operator: len(witnesses[operator])
             for operator in online_corpus.OPERATORS},
            {"add": 16, "sub": 36, "mul": 40})
        self.assertEqual(
            {operator: max(positions[operator].values())
             for operator in online_corpus.OPERATORS},
            {"add": 120, "sub": 141, "mul": 125})
        corpus = online_corpus.build_online_corpus()
        for task in corpus["tasks"]:
            self.assertLessEqual(task["candidate_index"],
                                 online_corpus.PRIMARY_BUDGET)
            polynomial = tuple(tuple(term)
                               for term in task["polynomial"])
            self.assertIn(polynomial, witnesses[task["root"]])
            self.assertEqual(
                task["candidate_index"],
                positions[task["root"]][polynomial])

    def test_shared_rows_are_disjoint_and_target_independent(self) -> None:
        corpus = online_corpus.build_online_corpus()
        metadata = corpus["protocol"]
        observed = [tuple(pair)
                    for pair in metadata["observed_input_pairs"]]
        held = [tuple(pair) for pair in metadata["held_input_pairs"]]
        self.assertEqual(len(observed), 6)
        self.assertEqual(len(held), 32)
        self.assertEqual(len(set(observed)), 6)
        self.assertEqual(len(set(held)), 32)
        self.assertFalse(set(observed) & set(held))
        for x, y in observed + held:
            self.assertTrue(-9 <= x <= 9 and -9 <= y <= 9)
        for task in corpus["tasks"]:
            self.assertEqual([row[:2] for row in task["observed"]],
                             [list(pair) for pair in observed])
            self.assertEqual([row[:2] for row in task["held"]],
                             [list(pair) for pair in held])

    def test_observed_rows_separate_targets_from_candidate_polynomials(
            self) -> None:
        corpus = online_corpus.build_online_corpus()
        _witnesses, _positions, all_polynomials, _count = (
            online_corpus._candidate_universe())
        rows = [tuple(pair) for pair in
                corpus["protocol"]["observed_input_pairs"]]
        signatures: dict[tuple[int, ...], set[tuple[tuple[int, int, int],
                                                  ...]]] = {}
        for polynomial in all_polynomials:
            signature = tuple(online_corpus.evaluate_polynomial(
                polynomial, x, y) for x, y in rows)
            signatures.setdefault(signature, set()).add(polynomial)
        for task in corpus["tasks"]:
            target = tuple(tuple(term) for term in task["polynomial"])
            signature = tuple(row[2] for row in task["observed"])
            self.assertEqual(signatures[signature], {target})

    def test_witnesses_match_independent_polynomial_evaluation(self) -> None:
        corpus = online_corpus.build_online_corpus()
        for task in corpus["tasks"]:
            polynomial = tuple(tuple(term)
                               for term in task["polynomial"])
            for row in task["observed"] + task["held"]:
                x, y, expected = row
                self.assertEqual(
                    online_corpus.evaluate_polynomial(polynomial, x, y),
                    expected)
                self.assertEqual(
                    online_corpus.evaluate_witness(task["witness"], x, y),
                    expected)

    def test_protocol_freezes_budgets_seeds_arms_and_policy(self) -> None:
        protocol = online_corpus.build_online_corpus()["protocol"]
        self.assertEqual(protocol["primary_budget"], 144)
        self.assertEqual(protocol["secondary_budget"], 432)
        self.assertEqual(protocol["model_seeds"], [0, 1, 2])
        self.assertEqual(len(protocol["arms"]), 5)
        self.assertEqual(protocol["policy"], {
            "input_features": 6,
            "hidden_units": 16,
            "steps": 6,
            "learning_rate": 0.05,
            "eligibility_decay": 0.0,
            "reward_baseline": 0.05,
            "weight_clip": 5.0,
            "update_clip": 0.05,
            "membrane_beta": 0.8,
            "threshold": 1.0,
            "initialization":
            "seeded random projection; no trained encoder",
            "credit_assignment":
            "current decision only; no cross-action trace",
        })
        self.assertEqual(protocol["held_feedback"],
                         "never returned to the policy")
        self.assertEqual(
            protocol["acceptance_gates"]["persistent_vs_frozen_reset"]
            ["first_lane_accuracy_margin"], 0.10)
        self.assertEqual(protocol["memory_reporting"],
                         "whole-run peak RSS and serialized model size only")

    def test_write_returns_digest_of_frozen_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "online_graph.json"
            digest = online_corpus.write_online_corpus(path)
            payload = path.read_bytes()
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
            loaded: dict[str, Any] = json.loads(payload)
            self.assertEqual(loaded, online_corpus.build_online_corpus())

    def test_checked_in_fixture_matches_generator(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "fixtures/online_graph_v1.json"
        payload = path.read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "a4f9af2dfc0d8c4539d700b032dcc223"
            "c82aa272f02497902024d9f371096140")
        self.assertEqual(json.loads(payload),
                         online_corpus.build_online_corpus())


if __name__ == "__main__":
    unittest.main()
