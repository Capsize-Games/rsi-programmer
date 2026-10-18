"""Focused checks for independent observed-context bundle validation."""
from __future__ import annotations

import copy
import hashlib
import json
import unittest
from typing import Any

from rsi_programmer.online_benchmark import CORPUS
from rsi_programmer.online_diagnostics import inspect_policy
from rsi_programmer.online_observed_context import ObservedContextPolicy
from scripts.audit_online_observed_context import (
    _context_current,
    _context_diagnostic,
    _normalized_context,
    _projection,
    _projection_digest,
    _validate_context_row,
)


class ObservedContextAuditTests(unittest.TestCase):
    """Reject context leakage and incorrect cold-policy evidence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tasks = json.loads(CORPUS.read_text())["tasks"]
        cls.task = cls.tasks[0]

    def _valid_row(
        self,
    ) -> tuple[dict[str, Any], ObservedContextPolicy, dict[str, Any]]:
        observed = tuple(tuple(row) for row in self.task["observed"])
        policy = ObservedContextPolicy(seed=0)
        before = policy.metrics()
        policy.set_observed_rows(observed)
        context = _normalized_context(observed)
        row: dict[str, Any] = {
            "seed": 0,
            "context_input_sha256": hashlib.sha256(
                json.dumps(context, separators=(",", ":"),
                           allow_nan=False).encode()).hexdigest(),
            "context_projection_sha256": _projection_digest(_projection(0)),
            "context_work": {"observation_scalars": 18,
                             "projection_mac": 288},
            "context_cpu_seconds": 0.001,
            "diagnostic_cpu_seconds": 0.0001,
            "diagnostic_before": inspect_policy(policy),
        }
        return row, policy, before

    def test_projection_and_cold_diagnostic_replay(self) -> None:
        matrix = _projection(0)
        self.assertEqual(len(matrix), 18)
        self.assertTrue(all(len(row) == 16 for row in matrix))
        self.assertTrue(set(value for row in matrix for value in row)
                        <= {-1, 0, 1})
        self.assertNotEqual(_projection_digest(matrix),
                            _projection_digest(_projection(1)))
        row, policy, before = self._valid_row()
        observed = tuple(tuple(values) for values in self.task["observed"])
        context = _normalized_context(observed)
        current = _context_current(context, matrix)
        self.assertEqual(policy._context_current, current)
        self.assertEqual(_context_diagnostic(policy, context, current),
                         inspect_policy(policy))
        self.assertEqual(_validate_context_row(
            row, self.task, policy, before), (context, current))

    def test_learned_context_diagnostics_match_modes_and_seeds(self) -> None:
        observed = tuple(tuple(row) for row in self.task["observed"])
        for seed in (0, 1, 2):
            for spiking in (True, False):
                with self.subTest(seed=seed, spiking=spiking):
                    policy = ObservedContextPolicy(seed=seed,
                                                   spiking=spiking)
                    policy.set_observed_rows(observed)
                    policy.start_episode("learned-probe")
                    action = policy.choose(
                        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                        ("add", "sub", "mul"))
                    policy.observe(0.73)
                    policy.end_episode()
                    policy.set_observed_rows(observed)
                    context = _normalized_context(observed)
                    current = _context_current(context, _projection(seed))
                    self.assertEqual(
                        _context_diagnostic(policy, context, current),
                        inspect_policy(policy))
                    self.assertIn(action, ("add", "sub", "mul"))

    def test_context_digest_work_diagnostic_and_cpu_corruption_rejected(
        self,
    ) -> None:
        changes = (
            lambda row: row.__setitem__("context_input_sha256", "0" * 64),
            lambda row: row.__setitem__("context_projection_sha256", "0" * 64),
            lambda row: row["context_work"].__setitem__("projection_mac", 287),
            lambda row: row["diagnostic_before"].__setitem__(
                "entropy_nats", 0.0),
            lambda row: row.__setitem__("context_cpu_seconds", -1.0),
        )
        for mutate in changes:
            with self.subTest(mutation=mutate):
                row, policy, before = self._valid_row()
                corrupt = copy.deepcopy(row)
                mutate(corrupt)
                with self.assertRaises(ValueError):
                    _validate_context_row(corrupt, self.task, policy, before)


if __name__ == "__main__":
    unittest.main()
