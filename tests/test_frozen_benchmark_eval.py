"""Evaluation boundaries and runner audit behavior."""
from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from rsi_programmer.benchmark_policy_random import SeededRandomPolicy
from scripts import benchmark_pre_eval as benchmark


class FrozenEvaluationTests(unittest.TestCase):
    """Exercise leakage guards and run-integrity checks cheaply."""

    def test_search_receives_only_training_rows_and_targets(self) -> None:
        """The search call has no held-out rows or oracle callback."""
        task = {
            "id": "unit_task",
            "max_tree_ops": 3,
            "train": [[1, 2, 3], [2, 4, 6]],
            "held_out": [[100, 200, 300]],
        }
        received: dict[str, Any] = {}

        def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
            received["args"] = args
            received["kwargs"] = kwargs
            return {"solved": False, "graph": None}

        result = benchmark._search_once(
            None, task, 32, "unit_task", capture,
        )
        self.assertFalse(result["solved"])
        self.assertEqual(received["args"][1], ((1, 2), (2, 4)))
        self.assertEqual(received["args"][2], (3, 6))
        self.assertEqual(received["args"][3:5], (3, 32))
        self.assertIsNone(received["kwargs"]["oracle"])
        self.assertNotIn((100, 200), received["args"])

    def test_graph_checks_cover_train_and_held_rows(self) -> None:
        """Both independent checks and search verdict govern acceptance."""
        task = {
            "train": [[1, 2, 3], [2, 4, 6]],
            "held_out": [[5, 7, 12]],
        }
        graph = {
            "input_types": {"x": "Int", "y": "Int"},
            "nodes": [{"op": "add", "type": "Int",
                       "args": ["x", "y"]}],
            "output": 0,
        }
        report = benchmark._validate_returned_graph(graph, task)
        self.assertTrue(report["train_valid"])
        self.assertTrue(report["held_out_valid"])
        self.assertTrue(benchmark._accepted(True, report))
        self.assertFalse(benchmark._accepted(False, report))
        task["held_out"][0][2] = 99
        report = benchmark._validate_returned_graph(graph, task)
        self.assertTrue(report["train_valid"])
        self.assertFalse(report["held_out_valid"])
        self.assertFalse(benchmark._accepted(True, report))

    def test_seeded_random_rank_is_reproducible(self) -> None:
        """Stable pseudo-random ordering does not use shared RNG state."""
        features = [
            [float(index == field) for field in range(12)]
            for index in range(5)
        ]
        first = SeededRandomPolicy(benchmark._random_seed(1, "task_a"))
        second = SeededRandomPolicy(benchmark._random_seed(1, "task_a"))
        self.assertEqual(first.rank(features), second.rank(features))
        self.assertNotEqual(
            first.rank(features),
            SeededRandomPolicy(
                benchmark._random_seed(2, "task_a"),
            ).rank(features),
        )

    def test_small_runner_keeps_errors_and_detects_mutation(self) -> None:
        """A fake full matrix retains search errors and flags drift."""
        def fake_model() -> Any:
            state = [0]
            return SimpleNamespace(
                state=state,
                weights=lambda: {"state": state[0]},
                metadata=lambda: {
                    "name": "fake", "trainable_parameters": 0,
                },
            )

        models = {name: fake_model() for name in (
            "heuristic", "untrained_snn", "trained_snn", "mlp",
            "recurrent",
        )}
        fit_records = {
            name: {
                "weights_after_fit_sha256": benchmark._weights_hash(model),
                "state_fingerprint_after_fit": benchmark._state_hash(model),
            }
            for name, model in models.items()
        }
        protocol = {
            "protocol_id": "tiny",
            "evaluation": {
                "candidate_scan_budgets": [1],
                "tree_bound": 3,
            },
            "training": {"seeds": [0], "epochs": 1},
            "corpus": {
                "counts": {"train_tasks": 0, "eval_tasks": 1},
                "families": ["balanced"],
            },
            "decision_rule": {
                "matched_success_targets": [0.5],
                "claim_scope": "tiny test only",
            },
        }
        task = {
            "id": "tiny_eval", "family": "balanced", "max_tree_ops": 3,
            "train": [[1, 2, 3], [2, 4, 6]],
            "held_out": [[5, 7, 12]],
        }
        corpus = {"splits": {"train": [], "eval": [task]}}
        env_calls = 0

        def environment() -> dict[str, Any]:
            nonlocal env_calls
            env_calls += 1
            return {"fixed": env_calls}

        def search(policy: Any, *args: Any,
                   **kwargs: Any) -> dict[str, Any]:
            del args, kwargs
            if policy is models["recurrent"]:
                raise RuntimeError("record this attempt")
            if policy is models["heuristic"]:
                policy.state[0] += 1
            return {"solved": True, "graph": {
                "input_types": {"x": "Int", "y": "Int"},
                "nodes": [{"op": "add", "type": "Int",
                           "args": ["x", "y"]}],
                "output": 0,
            }, "tried": 1}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (patch.object(benchmark, "frozen_inputs",
                               return_value=(protocol, corpus)),
                  patch.object(benchmark, "training_groups",
                               return_value=([], {"groups": 0})),
                  patch.object(benchmark, "_new_models",
                               return_value=(models, fit_records, 0.0)),
                  patch.object(benchmark, "_environment",
                               side_effect=environment),
                  patch.object(benchmark, "_source_manifest",
                               side_effect=[{"source": "before"},
                                            {"source": "after"}])):
                manifest = benchmark._run(output, search)

            rows = [json.loads(line) for line in
                    (output / "raw.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 7)
            self.assertTrue(any(row["error"] for row in rows))
            self.assertTrue(any(
                not row["state_frozen_during_eval"] for row in rows
            ))
            self.assertFalse(manifest["infrastructure_valid"])
            saved = json.loads((output / "manifest.json").read_text())
            self.assertFalse(saved["source_frozen"])
            self.assertFalse(saved["infrastructure_valid"])
            errors = [item["error"] for item in
                      saved["infrastructure_errors"]]
            self.assertIn("runtime environment changed during run", errors)
            self.assertIn("source or frozen input bytes changed during run",
                          errors)
            self.assertIn("RuntimeError: record this attempt", errors)
            self.assertIn("policy state changed during eval", errors)

    def test_tiny_runner_detects_missing_grid_rows(self) -> None:
        """Run accounting rejects an omitted matrix combination."""
        expected = {
            (0, "streaming", "a", 1),
            (0, "heuristic", "a", 1),
        }
        self.assertTrue(benchmark._grid_valid(expected, 2, expected))
        self.assertFalse(benchmark._grid_valid(expected, 1, expected))
        self.assertFalse(benchmark._grid_valid(expected, 2, set()))

if __name__ == "__main__":
    unittest.main()
