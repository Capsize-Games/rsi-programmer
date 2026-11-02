"""Fast mocked tests for campaign worker boundaries and audit claims."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_programmer import campaign_worker as worker


class CampaignWorkerTests(unittest.TestCase):
    def _recipe(self, seed: int) -> dict[str, object]:
        return {"execution_context": False, "seed": seed, "width": 2,
                "seconds": 0.1, "token_choices": 2}

    def _audit_spec(self) -> dict[str, object]:
        return {
            "schema_version": 1, "phase": "audit", "trial_index": 1,
            "held_seed": 12, "source_sha256": {},
            "audit_task_ids": ["fresh"],
            "audit_tasks": [{
                "task_id": "fresh", "family_id": "bounded_integer_list",
                "examples": [{"input": [1], "output": [1]}],
                "held_inputs": [[2]], "expected": [[2]],
            }],
        }

    def _model_result(
        self, _recipe: dict[str, object],
    ) -> tuple[object, dict[str, object], str]:
        class FakeModel:
            def state_dict(self) -> dict[str, object]:
                return {}

        return FakeModel(), {}, "fingerprint"

    def test_audit_choices_freeze_before_any_held_execution(self) -> None:
        events: list[str] = []

        def make_choices(_model: object, _recipe: dict[str, object],
                         tasks: list[dict[str, object]], _audit: bool
                         ) -> list[dict[str, object]]:
            self.assertEqual(tasks[0]["examples"], [
                {"input": [1], "output": [1]},
            ])
            events.append("choice")
            return [{"task_id": "fresh", "family_id":
                     "bounded_integer_list", "examples": [],
                     "chosen": {"tokens": ["for"]},
                     "controller_seconds": 0.0}]

        executions = 0

        def execute_task(_root: Path, _entry: dict[str, object],
                         task: dict[str, object], _seed: int,
                         _compiler: Path, audit: bool) -> dict[str, object]:
            nonlocal executions
            self.assertTrue(audit)
            events.append("held")
            executions += 1
            return {"task_id": task["task_id"],
                    "qualified": executions == 2}

        recipes = [self._recipe(1), self._recipe(2)]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            output = temporary_root / "worker"
            fake_package = temporary_root / "rsi_programmer"
            fake_package.mkdir()
            fake_module = fake_package / "campaign_worker.py"
            fake_module.touch()
            with patch.object(
                    worker, "_model", side_effect=self._model_result), \
                    patch.object(worker, "_make_choices",
                                 side_effect=make_choices), \
                    patch.object(worker, "_execute_task",
                                 side_effect=execute_task), \
                    patch.object(worker, "__file__", str(fake_module)), \
                    patch.object(worker, "state_fingerprint",
                                 return_value="fingerprint"), \
                    patch.object(worker, "digest", return_value="digest"), \
                    patch.object(worker.time, "monotonic", side_effect=[
                        0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                    ]):
                result = worker.run_campaign(
                    self._audit_spec(), output, audit=True,
                    baseline_recipe=recipes[0], selected_recipe=recipes[1])

        self.assertEqual(events, ["choice", "choice", "held", "held"])
        self.assertEqual(result["verdict"], "LOCAL_GAIN_ONLY")
        self.assertFalse(result["cross_family_evidence"])
        self.assertIsNone(result["retention_evidence"]["qualified"])

    def test_malformed_held_rows_fail_after_choices_freeze(self) -> None:
        spec = self._audit_spec()
        descriptor = spec["audit_tasks"][0]
        descriptor["expected"] = []
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            output = temporary_root / "worker"
            fake_package = temporary_root / "rsi_programmer"
            fake_package.mkdir()
            fake_module = fake_package / "campaign_worker.py"
            fake_module.touch()
            events: list[str] = []

            def choices(*_args: object) -> list[dict[str, object]]:
                events.append("choice")
                return [{
                    "task_id": "fresh",
                    "family_id": "bounded_integer_list",
                    "examples": [], "chosen": {"tokens": ["for"]},
                    "controller_seconds": 0.0,
                }]

            def validate_held(_root: Path, _entry: dict[str, object],
                              task: dict[str, object], seed: int,
                              _compiler: Path, audit: bool
                              ) -> dict[str, object]:
                self.assertTrue(audit)
                events.append("held-validation")
                worker._held(task, seed, audit)
                return {"qualified": False}

            with patch.object(
                    worker, "_model",
                    side_effect=self._model_result) as model, \
                    patch.object(worker, "_make_choices",
                                 side_effect=choices) as make_choices, \
                    patch.object(worker, "_execute_task",
                                 side_effect=validate_held) as execute_task, \
                    patch.object(worker, "__file__", str(fake_module)), \
                    patch.object(worker, "state_fingerprint",
                                 return_value="fingerprint"), \
                    patch.object(worker, "digest", return_value="digest"), \
                    patch.object(worker.time, "monotonic", return_value=0.0):
                with self.assertRaises(ValueError):
                    worker.run_campaign(
                        spec, output, audit=True,
                        baseline_recipe=self._recipe(1),
                        selected_recipe=self._recipe(2))
            self.assertEqual(model.call_count, 2)
            self.assertEqual(make_choices.call_count, 2)
            execute_task.assert_called_once()
            self.assertEqual(events, ["choice", "choice", "held-validation"])

    def test_malformed_recipe_fails_before_fit(self) -> None:
        spec: dict[str, object] = {
            "schema_version": 1, "phase": "development", "trial_index": 1,
            "held_seed": 1, "source_sha256": {},
            "development_task_ids": ["clip_negative_readings"],
            "recipe": {"seed": "wrong"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "worker"
            with patch.object(worker, "_model") as model:
                with self.assertRaises(ValueError):
                    worker.run_campaign(spec, output)
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
