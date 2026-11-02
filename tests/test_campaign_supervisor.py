"""Fast contract tests for durable campaign supervision."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from rsi_programmer import campaign_supervisor as supervisor


class CampaignSupervisorTests(unittest.TestCase):
    def _config(self, **overrides: object) -> dict[str, object]:
        config: dict[str, object] = {
            "worker": ["fake-worker"],
            "development_task_ids": ["task-a"],
            "recipes": [{"execution_context": False}],
            "max_trials": 2,
            "max_wall_seconds": 1000,
            "max_stagnant_trials": 10,
            "trial_timeout_seconds": 1,
            "retry_count": 0,
            "retry_backoff_seconds": 0,
            "poll_seconds": 0,
            "min_available_gib": 2,
            "stop_available_gib": 1,
            "audit_task_ids": [],
        }
        config.update(overrides)
        return config

    def _campaign(self, directory: Path, config: dict[str, object],
                  results: list[dict[str, object]] | None = None,
                  started_at: float | None = None) -> None:
        directory.mkdir()
        config_hash = hashlib.sha256(
            supervisor._canonical(config)).hexdigest()
        supervisor._atomic_json(directory / "campaign.json", {
            "config": config, "config_sha256": config_hash,
            "source_sha256": {},
        })
        supervisor._atomic_json(directory / "status.json", {
            "state": "RUNNING", "started_at": started_at or time.time(),
            "results": results or [], "best_score": None,
            "stagnant_trials": 0,
        })

    def test_resume_skips_completed_trials_without_duplicate_runs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "campaign"
            prior: dict[str, object] = {
                "status": "completed", "trial_index": 1,
                "recipe": {"execution_context": False},
                "development": {"qualified_tasks": 1},
            }
            self._campaign(directory, self._config(), [prior])
            (directory / "trial-001").mkdir()
            calls: list[int] = []

            def run_trial(_directory: Path, _config: dict[str, object],
                          spec: dict[str, object], trial_dir: Path,
                          audit: bool = False) -> dict[str, object]:
                self.assertFalse(audit)
                calls.append(int(spec["trial_index"]))
                trial_dir.mkdir()
                return {
                    "status": "completed", "trial_index": 2,
                    "recipe": spec["recipe"],
                    "development": {"qualified_tasks": 1},
                }

            with patch.object(
                    supervisor, "_source_manifest", return_value={}), \
                    patch.object(supervisor, "_run_trial",
                                 side_effect=run_trial):
                supervisor._execute(directory, self._config())

            self.assertEqual(calls, [2])
            state = json.loads((directory / "status.json").read_text())
            self.assertEqual(len(state["results"]), 2)
            self.assertEqual(state["state"], "LOCAL_GAIN_ONLY")

    def test_stop_and_wall_budget_have_distinct_terminal_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stopped = Path(temporary) / "stopped"
            self._campaign(stopped, self._config())
            (stopped / "STOP").touch()
            with patch.object(supervisor, "_source_manifest", return_value={}):
                supervisor._execute(stopped, self._config())
            self.assertEqual(json.loads((stopped / "status.json").read_text())[
                "state"], "STOPPED")

            expired = Path(temporary) / "expired"
            config = self._config(max_wall_seconds=0)
            self._campaign(expired, config, started_at=-10)
            with patch.object(
                    supervisor, "_source_manifest", return_value={}), \
                    patch.object(supervisor.time, "time", return_value=100):
                supervisor._execute(expired, config)
            self.assertEqual(json.loads((expired / "status.json").read_text())[
                "state"], "WALL_LIMIT")

    def test_low_memory_admission_times_out_before_worker_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            supervisor._atomic_json(directory / "status.json", {
                "started_at": time.time(),
            })
            trial_dir = directory / "trial-001"
            config = self._config(memory_wait_seconds=0)
            with patch.object(
                    supervisor, "_available_gib", return_value=0.5), \
                    patch.object(supervisor.subprocess, "Popen") as popen:
                result = supervisor._run_trial(
                    directory, config,
                    {"trial_index": 1, "recipe": {}}, trial_dir,
                )
            self.assertEqual(result["status"], "memory_admission_timeout")
            popen.assert_not_called()

    def test_resource_drop_stops_running_worker(self) -> None:
        class RunningWorker:
            pid = 1
            polls = iter([None, 0])

            def __init__(self, *_args: object, **_options: object):
                pass

            def poll(self) -> int | None:
                return next(self.polls)

            def wait(self, **_options: object) -> int:
                return 0

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            supervisor._atomic_json(directory / "status.json", {
                "started_at": time.time(),
            })
            config = self._config(min_available_gib=1,
                                  stop_available_gib=5)
            with patch.object(supervisor, "_available_gib",
                              side_effect=[10, 4, 4]), \
                    patch.object(supervisor.subprocess, "Popen",
                                 RunningWorker), \
                    patch.object(supervisor.os, "killpg"):
                result = supervisor._run_trial(
                    directory, config,
                    {"trial_index": 1, "recipe": {}},
                    directory / "trial-001",
                )
            self.assertEqual(result["status"], "resource_stop")

    def test_malformed_worker_result_fails_closed(self) -> None:
        class MalformedWorker:
            pid = 1

            def __init__(self, command: list[str], **_options: object):
                output = Path(command[command.index("--output") + 1])
                (output / "result.json").write_text("{broken")

            def poll(self) -> int:
                return 0

            def wait(self) -> int:
                return 0

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            supervisor._atomic_json(directory / "status.json", {
                "started_at": time.time(),
            })
            trial_dir = directory / "trial-001"
            config = self._config(min_available_gib=1,
                                  stop_available_gib=0)
            with patch.object(supervisor, "_available_gib", return_value=2), \
                    patch.object(supervisor.subprocess, "Popen",
                                 MalformedWorker):
                result = supervisor._run_trial(
                    directory, config,
                    {"trial_index": 1, "recipe": {}}, trial_dir,
                )
            self.assertEqual(result["status"], "infrastructure_failure")
            self.assertIn("invalid result.json", result["error"])

    def test_fake_worker_cli_result_is_collected(self) -> None:
        script = ("from pathlib import Path\n"
                  "import json, sys\n"
                  "args = sys.argv\n"
                  "spec = json.loads(Path(args[args.index('--spec') + 1])"
                  ".read_text())\n"
                  "out = Path(sys.argv[sys.argv.index('--output') + 1])\n"
                  "ids = spec['development_task_ids']\n"
                  "rows = [{'task_id': task, 'qualified': False} "
                  "for task in ids]\n"
                  "result = {'status': 'completed', 'phase': spec['phase'], "
                  "'trial_index': spec['trial_index'], 'recipe': "
                  "spec['recipe'], 'source_sha256': spec['source_sha256'], "
                  "'development': {'qualified_tasks': 0, 'task_count': "
                  "len(ids), 'per_task': rows}}\n"
                  "(out / 'result.json').write_text(json.dumps(result))\n")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            supervisor._atomic_json(directory / "status.json", {
                "started_at": time.time(),
            })
            worker = directory / "fake_worker.py"
            worker.write_text(script)
            spec = {"schema_version": 1, "phase": "development",
                    "trial_index": 1, "recipe": {"seed": 1},
                    "development_task_ids": ["task-a"],
                    "source_sha256": {}}
            config = self._config(worker=[sys.executable, str(worker)],
                                  min_available_gib=1,
                                  stop_available_gib=0)
            with patch.object(supervisor, "_available_gib", return_value=10):
                result = supervisor._run_trial(
                    directory, config, spec, directory / "trial-001",
                )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["development"]["qualified_tasks"], 0)

    def test_invalid_recovery_receipt_is_not_scored(self) -> None:
        class ReceiptWorker:
            pid = 1

            def __init__(self, command: list[str], **_options: object):
                output = Path(command[command.index("--output") + 1])
                spec = json.loads(Path(
                    command[command.index("--spec") + 1]).read_text())
                rows = [{"task_id": item, "qualified": False}
                        for item in spec["development_task_ids"]]
                receipt = {
                    "status": "completed", "phase": spec["phase"],
                    "trial_index": spec["trial_index"],
                    "recipe": spec["recipe"],
                    "source_sha256": spec["source_sha256"],
                    "development": {"qualified_tasks": 0,
                                    "task_count": len(rows),
                                    "per_task": rows},
                }
                (output / "result.json").write_text(json.dumps(receipt))

            def poll(self) -> int:
                return 0

            def wait(self, **_options: object) -> int:
                return 0

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            supervisor._atomic_json(directory / "status.json", {
                "started_at": time.time(),
            })
            trial_dir = directory / "trial-001"
            trial_dir.mkdir()
            (trial_dir / "result.json").write_text(json.dumps({
                "status": "completed", "phase": "audit",
                "trial_index": 99,
            }))
            spec = {"schema_version": 1, "phase": "development",
                    "trial_index": 1, "recipe": {"seed": 1},
                    "development_task_ids": ["task-a"],
                    "source_sha256": {}}
            config = self._config(min_available_gib=1,
                                  stop_available_gib=0)
            with patch.object(supervisor, "_available_gib", return_value=10), \
                    patch.object(supervisor.subprocess, "Popen",
                                 ReceiptWorker):
                result = supervisor._run_trial(
                    directory, config, spec, trial_dir,
                )
            self.assertEqual(result["status"], "completed")
            self.assertNotIn("recovered_receipt", result)
            self.assertTrue((directory / "trial-001-attempt-1" /
                             "result.json").exists())

    def test_retry_uses_a_fresh_worker_output_directory(self) -> None:
        output_dirs: list[Path] = []
        return_codes = iter([1, 0])

        class RetryWorker:
            pid = 1

            def __init__(self, command: list[str], **_options: object):
                output = Path(command[command.index("--output") + 1])
                output_dirs.append(output)
                if len(output_dirs) == 2:
                    spec = json.loads(Path(
                        command[command.index("--spec") + 1]).read_text())
                    rows = [{"task_id": item, "qualified": False}
                            for item in spec["development_task_ids"]]
                    receipt = {
                        "status": "completed", "phase": spec["phase"],
                        "trial_index": spec["trial_index"],
                        "recipe": spec["recipe"],
                        "source_sha256": spec["source_sha256"],
                        "development": {"qualified_tasks": 0,
                                        "task_count": len(rows),
                                        "per_task": rows},
                    }
                    (output / "result.json").write_text(json.dumps(receipt))
                self.return_code = next(return_codes)

            def poll(self) -> int:
                return self.return_code

            def wait(self, **_options: object) -> int:
                return self.return_code

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            supervisor._atomic_json(directory / "status.json", {
                "started_at": time.time(),
            })
            spec = {"schema_version": 1, "phase": "development",
                    "trial_index": 1, "recipe": {"seed": 1},
                    "development_task_ids": ["task-a"],
                    "source_sha256": {}}
            config = self._config(min_available_gib=1,
                                  stop_available_gib=0, retry_count=1)
            with patch.object(supervisor, "_available_gib", return_value=10), \
                    patch.object(supervisor.subprocess, "Popen", RetryWorker):
                result = supervisor._run_trial(
                    directory, config, spec, directory / "trial-001",
                )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(output_dirs), 2)
            self.assertNotEqual(output_dirs[0], output_dirs[1])
            self.assertTrue((output_dirs[0] / "worker.log").exists())
            self.assertTrue((output_dirs[1] / "worker.log").exists())

    def test_audit_local_gain_remains_local_and_uses_fresh_task_ids(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = self._config(audit_task_ids=["fresh-task"],
                                  baseline_recipe={"seed": 1})
            config_hash = hashlib.sha256(
                supervisor._canonical(config)).hexdigest()
            supervisor._atomic_json(directory / "campaign.json", {
                "config": config, "config_sha256": config_hash,
                "source_sha256": {},
            })
            results: list[dict[str, object]] = [{
                "status": "completed", "trial_index": 1,
                "recipe": {"execution_context": True},
                "development": {"qualified_tasks": 1},
            }]
            audit_specs: list[dict[str, object]] = []

            def run_trial(_directory: Path, _config: dict[str, object],
                          spec: dict[str, object], _trial_dir: Path,
                          audit: bool = False) -> dict[str, object]:
                self.assertTrue(audit)
                audit_specs.append(spec)
                return {"status": "completed", "verdict": "LOCAL_GAIN_ONLY",
                        "conclusion": "No fresh task gain."}

            with patch.object(
                    supervisor, "_source_manifest", return_value={}), \
                    patch.object(supervisor, "_run_trial",
                                 side_effect=run_trial):
                supervisor._finalize(directory, config, results, 1)

            self.assertEqual(audit_specs[0]["audit_task_ids"], ["fresh-task"])
            self.assertNotIn("development_task_ids", audit_specs[0])
            state = json.loads((directory / "status.json").read_text())
            self.assertEqual(state["state"], "LOCAL_GAIN_ONLY")
            self.assertEqual(state["audit"]["verdict"], "LOCAL_GAIN_ONLY")


if __name__ == "__main__":
    unittest.main()
