"""Contract tests for the frozen bounded-list edit-value evaluation."""
from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.check_edit_value import BANK, verify, verify_pre_fit
from scripts.measure_edit_value import (
    _baseline_fingerprint, _corpus, _fresh_signatures, _source_hashes, run,
)
from rsi_programmer.array_policy import ArrayPolicy
from rsi_programmer.edit_value_ranker import EditValueRanker
import scripts.measure_edit_value as measurement


class EditValueEvaluationTests(unittest.TestCase):
    def test_frozen_bank_and_checker_references_are_consistent(self) -> None:
        result = verify()
        protocol_sha = (
            "2c7dc36c284df6edfcafa44b6d4e3aeeeeaa67413c0d889fde0304beac441a1b")
        examples_sha = (
            "592593be8df66a61958a8f89afd3432902520ee3d7f65bf202e0b62cee668ffa")
        held_sha = (
            "02ebde24e29044e320aac2a25e05e64331e473d928538f2b1a6e2badf9cff1e5")
        self.assertEqual(result["fresh_tasks"], 6)
        self.assertEqual(result["fresh_signatures_distinct"], 6)
        self.assertEqual(result["held_cases_per_task"], 20)
        self.assertEqual(result["protocol_sha256"], protocol_sha)
        self.assertEqual(result["examples_sha256"], examples_sha)
        self.assertEqual(result["held_cases_sha256"], held_sha)

    def test_pre_fit_contract_does_not_open_held_rows(self) -> None:
        from unittest.mock import patch

        original = Path.read_text

        def guarded(path: Path, *args: object, **kwargs: object) -> str:
            if path.name == "held_cases.json":
                raise AssertionError("held rows were opened before choices")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", guarded):
            self.assertEqual(verify_pre_fit()["fresh_tasks"], 6)

    def test_training_pool_excludes_fresh_signatures_and_pins_sources(
            self) -> None:
        protocol = json.loads((BANK / "protocol.json").read_text())
        from rsi_programmer.array_curriculum import probes

        excluded = _fresh_signatures(protocol, probes())
        corpus = _corpus(excluded, 64)
        signatures = {str(row["signature"]) for row in corpus}
        self.assertTrue(corpus)
        self.assertFalse(signatures & excluded)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.pt"
            compiler = root / "compiler"
            guard = root / "guard.py"
            model = ArrayPolicy(seed=0, execution_context=True)
            torch.save(model.state_dict(), baseline)
            compiler.write_bytes(b"mock compiler artifact")
            guard.write_text("# mocked test resource guard\n")
            with patch("scripts.measure_edit_value.BASELINE", baseline), \
                    patch("scripts.measure_edit_value.COMPILER", compiler), \
                    patch("scripts.measure_edit_value.CURLEE_GUARD", guard):
                source_hashes = _source_hashes()
            self.assertIn("rsi_programmer/edit_value_ranker.py",
                          source_hashes)
            self.assertIn("baseline_checkpoint", source_hashes)
            self.assertIn("curlee_compiler", source_hashes)
            self.assertIn("curlee_resource_guard", source_hashes)
            artifact_paths = {
                "BASELINE": baseline,
                "COMPILER": compiler,
                "CURLEE_GUARD": guard,
            }
            for key in artifact_paths:
                missing = dict(artifact_paths)
                missing[key] = root / f"missing-{key}"
                with ExitStack() as stack:
                    for name, path in missing.items():
                        stack.enter_context(patch(
                            f"scripts.measure_edit_value.{name}", path))
                    with self.assertRaises(FileNotFoundError):
                        _source_hashes()

    def test_fresh_structures_are_not_claimed_as_new_families(self) -> None:
        protocol = json.loads((BANK / "protocol.json").read_text())
        self.assertEqual(protocol["development_tasks"], [
            "clip_negative_readings", "cap_positive_readings",
            "negative_magnitudes"])
        self.assertEqual(protocol["baseline_recheck_tasks"], [
            "clip_negative_readings", "cap_positive_readings",
            "negative_magnitudes"])
        self.assertEqual(protocol["retention_tasks"], [])
        self.assertIn("not_measured_for_edit_value_ranker",
                      protocol["retention_status"])
        self.assertTrue(all(task["task_family"] == "bounded_integer_list"
                            for task in protocol["fresh_tasks"]))
        self.assertEqual(len({task["task_type"]
                              for task in protocol["fresh_tasks"]}), 6)
        self.assertIn("Strings", protocol["claims"][2])

    def test_solver_visible_examples_are_separate_from_held_rows(self) -> None:
        examples = json.loads(
            (BANK / "fresh_examples.json").read_text())
        held = json.loads((BANK / "held_cases.json").read_text())
        self.assertEqual({row["task_id"] for row in examples},
                         {row["task_id"] for row in held})
        self.assertTrue(all(set(row) == {
            "task_id", "task_family", "task_type", "examples"}
            for row in examples))
        self.assertTrue(all(set(row) == {
            "task_id", "held_inputs", "expected"} for row in held))

    def test_run_smoke_persists_fit_before_held_read(self) -> None:
        examples = json.loads(
            (BANK / "fresh_examples.json").read_text())[0]
        program = ("for", "append", "value", "end", "return",
                   "output", "stop")
        baseline = ArrayPolicy(seed=0, execution_context=True)
        baseline_fingerprint = _baseline_fingerprint(baseline)

        def completed(*args: object, **kwargs: object) -> dict[str, object]:
            return {"program": program, "solved_examples": False,
                    "proof": "example_fit_only", "traces": [],
                    "actions_scored": 1, "cpu_seconds": 0.001}

        def fake_fit(self: EditValueRanker, *args: object,
                     **kwargs: object) -> dict[str, object]:
            return {"fit_cpu_seconds": 0.001, "transitions": 1}

        source_hashes = {"stub_source": "fixed"}
        metadata = {"model_fingerprint": baseline_fingerprint,
                    "checkpoint": "baseline.pt"}
        held_inputs_opened = False
        original_read_text = Path.read_text

        def guarded_read(path: Path, *args: object,
                         **kwargs: object) -> str:
            nonlocal held_inputs_opened
            if path.name == "held_cases.json":
                self.assertTrue((run_root / "choices-frozen.json").is_file())
                held_inputs_opened = True
            return original_read_text(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            baseline_path = temporary_root / "baseline.pt"
            compiler_path = temporary_root / "compiler"
            guard_path = temporary_root / "guard.py"
            torch.save(baseline.state_dict(), baseline_path)
            compiler_path.write_bytes(b"mock compiler artifact")
            guard_path.write_text("# mocked test resource guard\n")
            baseline_hash = hashlib.sha256(
                baseline_path.read_bytes()).hexdigest()
            run_root = Path(temporary) / "run"
            patches = [
                patch("scripts.measure_edit_value.verify_pre_fit",
                      return_value={"fresh_tasks": 6}),
                patch("scripts.measure_edit_value.verify",
                      return_value={"held_cases_per_task": 20}),
                patch("scripts.measure_edit_value._source_hashes",
                      return_value=source_hashes),
                patch("scripts.measure_edit_value._fresh_signatures",
                      return_value={"excluded"}),
                patch("scripts.measure_edit_value._corpus",
                      return_value=[{"signature": "train", "pairs": [],
                                     "programs": []}]),
                patch("scripts.measure_edit_value._baseline_metadata",
                      return_value=metadata),
                patch("scripts.measure_edit_value._sha",
                      side_effect=lambda path: (
                          baseline_hash
                          if path == baseline_path else hashlib.sha256(
                              Path(path).read_bytes()).hexdigest())),
                patch("scripts.measure_edit_value._task_sets",
                      return_value=([{"task_id": "dev_recheck",
                                      "task_type": "development",
                                      "family_id": "bounded_integer_list",
                                      "examples": [{"input": [1],
                                                    "output": [1]}]}],
                                    [{"task_id": examples["task_id"],
                                      "task_type": examples["task_type"],
                                      "family_id": examples["task_family"],
                                      "examples": examples["examples"]}])),
                patch("scripts.measure_edit_value.WEIGHTS_DIR",
                      Path(temporary) / "repo" / "weights"),
                patch("scripts.measure_edit_value.ROOT",
                      Path(temporary) / "repo"),
                patch("scripts.measure_edit_value.BASELINE", baseline_path),
                patch("scripts.measure_edit_value.BASELINE_SHA256",
                      baseline_hash),
                patch("scripts.measure_edit_value.COMPILER", compiler_path),
                patch("scripts.measure_edit_value.CURLEE_GUARD", guard_path),
                patch.object(EditValueRanker, "fit", fake_fit),
                patch.object(EditValueRanker, "synthesize", completed),
                patch.object(EditValueRanker, "untrained_synthesize",
                             completed),
                patch.object(EditValueRanker, "classical_synthesize",
                             completed),
                patch.object(ArrayPolicy, "synthesize", completed),
                patch("scripts.measure_edit_value._execute_task",
                      side_effect=lambda root, entry, task, seed, compiler,
                      audit: {"task_id": task["task_id"],
                              "qualified": False}),
                patch.object(Path, "read_text", guarded_read),
            ]
            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                result = run(run_root, training_tasks=1, transitions=1,
                             epochs=1, rollouts=1, rollout_steps=1)
            self.assertTrue(held_inputs_opened)
            self.assertEqual(result["status"], "completed")
            self.assertTrue((run_root / "fit-traces.json").is_file())
            relative = Path(result["fit_artifacts"]["checkpoint_path"])
            checkpoint = temporary_root / "repo" / relative
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                result["fit_artifacts"]["checkpoint_sha256"])
            self.assertTrue((run_root / "choices-frozen.json").is_file())


if __name__ == "__main__":
    unittest.main()
