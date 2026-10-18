"""Focused corruption checks for the independent online result auditor."""
from __future__ import annotations

import copy
import hashlib
import unittest
from pathlib import Path
from typing import Any

from rsi_programmer.online_graph import (
    check_result,
    iter_candidates,
    run_episode,
)
from rsi_programmer.probe import interpret
from scripts.audit_online_graph import (
    _phase_counts,
    audit_episode_row,
    verify_source_hashes,
)


def _fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one valid streaming attempt without invoking the benchmark."""
    graph = next(iter_candidates("add"))
    observed = ((-2, 3, interpret(graph, -2, 3)),
                (4, -1, interpret(graph, 4, -1)))
    held = ((1, 5, interpret(graph, 1, 5)),)
    out = run_episode(None, observed, budget=1, episode_id="audit-test")
    verdict = check_result(out["graph"], held, ())
    task = {"id": "audit-test", "phase": 0, "root": "add",
            "observed": [list(row) for row in observed],
            "held": [list(row) for row in held], "polynomial": []}
    row = {"arm": "grammar_streaming", "seed": 0, "budget": 1,
           "task_id": "audit-test", "phase": 0, "root": "add",
           "first_lane_correct": out["attempts"][0]["action"] == "add",
           "policy_work": {}, "weight_before": None,
           "weight_after": None, "reset_cpu_seconds": 0.0,
           **out, **verdict}
    return row, task


class OnlineAuditTests(unittest.TestCase):
    """Ensure tampering with raw attempt evidence is rejected."""

    def test_accepts_valid_small_streaming_episode(self) -> None:
        row, task = _fixture()
        audit_episode_row(row, task, 1, None)

    def test_rejects_corrupted_episode_records(self) -> None:
        row, task = _fixture()
        changes = (
            lambda data: data["attempts"][0]["features"].__setitem__(4, 0.0),
            lambda data: data["attempts"][0].__setitem__("reward", 0.25),
            lambda data: data["attempts"][0]["graph"]["nodes"][0]
            .__setitem__("value", 99),
            lambda data: data["attempts"].clear(),
            lambda data: data.__setitem__("node_row_operations", 1),
            lambda data: data.__setitem__("episode_cpu_seconds", -1.0),
        )
        for mutate in changes:
            with self.subTest(mutation=mutate):
                corrupt = copy.deepcopy(row)
                mutate(corrupt)
                with self.assertRaises(ValueError):
                    audit_episode_row(corrupt, task, 1, None)

    def test_rejects_source_hash_mismatch(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source_files: dict[str, str] = {}
        for directory in ("rsi_programmer", "tests", "scripts", "fixtures"):
            for path in (root / directory).rglob("*"):
                if path.is_file() and path.suffix in {".py", ".json", ".js"}:
                    source_files[str(path.relative_to(root))] = hashlib.sha256(
                        path.read_bytes()).hexdigest()
        for name in ("requirements.txt", "justfile",
                     ".github/workflows/ci.yml"):
            path = root / name
            if path.is_file():
                source_files[name] = hashlib.sha256(
                    path.read_bytes()).hexdigest()
        source_files["requirements.txt"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            verify_source_hashes(source_files, root)

    def test_rejects_missing_source_manifest_entry(self) -> None:
        with self.assertRaisesRegex(ValueError, "source manifest path set"):
            verify_source_hashes({}, Path(__file__).resolve().parents[1])

    def test_recomputes_phase_counts(self) -> None:
        rows = [{"phase": 0, "accepted": True,
                 "first_lane_correct": False},
                {"phase": 1, "accepted": False,
                 "first_lane_correct": True}]
        self.assertEqual(_phase_counts(rows), [
            {"phase": 0, "accepted": 1, "first_lane_correct": 0},
            {"phase": 1, "accepted": 0, "first_lane_correct": 1},
            {"phase": 2, "accepted": 0, "first_lane_correct": 0},
        ])


if __name__ == "__main__":
    unittest.main()
