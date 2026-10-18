"""Focused regressions from the 2026-10-17 integrity review."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

from rsi_programmer.bytecode import HALT, STORE, binary_holds
from rsi_programmer.evolve import _baseline_cost, _fitkey, _front
from rsi_programmer.evolve import _search_cost
from rsi_programmer.evolve import PolicyLoop
from rsi_programmer.lineage import LineageError, LineageStore
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.proxy import ProxyScreen
from rsi_programmer.tasks import _size_of
from rsi_programmer.probe import Expr
from rsi_programmer.words import decode, oracle_holds_words


class IntegrityReviewTests(unittest.TestCase):
    def test_lineage_appends_run_scoped_records_and_checks_digest(
            self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lineage.sqlite"
            first = LineageStore(path)
            written = first.record_intent("same", "mission", 0, 1,
                                          {"kind": "first"})
            first.close()
            second = LineageStore(path)
            second.record_intent("same", "mission", 0, 1,
                                 {"kind": "second"})
            rows = second.commands_in("mission", "prepared")
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({r["command_id"] for r in rows}), 2)
            self.assertIn(written["digest"], {r["digest"] for r in rows})
            second.close()
            conn = sqlite3.connect(path)
            conn.execute("UPDATE commands SET digest='bad'")
            conn.commit()
            conn.close()
            store = LineageStore(path)
            with self.assertRaises(LineageError):
                store.commands_in("mission", "prepared")
            store.close()

    def test_unmeasured_entries_never_win_evolution_selection(
            self) -> None:
        good = {"fitness": {"correctness": 1, "refinability": 0,
                            "efficiency": .5, "recovery": .5}}
        skipped = {"fitness": {"correctness": 1, "refinability": 0,
                               "efficiency": 1, "recovery": 1},
                   "screen_skipped": True}
        self.assertLess(_fitkey(skipped), _fitkey(good))
        self.assertEqual(_front([skipped, good]), [good])
        self.assertIs(max([skipped, good], key=_fitkey), good)

    def test_skipped_child_has_no_measured_fitness_in_archive(
            self) -> None:
        class Policy:
            def snapshot(self) -> dict[str, Any]:
                return {"weights": {}, "spec": {}}

        class Screen:
            def rank_policies(self, policies: dict[str, Any]
                              ) -> dict[str, float]:
                return {key: 1.0 for key in policies}

            def fidelity(self, proxy: dict[str, float],
                         full: dict[str, float]) -> dict[str, Any]:
                return {"agreement": 1.0, "pairs": len(full)}

        class Ledger:
            def __init__(self) -> None:
                self.entries: list[dict[str, Any]] = []

            def record_intent(self, command: str, mission: str, term: int,
                              gen: int, intent: dict[str, Any]) -> None:
                self.entries.append(intent)

        loop = object.__new__(PolicyLoop)
        loop._screen = Screen()
        loop._stop_after = 1
        loop._archive = []
        loop._ledger = Ledger()

        def evaluate(policy: Policy) -> dict[str, Any]:
            del policy
            return {
                "gates_pass": True, "correctness": 1.0,
                "efficiency": .5, "recovery": .5,
                "refinability": 0.0}

        loop.evaluate = evaluate
        parents = [{"id": "p0", "fitness": {
            "gates_pass": True, "correctness": 1.0,
            "efficiency": .5, "recovery": .5,
            "refinability": 0.0}}]
        pop = [{"id": f"c{i}", "policy": Policy(), "parent": "p0",
                "operator": "perturb"} for i in range(3)]
        loop._evaluate_pop(pop, parents, 1)
        skipped = [entry for entry in loop._archive
                   if entry["screen_skipped"]]
        self.assertEqual(len(skipped), 2)
        for entry in skipped:
            self.assertIsNone(entry["fitness"])
            self.assertFalse(entry["measured"])
            self.assertIsNotNone(entry["inherited_fitness"])
        self.assertEqual(skipped, [entry for entry in loop._ledger.entries
                                   if entry["screen_skipped"]])

    def test_evolution_cost_prefers_canonical_signature_counter(
            self) -> None:
        self.assertEqual(_search_cost({"tried": 12,
                                       "signature_evaluations": 7}), 7)
        self.assertEqual(_search_cost({"tried": 12}), 12)
        self.assertEqual(_baseline_cost({"tried": 12,
                                         "signature_evaluations": 7}), 7)

    def test_concurrent_duplicate_intent_is_inserted_once(
            self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LineageStore(Path(tmp) / "lineage.sqlite")
            def append(kind: str) -> dict[str, Any] | None:
                try:
                    return store.record_intent(
                        "same", "mission", 0, 1, {"kind": kind})
                except LineageError:
                    return None

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(append, ["first", "second"]))
            self.assertEqual(sum(result is not None
                                 for result in results), 1)
            rows = store.commands_in("mission", "prepared")
            self.assertEqual(len(rows), 1)
            store.close()

    def test_bytecode_vm_errors_and_trailing_material_reject(
            self) -> None:
        task = {"train": [[1, 2, 3]], "held_out": []}
        self.assertFalse(binary_holds(task, bytes([STORE, HALT])))
        self.assertFalse(binary_holds(task, bytes([HALT, HALT])))

    def test_word_decoder_checks_operands_and_stop_boundary(
            self) -> None:
        task = {"train": [[1, 2, 3]], "held_out": []}
        # add operand references must be in the word alphabet.
        self.assertFalse(oracle_holds_words(task, [0, 999, 1, 8]))
        self.assertFalse(oracle_holds_words(task, [8, 7]))
        with self.assertRaises(ValueError):
            decode([0, 999, 1, 8])

    def test_proxy_uses_configured_seed_offsets(self) -> None:
        calls: list[int] = []

        class FakePolicy:
            pass

        def trained(seed: int, groups: list[Any],
                    epochs: int) -> FakePolicy:
            del groups, epochs
            calls.append(seed)
            return FakePolicy()

        # Build only the screen state needed by screen(), bypassing fixture IO.
        screen = object.__new__(ProxyScreen)

        def groups() -> dict[str, list[str]]:
            return {"a": ["group-a"], "b": ["group-b"]}

        screen._groups_by_task = groups
        screen._shadow_ids = ["a", "b"]
        screen._subset = [{"id": "a"}]
        screen._hold = [{"id": "b"}]
        screen._epochs = 1
        screen._seeds = (10, 20, 30)
        screen._shadow = "unused"
        screen._novel = "unused"
        screen._threshold = .8
        screen._min_pairs = 3
        import rsi_programmer.proxy as proxy_module
        prior_trained = proxy_module._trained
        prior_gain = proxy_module._mean_gain
        proxy_module._trained = trained
        proxy_module._mean_gain = lambda policy, tasks: 0.0
        try:
            screen.screen([{"id": "c", "seed": 100}])
        finally:
            proxy_module._trained = prior_trained
            proxy_module._mean_gain = prior_gain
        self.assertEqual(calls, [110, 120, 130])

    def test_proxy_subsets_trace_groups_by_task_id(self) -> None:
        screen = object.__new__(ProxyScreen)
        screen._shadow_ids = ["a", "b"]
        screen._subset = [{"id": "b"}, {"id": "a"}]

        def groups() -> dict[str, list[str]]:
            return {"a": ["group-a"], "b": ["group-b"]}

        screen._groups_by_task = groups
        self.assertEqual(screen._subset_groups(),
                         [["group-b"], ["group-a"]])

    def test_proxy_rejects_trace_group_count_mismatch(self) -> None:
        screen = object.__new__(ProxyScreen)
        screen._shadow_ids = ["a", "b"]
        screen._subset = [{"id": "a"}]

        def incomplete() -> dict[str, list[str]]:
            return {"a": ["group-a"]}

        screen._groups_by_task = incomplete
        with self.assertRaisesRegex(ValueError, "match"):
            screen._subset_groups()

    def test_trace_groups_bind_records_by_task_id(self) -> None:
        class Recorder:
            def __init__(self, fixture: str) -> None:
                del fixture

            def load_family(self) -> dict[str, Any]:
                return {"tasks": [{"id": "a"}, {"id": "b"}]}

            def record_all(self) -> dict[str, Any]:
                return {"records": [{"task": "b", "group": "B"},
                                    {"task": "a", "group": "A"}]}

        def fake_group(policy: SearchPolicy, task: dict[str, Any],
                       frozen: dict[str, dict[str, Any]]) -> list[str]:
            del policy
            return [frozen[str(task["id"])]["group"]]

        policy = SearchPolicy(seed=0)
        with patch("rsi_programmer.policy.TraceRecorder", Recorder), \
                patch.object(SearchPolicy, "_task_group", fake_group):
            groups = policy.trace_groups_by_task("fixture")
        self.assertEqual(groups, {"a": ["A"], "b": ["B"]})

    def test_rebuilt_expression_size_counts_each_operation_once(
            self) -> None:
        leaf = Expr("input", "x", None, 0)
        child = Expr("add", leaf, leaf, 1)
        self.assertEqual(_size_of("mul", child, leaf, None), 2)


if __name__ == "__main__":
    unittest.main()
