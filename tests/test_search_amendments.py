"""Focused regressions for enumerator accounting and bounded alternatives."""
from __future__ import annotations

import unittest
from typing import Any

from rsi_programmer.probe import CMP, SELECT, Expr
from rsi_programmer.probe import enumerate_programs, enumerate_programs_report
from rsi_programmer.search import guided_search


class EnumerationReportTests(unittest.TestCase):
    @staticmethod
    def _expected_counts(
            records: list[tuple[Expr, tuple[int, ...], int]],
            found: Expr | None, root_op: str | None,
            deduplicate: bool) -> tuple[int, int | None]:
        examined = 0
        raw_position = 0
        previous_size: int | None = None
        seen: dict[int, set[tuple[str, tuple[int, ...]]]] = {}
        for expr, sig, _ordinal in records:
            if expr.size != previous_size:
                previous_size = expr.size
                raw_position = 0
            eligible = (expr.op not in CMP
                        and (root_op is None or expr.op == root_op))
            if eligible:
                raw_position += 1
                key = (expr.op, sig)
                size_seen = seen.setdefault(expr.size, set())
                keep = not deduplicate or key not in size_seen
                size_seen.add(key)
                if keep:
                    examined += 1
            if expr is found:
                return examined, raw_position
        return examined, None

    def test_report_separates_signature_work_from_match_position(self) -> None:
        report = enumerate_programs_report(
            ((1, 2), (2, 5)), (3, 7), 1)

        self.assertEqual(report["signature_evaluations"], 12)
        self.assertEqual(report["examined"], 4)
        self.assertEqual(report["first_match_position"], 4)
        self.assertEqual(report["cost_unit"],
                         "candidate_signature_evaluations")
        self.assertIsNone(report["budget_unit"])
        self.assertEqual(report["bound_unit"], "tree_operations")

    def test_branch_miss_excludes_bool_candidates_from_examined(self) -> None:
        records: list[tuple[Expr, tuple[int, ...], int]] = []

        def record(expr: Expr, sig: tuple[int, ...], ordinal: int) -> None:
            records.append((expr, sig, ordinal))

        report = enumerate_programs_report(
            ((1, 2), (2, 4)), (999, 999), 2, branch=True,
            on_candidate=record)
        expected, _position = self._expected_counts(
            records, None, None, True)

        self.assertIsNone(report["expr"])
        self.assertEqual(report["signature_evaluations"], len(records))
        self.assertEqual(report["examined"], expected)

    def test_select_root_and_no_dedup_callback_counters(self) -> None:
        rows = ((1, 2), (3, 2))
        expected = (2, 3)
        records: list[tuple[Expr, tuple[int, ...], int]] = []

        def record(expr: Expr, sig: tuple[int, ...], ordinal: int) -> None:
            records.append((expr, sig, ordinal))

        report = enumerate_programs_report(
            rows, expected, 2, root_op=SELECT, deduplicate=False,
            branch=True, on_candidate=record)
        reference: list[tuple[Expr, tuple[int, ...], int]] = []

        def capture(expr: Expr, sig: tuple[int, ...], ordinal: int) -> None:
            reference.append((expr, sig, ordinal))

        enumerate_programs(rows, expected, 2, root_op=SELECT,
                           deduplicate=False, branch=True,
                           on_candidate=capture)
        expected_examined, expected_position = self._expected_counts(
            records, report["expr"], SELECT, False)

        self.assertIsNotNone(report["expr"])
        self.assertEqual(report["expr"].op, SELECT)
        self.assertEqual(report["examined"], expected_examined)
        self.assertEqual(report["first_match_position"], expected_position)
        self.assertEqual(report["signature_evaluations"], len(records))
        self.assertEqual([item[2] for item in records],
                         list(range(1, len(records) + 1)))
        self.assertEqual(records, reference)


class LowerLayerAlternativeTests(unittest.TestCase):
    def test_signature_cap_keeps_complete_layer_feature_denominator(
            self) -> None:
        class Capture:
            def __init__(self) -> None:
                self.progress: list[float] = []

            def rank(self, feats: list[list[float]],
                     *args: Any) -> list[int]:
                self.progress = [row[-2] for row in feats]
                return list(range(len(feats)))

        policy = Capture()
        out = guided_search(policy, ((1, 2),), (997,), 1, 10,
                            signature_budget=3)
        self.assertFalse(out["solved"])
        self.assertEqual(policy.progress, [1 / 12, 2 / 12, 3 / 12])

    def test_capped_alternatives_recover_held_correct_candidate(self) -> None:
        class Ordered:
            def rank(self, feats: list[list[float]],
                     *args: Any) -> list[int]:
                return list(range(len(feats)))

        rows = ((1, 1), (2, 2), (3, 3))
        expected = (4, 16, 36)
        held = ((1, 2), (2, 1))

        def held_oracle(graph: dict[str, Any]) -> bool:
            from rsi_programmer.probe import interpret

            return [interpret(graph, x, y) for x, y in held] == [9, 9]

        baseline = guided_search(
            Ordered(), rows, expected, 3, 1000,
            oracle=held_oracle)
        self.assertFalse(baseline["solved"])
        self.assertTrue(baseline["equivalent_alternatives_truncated"])

        out = guided_search(
            Ordered(), rows, expected, 3, 1000,
            oracle=held_oracle, alternatives_per_signature=2)

        self.assertTrue(out["solved"])
        self.assertTrue(held_oracle(out["graph"]))
        self.assertEqual(out["equivalent_alternatives_limit"], 2)
        self.assertGreater(out["equivalent_alternatives_retained"], 0)
        self.assertGreater(out["equivalent_alternatives_dropped"], 0)

    def test_alternative_cap_reports_truncation(self) -> None:
        class Ordered:
            def rank(self, feats: list[list[float]],
                     *args: Any) -> list[int]:
                return list(range(len(feats)))

        out = guided_search(Ordered(), ((1, 1), (2, 2)), (2, 4),
                            1, 1000, alternatives_per_signature=1)

        self.assertIn("equivalent_alternatives_dropped", out)
        self.assertEqual(out["equivalent_alternatives_limit"], 1)

    def test_alternative_expansion_has_signature_work_budget(self) -> None:
        class Ordered:
            def rank(self, feats: list[list[float]],
                     *args: Any) -> list[int]:
                return list(range(len(feats)))

        out = guided_search(
            Ordered(), ((1, 1), (2, 2)), (4, 8), 2, 100,
            alternatives_per_signature=2,
            signature_budget=3)

        self.assertFalse(out["solved"])
        self.assertEqual(out["signature_evaluations"], 3)
        self.assertTrue(out["signature_budget_exhausted"])
        self.assertEqual(out["verdict"], "signature-budget-exhausted")

    def test_alternative_limits_require_integer_values(self) -> None:
        class Ordered:
            def rank(self, feats: list[list[float]],
                     *args: Any) -> list[int]:
                return list(range(len(feats)))

        invalid_caps: list[Any] = [True, 1.5, "2"]
        for cap in invalid_caps:
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                guided_search(Ordered(), ((1, 2),), (3,), 1, 10,
                              alternatives_per_signature=cap)
        invalid_budgets: list[Any] = [True, 1.5, "2"]
        for budget in invalid_budgets:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                guided_search(Ordered(), ((1, 2),), (3,), 1, 10,
                              signature_budget=budget)

    def test_zero_signature_budget_is_valid_and_does_no_work(self) -> None:
        class Ordered:
            def rank(self, feats: list[list[float]],
                     *args: Any) -> list[int]:
                return list(range(len(feats)))

        out = guided_search(
            Ordered(), ((1, 2),), (3,), 1, 10,
            alternatives_per_signature=2, signature_budget=0)

        self.assertFalse(out["solved"])
        self.assertEqual(out["signature_evaluations"], 0)
        self.assertTrue(out["signature_budget_exhausted"])
        self.assertEqual(out["verdict"], "signature-budget-exhausted")


if __name__ == "__main__":
    unittest.main()
