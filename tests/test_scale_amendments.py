"""Focused contracts for scale surrogate evidence integration."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import rsi_programmer.scale as scale_module
from rsi_programmer.scale import ScaleUp


def _fake_search_policy(seed: int) -> SimpleNamespace:
    _ = seed
    return SimpleNamespace(trace_groups=lambda fixture:
                           [[([0.0], 1), ([1.0], 0)]])


def _scale_stub() -> ScaleUp:
    loop = ScaleUp.__new__(ScaleUp)
    loop._seed = 3
    loop._shadow = "shadow.json"
    loop._ha_tasks = [{"id": "hard-a"}]
    loop._base = {"hard": {"tasks": {
        "hard-a": {"signature_evaluations": 12}}}}
    loop._exprs = {"hard-a": object()}
    loop._surrogate_evidence = []
    loop._hard_training_group_count = 0
    loop._hard_selection_group_count = 0
    loop._rich_fitted_group_count = 0
    return loop


class ScaleAmendmentTests(unittest.TestCase):
    def test_surrogate_evidence_flows_to_report(self) -> None:
        loop = _scale_stub()
        evidence = [{"task": "hard-a", "included": True,
                     "negative_pairs": 7,
                     "training_negative_pairs": 4,
                     "selection_negative_pairs": 3}]
        hard_train = [[([0.0], 1), ([1.0], 0)]]
        rich_train = [[([0.0], 1), ([1.0], 0)]]
        selection = [[[1.0], [0.0]]]

        def pairs(*_args: object, diagnostics: list | None = None
                  ) -> tuple[list, list, list]:
            assert diagnostics is not None
            diagnostics.extend(evidence)
            return hard_train, rich_train, selection

        with patch.object(scale_module, "SearchPolicy",
                          _fake_search_policy), \
                patch.object(scale_module, "family_pairs", pairs):
            groups, extra = loop.surrogate_groups()

        self.assertEqual(groups[-1], hard_train[0])
        self.assertEqual(extra, rich_train)
        self.assertEqual(loop._sel, selection)
        self.assertEqual(loop._surrogate_evidence, evidence)
        self.assertEqual(loop._hard_training_group_count, 1)
        self.assertEqual(loop._hard_selection_group_count, 1)

        loop._rich = [0.1, 0.2, 0.3, 0.4]
        loop._rich_fitted_group_count = 1
        fidelity = loop._consts({"stopped": False})["fidelity"]
        report = fidelity["surrogate_evidence"]
        self.assertEqual(report["tasks"], evidence)
        self.assertEqual(report["training_negative_pairs"], 4)
        self.assertEqual(report["selection_negative_pairs"], 3)
        self.assertEqual(report["rich_fitted_group_count"], 1)

    def test_positive_only_or_empty_selection_fails_early(self) -> None:
        loop = _scale_stub()

        def no_evidence(*_args: object, diagnostics: list | None = None
                        ) -> tuple[list, list, list]:
            assert diagnostics is not None
            diagnostics.append({"task": "hard-a", "included": False,
                                "reason": "reachability_truncated"})
            return ([[([0.0], 1)]], [[([0.0], 1)]], [[[0.0]]])

        with patch.object(scale_module, "SearchPolicy",
                          _fake_search_policy), \
                patch.object(scale_module, "family_pairs", no_evidence):
            with self.assertRaisesRegex(ValueError,
                                        "complete hard training groups"):
                loop.surrogate_groups()

    def test_valid_training_without_selection_fails(self) -> None:
        loop = _scale_stub()

        def no_selection(*_args: object, diagnostics: list | None = None
                         ) -> tuple[list, list, list]:
            assert diagnostics is not None
            diagnostics.append({"task": "hard-a", "included": True,
                                "negative_pairs": 2,
                                "training_negative_pairs": 2,
                                "selection_negative_pairs": 0})
            return ([[([0.0], 1), ([1.0], 0)]],
                    [[([0.0], 1), ([1.0], 0)]], [])

        with patch.object(scale_module, "SearchPolicy",
                          _fake_search_policy), \
                patch.object(scale_module, "family_pairs", no_selection):
            with self.assertRaisesRegex(ValueError,
                                        "non-vacuous hard selection"):
                loop.surrogate_groups()

    def test_prepare_rejects_positive_only_rich_fit(self) -> None:
        loop = _scale_stub()
        groups = [[([0.0], 1), ([1.0], 0)]]
        positive_only = [[([0.0], 1)]]
        with patch.object(ScaleUp, "surrogate_groups",
                          return_value=(groups, positive_only)), \
                patch.object(scale_module, "fit_rich",
                             return_value=[0.0, 0.0, 0.0, 0.0]):
            with self.assertRaisesRegex(ValueError,
                                        "non-vacuous rich fit group"):
                loop._prepare()


if __name__ == "__main__":
    unittest.main()
