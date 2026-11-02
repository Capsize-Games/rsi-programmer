"""Tests for rollout-labeled edit-value learning."""
from __future__ import annotations

import unittest

import torch

from rsi_programmer.edit_value_ranker import EditValueRanker, _state_features


class EditValueRankerTests(unittest.TestCase):
    def test_fit_keeps_success_and_failure_transition_traces(self) -> None:
        teacher = tuple("for append value append value end return "
                        "output stop".split())
        corpus = [{
            "signature": "train-duplicate-v1",
            "pairs": [([1, 2], [1, 1, 2, 2]), ([0], [0, 0])],
            "programs": [teacher],
        }]
        model = EditValueRanker(seed=17)
        report = model.fit(corpus, excluded_signatures={"sealed-v1"},
                           epochs=2, rollout_count=1, rollout_steps=12)
        self.assertEqual(report["fitted_signatures"],
                         ("train-duplicate-v1",))
        self.assertGreater(report["transitions"], 0)
        labeled = [trace for trace in model.traces
                   if trace["label"] is not None]
        self.assertTrue(any(trace["label"] == 1.0 for trace in labeled))
        self.assertTrue(any(trace["label"] == 0.0 for trace in labeled))
        self.assertTrue(any(trace["status"] == "censored"
                            for trace in model.traces))
        self.assertGreater(max(trace["depth"] for trace in model.traces),
                           len(teacher) // 2)
        self.assertGreater(report["depth_bin_histogram"]["late"], 0)
        self.assertEqual(report["seed"], 17)
        self.assertEqual(len(report["trace_sha256"]), 64)

    def test_fit_rejects_all_excluded_corpus(self) -> None:
        corpus = [{"signature": "held", "pairs": [([1], [1])]}]
        with self.assertRaisesRegex(ValueError, "empty after exclusions"):
            EditValueRanker().fit(corpus, excluded_signatures={"held"})

    def test_synthesis_reports_only_example_fit(self) -> None:
        corpus = [{
            "signature": "train-identity-v1",
            "pairs": [([1], [1]), ([-2], [-2])],
            "programs": [tuple(("for append value end return output stop")
                               .split())],
        }]
        model = EditValueRanker(seed=3)
        model.fit(corpus, epochs=2, rollout_count=1, rollout_steps=12)
        result = model.synthesize(corpus[0]["pairs"], budget=40,
                                  beam_width=8)
        self.assertEqual(result["proof"], "example_fit_only")
        self.assertLessEqual(result["actions_scored"], 40)
        self.assertTrue(result["traces"])
        if result["program"] is not None:
            self.assertTrue(result["solved_examples"])

    def test_features_condition_on_requested_outputs(self) -> None:
        corpus = [{
            "signature": "train-features-v1",
            "pairs": [([1], [1])],
            "programs": [tuple(("for append value end return output stop")
                               .split())],
        }]
        left = _state_features((), "for", corpus[0]["pairs"])
        changed_targets = [([1], [999])]
        right = _state_features((), "for", changed_targets)
        self.assertNotEqual(left, right)

    def test_future_outcomes_are_labels_not_feature_fields(self) -> None:
        corpus = [{
            "signature": "train-label-boundary-v1",
            "pairs": [([1], [1])],
            "programs": [tuple(("for append value end return output stop")
                               .split())],
        }]
        model = EditValueRanker(seed=4)
        model.fit(corpus, epochs=2, rollout_count=1, rollout_steps=12,
                  max_transitions=24)
        self.assertTrue(all("features" not in trace
                            for trace in model.traces))
        self.assertTrue(any(trace["label"] == 1.0
                            for trace in model.traces
                            if trace["label"] is not None))

    def test_classical_control_uses_same_action_budget(self) -> None:
        pairs = [([1], [1])]
        model = EditValueRanker(seed=8)
        learned = model.synthesize(pairs, budget=12, beam_width=4)
        classical = model.classical_synthesize(
            pairs, budget=12, beam_width=4)
        self.assertLessEqual(learned["actions_scored"], 12)
        self.assertLessEqual(classical["actions_scored"], 12)
        self.assertIn("cpu_seconds", classical)

    def test_constructor_preserves_process_rng(self) -> None:
        state = torch.random.get_rng_state().clone()
        EditValueRanker(seed=99)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
