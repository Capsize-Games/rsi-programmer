"""Regression contracts for search accounting and feature encoding."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from rsi_programmer.actions import ActionSpace
from rsi_programmer.author import _label_layer, _new_ctx, _score_order
from rsi_programmer.author import beam_search
from rsi_programmer.lattice import BitLattice
from rsi_programmer.ltrain import fit_packs, pack_rows
from rsi_programmer.modules import ModularPolicy
from rsi_programmer.policy import SearchPolicy, _label_final, layer_triples
from rsi_programmer.probe import interpret, terminals, validate_graph
from rsi_programmer.sandbox import SandboxRunner
from rsi_programmer.search import guided_search


class SearchReviewTests(unittest.TestCase):
    def test_work_scan_budget_and_minimal_graph(self) -> None:
        class Ordered:
            def rank(self, feats: list[list[float]],
                     *args: Any) -> list[int]:
                return list(range(len(feats)))
        rows = ((1, 2), (2, 5))
        miss = guided_search(Ordered(), rows, (3, 7), 1, 1)
        self.assertFalse(miss["solved"])
        self.assertEqual(miss["tried"], 1)
        self.assertEqual(miss["examined"], 1)
        self.assertEqual(miss["signature_evaluations"], 12)
        self.assertEqual(miss["budget_unit"], "candidate_scans")
        hit = guided_search(Ordered(), rows, (3, 7), 1, 4)
        self.assertTrue(hit["solved"])
        self.assertEqual(hit["examined"], 4)
        self.assertEqual(hit["signature_evaluations"], 12)
        self.assertEqual(len(hit["graph"]["nodes"]), 1)
        self.assertEqual(interpret(hit["graph"], 3, 4), 7)
        zero = guided_search(Ordered(), rows, (3, 7), 1, 0)
        self.assertFalse(zero["solved"])
        self.assertEqual(zero["signature_evaluations"], 0)

    def test_grammar_miss_is_not_budget_exhaustion(self) -> None:
        class Ordered:
            def rank(self, feats: list[list[float]],
                     *args: Any) -> list[int]:
                return list(range(len(feats)))
        out = guided_search(Ordered(), ((1, 2),), (999,), 1, 100)
        self.assertEqual(out["verdict"], "no-solution-within-bound")

    def test_training_and_scoring_share_layer_ordinals(self) -> None:
        class Capturing:
            feats: list[list[float]]
            def rank(self, feats: list[list[float]],
                     *args: Any) -> list[int]:
                self.feats = feats
                return list(range(len(feats)))
            def scores(self, feats: list[list[float]]) -> list[float]:
                self.feats = feats
                return [0.0] * len(feats)
        rows, target = ((1, 2), (2, 5)), (3, 7)
        triples = layer_triples({0: list(terminals())}, 1)
        trained = _label_final(triples, rows, target, 1, 1, 999, 1234)
        expected = [f for f, _ in trained]
        labelled = _label_layer(1, triples, rows, target, 1,
                                999, 10 ** 9, {target})
        self.assertEqual(expected, [f for f, _ in labelled])
        p = Capturing()
        guided_search(p, rows, target, 1, 100)
        self.assertEqual(p.feats, expected)
        ctx = _new_ctx()
        ctx["tried"] = 19
        _score_order(p, ctx, rows, target, 1, 1, 10 ** 9, [0.] * 4)
        self.assertEqual(p.feats, expected)
        self.assertEqual(expected[-1][-2:], [1.0, 0.0])

    def test_bonus_roundtrip_and_rank_use_one_score_contract(self) -> None:
        p = SearchPolicy(seed=0)
        feats = [[float(i == j) for j in range(12)] for i in range(6)]
        p.set_rich_bonus([1., 2., 3., 4.])
        scores = p.scores(feats)
        self.assertEqual(p.rank(feats), sorted(range(6),
                         key=scores.__getitem__, reverse=True))
        twin = SearchPolicy(seed=7)
        twin.restore(p.snapshot())
        self.assertEqual(scores, twin.scores(feats))
        self.assertEqual(p.rank(feats), twin.rank(feats))

    def test_restore_training_moves_all_live_networks(self) -> None:
        feats = [[float(i == j) for j in range(12)] for i in range(6)]
        for cls in (SearchPolicy, ModularPolicy, BitLattice):
            with self.subTest(network=cls.__name__):
                p = cls(seed=0, lr=.06)
                p._opt.param_groups[0]["weight_decay"] = .01
                p.restore(p.snapshot())
                before = {k: v.clone() for k, v in p.weights().items()}
                if cls is SearchPolicy:
                    p.fit([[([0.] * 12, 1), ([1.] * 12, 0)]], 2)
                elif cls is ModularPolicy:
                    p.fit([{"feats": feats, "pos": [0, 1, 2],
                            "neg": [3, 4, 5],
                            "ops": [0, 1, 2, 0, 1, 2], "gap": 1.}], 2)
                else:
                    pack = pack_rows([(f, i % 2)
                                      for i, f in enumerate(feats)])
                    fit_packs(p, [pack], 2, .2)
                self.assertTrue(any(not torch.equal(v, before[k])
                                    for k, v in p.weights().items()))
                self.assertEqual(p._opt.param_groups[0]["lr"], .06)
                self.assertEqual(p._opt.param_groups[0]["weight_decay"],
                                 .01)

    def test_bool_intermediates_and_int_final_output(self) -> None:
        graph: dict[str, Any] = {"input_types": {"x": "Int", "y": "Int"},
                                 "nodes": [], "output": "x"}
        graph = ActionSpace().apply(graph, {"kind": "apply", "op": "lt",
                                          "left": "x", "right": "y"})
        with self.assertRaisesRegex(ValueError, "output must be Int"):
            validate_graph(graph)
        validate_graph(graph, require_int_output=False)
        with tempfile.TemporaryDirectory() as tmp:
            graph["nodes"][0]["op"] = []
            out = SandboxRunner(Path(tmp)).run_candidate(graph, [[1, 2]])
            self.assertEqual(out["status"], "rejected")

    def test_beam_reports_signature_work_and_fallback_attempts(self) -> None:
        class Ordered:
            def scores(self, feats: list[list[float]]) -> list[float]:
                return [0.] * len(feats)
            def rank(self, feats: list[list[float]],
                     *args: Any) -> list[int]:
                return list(range(len(feats)))
        out = beam_search(Ordered(), ((1, 2), (2, 5)), (3, 7),
                          1, 100, width=1, fallback=False)
        self.assertTrue(out["solved"])
        self.assertGreaterEqual(out["signature_evaluations"], 12)
        self.assertEqual(out["committed_nodes"], out["tried"])
        self.assertEqual(out["tried_unit"], "committed_nodes")


if __name__ == "__main__":
    unittest.main()
