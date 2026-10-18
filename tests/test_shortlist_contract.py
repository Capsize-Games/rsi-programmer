"""Contract checks for complete-fallback shortlist layer semantics."""
from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from rsi_programmer import pre_eval, shortlist_pre_eval
from rsi_programmer.probe import Expr


def shape(expr: Expr | str | int) -> Any:
    """Return a structural key for an expression independent of object IDs."""
    if isinstance(expr, Expr):
        if expr.op in {"input", "const"}:
            return (expr.op, expr.left)
        return (expr.op, shape(expr.left), shape(expr.right))
    return expr


class ShortlistContractTests(unittest.TestCase):
    def test_unshortlisted_operand_is_retained_and_used_later(self) -> None:
        rows = ((1, 2),)
        expected = (10,)
        selected_shapes: list[list[Any]] = []

        class CapturePolicy:
            def rank(self, features: list[list[float]], triples: list[Any],
                     _gap: float) -> list[int]:
                selected_shapes.append([
                    (op, shape(left), shape(right))
                    for op, left, right in triples
                ])
                return list(range(len(triples)))

        original_expr = shortlist_pre_eval.Expr
        created: list[Expr] = []

        def make_expr(*args: Any, **kwargs: Any) -> Expr:
            expr = original_expr(*args, **kwargs)
            created.append(expr)
            return expr

        later_layer_uses: list[tuple[int, int]] = []
        original_iter = pre_eval._iter_triples

        def record_iter(levels: dict[int, list[Expr]], size: int):
            for triple in original_iter(levels, size):
                if size == 2:
                    _op, left, right = triple
                    later_layer_uses.append((id(left), id(right)))
                yield triple

        with patch.object(
                shortlist_pre_eval, "Expr", side_effect=make_expr), \
                patch.object(
                    pre_eval, "_iter_triples", side_effect=record_iter):
            result = shortlist_pre_eval.shortlist_pre_evaluation_search(
                CapturePolicy(), rows, expected, 2, 10_000,
                window_size=12, shortlist_size=1)

        self.assertFalse(result["solved"])
        self.assertEqual(len(selected_shapes[0]), 1)
        required_shape = (
            "mul", ("input", "x"), ("input", "y"))
        self.assertNotIn(required_shape, selected_shapes[0])
        required = next(
            expr for expr in created
            if expr.size == 1 and expr.op == "mul"
            and shape(expr.left) == ("input", "x")
            and shape(expr.right) == ("input", "y")
        )
        self.assertTrue(any(
            id(required) in pair for pair in later_layer_uses))

    def test_completed_layers_match_streaming_generation_order(self) -> None:
        rows = ((1, 2),)
        expected = (10**9,)

        def trace_search(
                shortlist: bool
                ) -> tuple[dict[str, Any], dict[int, list[Any]],
                           dict[int, list[tuple[str, int, int]]],
                           list[tuple[str, int, int]], int]:
            layers: dict[int, list[Any]] = {}
            generated_ids: dict[int, list[tuple[str, int, int]]] = {}
            evaluated: list[tuple[str, int, int]] = []
            candidate_calls = 0
            original_iter = pre_eval._iter_triples
            original_pre_expr = pre_eval.Expr
            original_short_expr = shortlist_pre_eval.Expr
            original_candidate = pre_eval._candidate_vector

            def record_iter(levels: dict[int, list[Expr]], size: int):
                for op, left, right in original_iter(levels, size):
                    layers.setdefault(size, []).append(
                        (op, shape(left), shape(right)))
                    generated_ids.setdefault(size, []).append(
                        (op, id(left), id(right)))
                    yield op, left, right

            def record_candidate(op: str, left: tuple[int, ...],
                                 right: tuple[int, ...]) -> tuple[int, ...]:
                nonlocal candidate_calls
                candidate_calls += 1
                return original_candidate(op, left, right)

            def make_pre_expr(op: str, left: Any, right: Any = None,
                              size: int = 0, third: Any = None) -> Expr:
                evaluated.append((op, id(left), id(right)))
                return original_pre_expr(op, left, right, size, third)

            def make_short_expr(op: str, left: Any, right: Any = None,
                                size: int = 0,
                                third: Any = None) -> Expr:
                evaluated.append((op, id(left), id(right)))
                return original_short_expr(op, left, right, size, third)

            if shortlist:
                class ReversePolicy:
                    def rank(self, features: list[list[float]], *args: Any
                             ) -> list[int]:
                        return list(reversed(range(len(features))))

                search = shortlist_pre_eval.shortlist_pre_evaluation_search
                call = lambda: search(
                    ReversePolicy(), rows, expected, 3, 1_000_000,
                    window_size=64, shortlist_size=16)
            else:
                search = pre_eval.pre_evaluation_search
                call = lambda: search(
                    None, rows, expected, 3, 1_000_000)
            with patch.object(pre_eval, "_iter_triples",
                              side_effect=record_iter), \
                    patch.object(pre_eval, "_candidate_vector",
                                 side_effect=record_candidate), \
                    patch.object(pre_eval, "Expr",
                                 side_effect=make_pre_expr), \
                    patch.object(shortlist_pre_eval, "Expr",
                                 side_effect=make_short_expr):
                result = call()
            self.assertEqual(candidate_calls, result["signature_evaluations"])
            generated = [key for layer in generated_ids.values()
                         for key in layer]
            self.assertEqual(len(generated), len(set(generated)))
            generated_counts = {key: generated.count(key)
                                for key in set(generated)}
            evaluated_counts = {key: evaluated.count(key)
                                for key in set(evaluated)}
            self.assertEqual(set(evaluated_counts), set(generated_counts))
            if shortlist:
                self.assertEqual(evaluated_counts, generated_counts)
            else:
                self.assertTrue(all(
                    generated_counts[key] <= evaluated_counts[key] <= 2
                    for key in generated_counts))
                self.assertEqual(
                    len(evaluated), result["signature_evaluations"]
                    + result["retained_operand_vectors"])
            return (result, layers, generated_ids, evaluated,
                    candidate_calls)

        stream, stream_layers, _, _, _ = trace_search(False)
        shortlist, shortlist_layers, _, _, _ = trace_search(True)
        self.assertFalse(stream["solved"])
        self.assertFalse(shortlist["solved"])
        self.assertEqual(stream_layers, shortlist_layers)
        stream_widths = [len(stream_layers[size])
                         for size in sorted(stream_layers)]
        self.assertEqual(stream_widths, shortlist["layer_widths"])
        self.assertEqual(stream_widths,
                         shortlist["layer_generated_descriptors"])
        self.assertEqual(
            stream["retained_operand_vectors"],
            shortlist["retained_operand_vectors"])
        self.assertEqual(shortlist["permanently_pruned_descriptors"], 0)
        self.assertEqual(shortlist["descriptor_replays"], 0)


if __name__ == "__main__":
    unittest.main()
