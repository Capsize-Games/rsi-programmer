"""Loop-body transfer preserves qualified provenance and outer scope."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import cast
import unittest
from unittest.mock import Mock

import torch

from rsi_programmer.statement_body_contexts import mine, qualify_bodies
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_fragment_search import synthesize
from rsi_programmer.statement_policy import StatementPolicy
from rsi_programmer.statement_syntax import TOKENS
from rsi_programmer.stateful_tasks import WITNESSES

ROOT = Path(__file__).resolve().parents[1]
DONORS = ROOT / "artifacts/statement-programs-v1-execution-20261101"


class StatementBodyTests(unittest.TestCase):
    """A hole can alter updates without losing donor execution evidence."""

    def test_qualified_donors_share_one_body_context(self) -> None:
        library = qualify_bodies(DONORS)
        self.assertEqual(len(library["condition_fragments"]), 3)
        self.assertEqual(len(library["fragments"]), 1)
        context = library["fragments"][0]
        self.assertEqual(context["donors"], sorted(library["donors"]))
        for name in ("maximum_floor_zero", "minimum_ceiling_zero"):
            supplied = tuple(WITNESSES[name].split())
            rebuilt = (*context["prefix"], *supplied[1:-4],
                       *context["suffix"])
            self.assertEqual(rebuilt, supplied)
            self.assertEqual(execute(rebuilt, [-2, 5, 0]),
                             5 if name.startswith("maximum") else -2)

    def test_outer_scope_survives_body_replacement(self) -> None:
        source = tuple("set one for if gt value acc set value end end "
                       "set add acc one return acc stop".split())
        context, = mine([("scoped", source)])
        rebuilt = (*context["prefix"], "set", "sub", "acc", "one",
                   *context["suffix"])
        self.assertEqual(execute(rebuilt, [4, 8, -1]), -1)
        self.assertEqual(execute(rebuilt, []), 2)

    def test_a_program_without_a_loop_has_no_body_to_transfer(self) -> None:
        self.assertEqual(mine([("constant", ("return", "one", "stop"))]), [])

    def test_changed_source_cannot_qualify_body_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "donors"
            shutil.copytree(DONORS, root)
            path = root / "odd_count/python/clean_readings.py"
            path.write_text(path.read_text() + "\n# Changed after checking.\n")
            with self.assertRaises(ValueError):
                qualify_bodies(root)

    def test_all_choices_can_recover_a_pruned_legal_operand(self) -> None:
        # The fifth-ranked leaf is the only fitting simple operand. This
        # tests search reachability, not a claim about a trained network.
        model = Mock(spec=StatementPolicy)
        model.observed_features.return_value = torch.zeros((1, 51))
        order = ["zero", "one", "two", "minus_one", "acc"]

        def scores(matrix: torch.Tensor,
                   prefix: torch.Tensor) -> torch.Tensor:
            logits = torch.full((len(prefix), len(TOKENS)), -20.0)
            for rank, token in enumerate(order):
                logits[:, TOKENS.index(token)] = -float(rank)
            return logits

        model.side_effect = scores
        context = {"id": "operand-test", "donors": ["test"],
                   "prefix": ("for", "if", "gt", "value"),
                   "suffix": ("set", "value", "end", "end", "return",
                              "acc", "stop")}
        examples = [([8, 6], 8)]
        policy = cast(StatementPolicy, model)
        limited = synthesize(policy, examples, [context])
        expanded = synthesize(policy, examples, [context],
                              choices_per_prefix=None)
        self.assertIsNone(limited["tokens"])
        self.assertEqual(execute(expanded["tokens"], [9, 1, 7]), 9)
