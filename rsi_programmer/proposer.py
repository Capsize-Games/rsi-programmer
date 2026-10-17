"""Diagnostic-conditioned proposer over the edit alphabet.

The loop enumerates blind seeded-order edits; this module
ranks the same alphabet from lattice/beam diagnostics so
trials spend first where the signals point. Every proposal
is validate_edit-checked before it may reach the gate, and
each carries provenance naming its driving signal. The
fence holds structurally: candidates come only from
all_edits, which cannot name fenced fields.
"""
from __future__ import annotations

from typing import Any

from rsi_programmer.edits import FIELDS, all_edits, validate_edit

Diagnostics = dict[str, Any]
Proposal = dict[str, Any]

CAPACITY = ("hidden", "steps", "pop", "generations")
COUPLING = ("joint_lambda", "joint_epochs", "margin")


class EditProposer:
    """Rank edit candidates from diagnostic signals."""

    def _order(self, diag: Diagnostics
               ) -> list[tuple[str, str, str]]:
        """Field priority as (field, rule, signal) triples."""
        order: list[tuple[str, str, str]] = []

        def add(fields: Any, rule: str, signal: str) -> None:
            for field in fields:
                if field in FIELDS and field not in [
                        named for named, _, _ in order]:
                    order.append((field, rule, signal))

        add(diag.get("suspect_fields", []), "suspect",
            "suspect-fields=%s" % diag.get("suspect_fields",
                                           []))
        add(diag.get("promotion_history", []), "repeat",
            "prior-promotions=%s" % diag.get(
                "promotion_history", []))
        wrong = diag.get("agreed_wrong", 0)
        if wrong:
            add(COUPLING, "coupling",
                "agreed-wrong=%s" % wrong)
        hist = diag.get("reject_histogram", {})
        top = max(hist, key=lambda key: hist[key]) if hist \
            else ""
        named = diag.get("named_rate", 1.0)
        if top == "no-authored-solution" or named < 0.5:
            add(CAPACITY, "capacity",
                "rejects=%s named=%.3f" % (top, named))
        add(FIELDS, "baseline", "alphabet-order")
        return order

    def rank(self, config: dict[str, Any],
             diag: Diagnostics) -> list[Proposal]:
        """Validate-checked proposals, best signal first."""
        order = self._order(diag)
        place = {field: num
                 for num, (field, _, _) in enumerate(order)}
        info = {field: (rule, signal)
                for field, rule, signal in order}
        pool = all_edits(dict(config))
        pool.sort(key=lambda edit: place[str(edit["field"])])
        out: list[Proposal] = []
        for edit in pool:
            validate_edit(edit)
            rule, signal = info[str(edit["field"])]
            out.append({"edit": edit,
                        "provenance": {"rule": rule,
                                       "signal": signal}})
        return out
