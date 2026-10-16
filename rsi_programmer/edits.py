#!/usr/bin/env python3
"""Typed trainer-edit operators for self-edit proposals.

Edits are data, never code: each names one trainer field and one of
its enumerated values. Applying an edit returns a new config and
never mutates the caller's. The alphabet structurally excludes the
judge: no operator can name the gate, sandbox, lineage, validator,
or oracle, and this module imports none of them.
"""
from __future__ import annotations

from typing import Any

FENCED = ("promote", "sandbox", "lineage", "validate", "oracle",
          "gate")

FIELDS: dict[str, tuple[Any, ...]] = {
    "hidden": (8, 16, 32),
    "steps": (4, 6, 8),
    "lr": (0.01, 0.05, 0.1),
    "beta": (0.8, 0.9, 0.95),
    "margin": (0.5, 1.0, 2.0),
    "joint_lambda": (0.0, 0.2, 0.5),
    "joint_epochs": (3, 5, 10),
    "pop": (2, 4, 8),
    "sigma": (0.02, 0.05, 0.1, 0.2),
    "generations": (1, 3, 5),
    "evolve_budget_s": (120, 240, 480),
}

DEFAULT_CONFIG: dict[str, Any] = {
    "hidden": 16, "steps": 6, "lr": 0.05, "beta": 0.9,
    "margin": 1.0, "joint_lambda": 0.2, "joint_epochs": 5,
    "pop": 4, "sigma": 0.05, "generations": 3,
    "evolve_budget_s": 240,
}


def validate_edit(edit: dict[str, Any]) -> None:
    """Reject malformed, unknown, or out-of-range edits."""
    if not isinstance(edit, dict):
        raise ValueError("edit must be a mapping")
    if set(edit) != {"op", "field", "value"}:
        raise ValueError("malformed edit")
    if edit.get("op") != "set":
        raise ValueError("unknown operator")
    field = edit.get("field")
    if field not in FIELDS:
        raise ValueError("unknown field")
    lowered = str(field).lower()
    if any(token in lowered for token in FENCED):
        raise ValueError("fenced field")
    if edit.get("value") not in FIELDS[field]:
        raise ValueError("value out of range")


def apply_edit(config: dict[str, Any],
               edit: dict[str, Any]) -> dict[str, Any]:
    """Validated edit applied to a copy; input never mutates."""
    validate_edit(edit)
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    field = str(edit["field"])
    if field not in config:
        raise ValueError("field missing from config")
    out = dict(config)
    out[field] = edit["value"]
    return out


def invert(edit: dict[str, Any],
           before: dict[str, Any]) -> dict[str, Any]:
    """The edit restoring the field's pre-edit value."""
    validate_edit(edit)
    field = str(edit["field"])
    return {"op": "set", "field": field,
            "value": before[field]}


def describe(edit: dict[str, Any],
             before: dict[str, Any]) -> str:
    """Human diff line for lineage records."""
    validate_edit(edit)
    field = str(edit["field"])
    return "%s: %s -> %s" % (field, before.get(field),
                             edit["value"])


def all_edits(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Every non-trivial edit from the current config."""
    edits: list[dict[str, Any]] = []
    for field, values in FIELDS.items():
        for value in values:
            if config.get(field) != value:
                edits.append({"op": "set", "field": field,
                              "value": value})
    return edits
