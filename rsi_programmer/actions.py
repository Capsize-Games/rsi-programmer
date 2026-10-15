#!/usr/bin/env python3
"""Typed program-DAG edit action space for SNN RSI shadow stage.

Mirrors the grammar of scripts/rsi_programmer.probe.py,
which stays untouched: binary Int ops (add/sub/mul) over inputs
x/y. Every edit is validated by the probe typechecker before it
is accepted into a graph.
"""
from __future__ import annotations

from typing import cast

from rsi_programmer.probe import CONSTS, OPS, validate_graph


class ActionSpace:
    """Typed edit actions covering each enumerator decision."""

    def __init__(self) -> None:
        """Bind the probe grammar operation and input names."""
        self._ops: tuple[str, ...] = tuple(OPS)
        self._inputs: tuple[str, str] = ("x", "y")

    def ops(self) -> tuple[str, ...]:
        """Return the grammar operation names in fixed order."""
        return self._ops

    def inputs(self) -> tuple[str, str]:
        """Return the typed input references of the DAG."""
        return self._inputs

    def input_actions(self) -> list[dict[str, str]]:
        """List the terminal emit decisions of the enumerator."""
        return [{"kind": "input", "ref": ref} for ref in self._inputs]

    def const_actions(self) -> list[dict[str, object]]:
        """List the constant emit decisions over the IR range."""
        return [{"kind": "const", "value": value}
                for value in CONSTS]

    def apply_actions(
        self, refs: list[str | int]
    ) -> list[dict[str, str | int]]:
        """List every apply decision over op, left, and right."""
        choices: list[dict[str, str | int]] = []
        for op in self._ops:
            for left in refs:
                for right in refs:
                    choices.append({"kind": "apply", "op": op,
                                    "left": left, "right": right})
        return choices

    def _check_ref(self, ref: object, bound: int) -> None:
        """Reject references outside the inputs or built nodes."""
        if isinstance(ref, str):
            if ref not in self._inputs:
                raise ValueError("unknown input reference")
            return
        if type(ref) is not int or ref < 0 or ref >= bound:
            raise ValueError("invalid node reference")

    def _extended(
        self, graph: dict[str, object], action: dict[str, object]
    ) -> dict[str, object]:
        """Return the graph with the apply/const edit appended."""
        nodes = list(cast(list[dict[str, object]], graph["nodes"]))
        if action.get("kind") == "const":
            nodes.append({"args": [], "op": "const",
                          "type": "Int", "value": action["value"]})
        else:
            nodes.append({"args": [action["left"], action["right"]],
                          "op": cast(str, action["op"]), "type": "Int"})
        return {"input_types": {"x": "Int", "y": "Int"},
                "nodes": nodes, "output": len(nodes) - 1}

    def typecheck(
        self, graph: dict[str, object], action: dict[str, object]
    ) -> None:
        """Validate an edit against the typed grammar."""
        kind = action.get("kind")
        if kind == "input":
            if action.get("ref") not in self._inputs:
                raise ValueError("unknown input emit")
            return
        if kind == "const":
            if (set(action) != {"kind", "value"}
                    or action.get("value") not in CONSTS):
                raise ValueError("constant out of range")
            validate_graph(self._extended(graph, action))
            return
        keys = {"kind", "left", "op", "right"}
        if kind != "apply" or set(action) != keys:
            raise ValueError("malformed action")
        if action.get("op") not in OPS:
            raise ValueError("unknown operation")
        nodes = graph.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("malformed graph")
        self._check_ref(action.get("left"), len(nodes))
        self._check_ref(action.get("right"), len(nodes))
        validate_graph(self._extended(graph, action))

    def apply(
        self, graph: dict[str, object], action: dict[str, object]
    ) -> dict[str, object]:
        """Typecheck an edit, then return the accepted graph."""
        self.typecheck(graph, action)
        if action.get("kind") == "input":
            return cast(dict[str, object], graph)
        return self._extended(graph, action)
