"""Observed-only state for sequential typed program construction."""
from __future__ import annotations

import copy
from typing import Any

from rsi_programmer.actions import ActionSpace
from rsi_programmer.probe import OPS

Rows = tuple[tuple[int, int, int], ...]
Graph = dict[str, Any]
Action = dict[str, str | int]


class ConstructionState:
    """Execute chosen graph edits without receiving held expectations.

    Slots zero and one refer to x and y; later slots refer to appended
    nodes. Cached vectors contain outputs on observed rows only. STOP is
    an explicit controller action, not an automatic exact-match shortcut.
    """

    def __init__(self, observed: Rows, max_tree_ops: int = 4) -> None:
        if (not isinstance(observed, tuple) or not observed or any(
                not isinstance(row, tuple) or len(row) != 3
                or any(type(value) is not int for value in row)
                for row in observed)):
            raise ValueError("observed rows must be integer triples")
        if type(max_tree_ops) is not int or not 1 <= max_tree_ops <= 8:
            raise ValueError("tree bound must be an integer from 1 to 8")
        self._observed = observed
        self._bound = max_tree_ops
        self._graph: Graph = {
            "input_types": {"x": "Int", "y": "Int"},
            "nodes": [], "output": "x",
        }
        self._vectors = [tuple(row[0] for row in observed),
                         tuple(row[1] for row in observed)]
        self._costs = [0, 0]
        self._stopped = False

    @property
    def observed(self) -> Rows:
        """The public task examples; no task metadata or held rows."""
        return self._observed

    @property
    def bound(self) -> int:
        return self._bound

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def node_count(self) -> int:
        return len(self._vectors) - 2

    @property
    def output_slot(self) -> int:
        return self.node_count + 1 if self.node_count else 0

    def graph(self) -> Graph:
        """Return detached program data; caller edits cannot change state."""
        return copy.deepcopy(self._graph)

    def vectors(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self._vectors)

    def costs(self) -> tuple[int, ...]:
        return tuple(self._costs)

    def observed_match(self) -> bool:
        """Compare this partial program with the public specification."""
        return self._vectors[self.output_slot] == tuple(
            row[2] for row in self._observed
        )

    def apply(self, action: Action) -> ConstructionState:
        """Return a new state after one chosen operation or STOP."""
        if self._stopped:
            raise ValueError("cannot extend a stopped program")
        if not isinstance(action, dict):
            raise ValueError("construction action must be a mapping")
        if action == {"op": "stop"}:
            result = copy.deepcopy(self)
            result._stopped = True
            return result
        if set(action) != {"op", "left", "right"}:
            raise ValueError("malformed construction action")
        op = action["op"]
        if not isinstance(op, str) or op not in OPS:
            raise ValueError("unknown construction operation")
        left, right = action["left"], action["right"]
        for slot in (left, right):
            if type(slot) is not int or not 0 <= slot < len(self._vectors):
                raise ValueError("operand slot is unavailable")
        assert isinstance(left, int) and isinstance(right, int)
        cost = self._costs[left] + self._costs[right] + 1
        if cost > self._bound or self.node_count >= self._bound:
            raise ValueError("construction budget exhausted")
        refs: list[str | int] = ["x", "y", *range(self.node_count)]
        graph = ActionSpace().apply(self._graph, {
            "kind": "apply", "op": op,
            "left": refs[left], "right": refs[right],
        })
        result = ConstructionState(self._observed, self._bound)
        result._graph = graph
        result._vectors = [*self._vectors, tuple(
            OPS[op](a, b) for a, b in
            zip(self._vectors[left], self._vectors[right])
        )]
        result._costs = [*self._costs, cost]
        return result
