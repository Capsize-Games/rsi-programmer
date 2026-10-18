"""Focused contract tests for the v2 benchmark runner."""
from __future__ import annotations

import hashlib
from typing import Any


def _tiny_task() -> dict[str, Any]:
    return {
        "id": "tiny",
        "family": "balanced",
        "train": [[0, 0, 0], [1, 0, 1], [0, 1, 1]],
        "held_out": [[1, 1, 2]],
        "max_tree_ops": 1,
        "generator": {
            "program": {
                "op": "add",
                "left": {"op": "input", "name": "x"},
                "right": {"op": "input", "name": "y"},
            },
        },
    }
