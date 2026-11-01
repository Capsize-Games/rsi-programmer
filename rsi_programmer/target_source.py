"""Lower a learned arithmetic graph to executable Python or CURLEE source."""
from __future__ import annotations

import keyword
import re
from typing import Any

from rsi_programmer.probe import validate_graph

SYMBOLS = {"add": "+", "sub": "-", "mul": "*"}


def _reference(value: str | int) -> str:
    if value in ("x", "y"):
        return str(value)
    if type(value) is int and value >= 0:
        return f"n{value}"
    raise ValueError("unsupported source reference")


def emit_function(graph: dict[str, Any], language: str,
                  name: str = "solve") -> str:
    """Compile decisions to syntax; this emitter chooses no algorithm.

    The current backend covers only the learner's pure arithmetic grammar.
    It accepts validated data, never arbitrary generated source text.
    """
    validate_graph(graph)
    if language not in ("python", "curlee"):
        raise ValueError("target must be python or curlee")
    if (not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", name)
            or keyword.iskeyword(name) or name in ("fn", "let", "static")):
        raise ValueError("invalid function name")
    python = language == "python"
    lines = ([f"def {name}(x: int, y: int) -> int:"] if python else
             [f"fn {name}(x: Int, y: Int) -> Int {{"])
    for index, node in enumerate(graph["nodes"]):
        if node["op"] not in SYMBOLS:
            raise ValueError("target backend supports arithmetic nodes only")
        left, right = (_reference(ref) for ref in node["args"])
        expression = f"{left} {SYMBOLS[node['op']]} {right}"
        if python:
            lines.append(f"    n{index} = {expression}")
        else:
            lines.append(f"    let n{index}: Int = {expression};")
    result = _reference(graph["output"])
    lines.append(f"    return {result}" + ("" if python else ";"))
    if not python:
        lines.append("}")
    return "\n".join(lines) + "\n"
