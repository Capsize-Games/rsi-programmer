#!/usr/bin/env python3
"""Bit-slice word codec and whole-word oracle (lattice v30).

One synchronized lattice round emits one 4-bit word; head i
drives bit i (weight 2**i). Words decode through the fixed
ISA v0 table below into a typed DAG. The decoder alphabet
is the safety boundary: reserved words hard-reject, every
instruction passes ActionSpace typecheck via validate_graph,
and graphs are built by the decoder, never by the heads.
"""
from __future__ import annotations

from typing import Any

from rsi_programmer.probe import CONSTS, interpret, validate_graph

ISA = {0: "add", 1: "sub", 2: "mul", 3: "lt", 4: "eq",
       5: "select", 6: "input", 7: "const", 8: "stop"}
OPCODES = {name: code for code, name in ISA.items()}
STOP = OPCODES["stop"]


def pack(bits: list[int]) -> int:
    """Four head bits (head i = weight 2**i) to one word."""
    if len(bits) != 4 or any(b not in (0, 1) for b in bits):
        raise ValueError("need exactly four binary bits")
    return bits[0] + 2 * bits[1] + 4 * bits[2] + 8 * bits[3]


def unpack(word: int) -> list[int]:
    """One word to four head bits, LSB first."""
    if type(word) is not int or not 0 <= word <= 15:
        raise ValueError("word out of range")
    return [word & 1, (word >> 1) & 1, (word >> 2) & 1,
            (word >> 3) & 1]


def _resolve(refs: list[Any], raw: int) -> Any:
    """Ref-word into the live table; wraps deterministically."""
    return refs[raw % len(refs)]


def decode(words: list[int]) -> dict[str, Any]:
    """Decode a word sequence to a typed DAG; reject unknown.

    Opcode word then operand ref-words; ref-words index the
    live table (x, y, built nodes). The sequence must end in
    stop and the program needs one Int node for output.
    """
    nodes: list[dict[str, Any]] = []
    types: list[str] = []
    refs: list[Any] = ["x", "y"]
    output: Any = None
    pos = 0
    while pos < len(words):
        token = words[pos]
        pos += 1
        if type(token) is not int or not 0 <= token <= 15:
            raise ValueError("word out of range")
        if token not in ISA:
            raise ValueError("unknown-token")
        name = ISA[token]
        if name == "stop":
            break
        need = 3 if name == "select" else 1 if name in (
            "input", "const") else 2
        if pos + need > len(words):
            raise ValueError("truncated-word")
        args, pos = words[pos:pos + need], pos + need
        if name == "input":
            # Terminals live in refs already; the word only
            # names x (even) or y (odd) for the author trace.
            continue
        if name == "const":
            value = CONSTS[args[0] % len(CONSTS)]
            nodes.append({"op": "const", "type": "Int",
                          "args": [], "value": value})
            types.append("Int")
        else:
            operands = [_resolve(refs, raw) for raw in args]
            ntype = "Bool" if name in ("lt", "eq") else "Int"
            nodes.append({"op": name, "type": ntype,
                          "args": operands})
            types.append(ntype)
        refs.append(len(nodes) - 1)
        if types[-1] == "Int":
            output = len(nodes) - 1
    else:
        raise ValueError("missing-stop")
    if output is None:
        raise ValueError("no-int-output")
    graph = {"input_types": {"x": "Int", "y": "Int"},
             "nodes": nodes, "output": output}
    validate_graph(graph)
    return graph


def oracle_holds_words(task: dict[str, Any],
                       words: list[int]) -> bool:
    """Whole-word oracle: decode then check train + held-out."""
    try:
        graph = decode(words)
    except ValueError:
        return False
    rows = list(task["train"]) + list(task["held_out"])
    for case in rows:
        if interpret(graph, case[0], case[1]) != case[2]:
            return False
    return True
