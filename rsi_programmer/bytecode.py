#!/usr/bin/env python3
"""Stack bytecode with a tiny 16-bit VM (binary lowering v32).

Expression DAGs lower to byte streams; the VM executes them
with 16-bit two's-complement wraparound. The decoder
alphabet is the safety boundary (unknown opcodes, truncated
operands, and trailing bytes hard-reject), and the binary
oracle checks VM behavior against expected outputs, never
the expression layer. select/compare nodes are out of
scope: no solved task needs them.
"""
from __future__ import annotations

from typing import Any

from rsi_programmer.probe import validate_graph

HALT = 0x00
PUSH_X = 0x01
PUSH_Y = 0x02
PUSH_IMM16 = 0x03
FETCH = 0x04
STORE = 0x05
ADD = 0x06
SUB = 0x07
MUL = 0x08
NAMES = {HALT: "HALT", PUSH_X: "PUSH_X", PUSH_Y: "PUSH_Y",
         PUSH_IMM16: "PUSH_IMM16", FETCH: "FETCH",
         STORE: "STORE", ADD: "ADD", SUB: "SUB", MUL: "MUL"}
HAS_OPERAND = {PUSH_IMM16: 2, FETCH: 1, STORE: 1}
MASK = 0xFFFF
STACK_LIMIT = 1024


def wrap16(value: int) -> int:
    """Two's-complement signed 16-bit value."""
    value &= MASK
    return value - 0x10000 if value & 0x8000 else value


def _push(arg: Any, out: bytearray) -> None:
    """Emit input push or slot fetch for one reference."""
    if arg == "x":
        out.append(PUSH_X)
    elif arg == "y":
        out.append(PUSH_Y)
    else:
        out.append(FETCH)
        out.append(int(arg))


def assemble(graph: dict[str, Any]) -> bytes:
    """Lower a validated DAG to a byte stream program."""
    validate_graph(graph)
    if len(graph["nodes"]) > 255:
        raise ValueError("too many nodes for one slot byte")
    out = bytearray()
    for slot, node in enumerate(graph["nodes"]):
        op = node["op"]
        if op == "const":
            out.append(PUSH_IMM16)
            out += int(node["value"]).to_bytes(
                2, "little", signed=True)
        elif op in ("add", "sub", "mul"):
            _push(node["args"][0], out)
            _push(node["args"][1], out)
            out.append({"add": ADD, "sub": SUB,
                        "mul": MUL}[op])
        else:
            raise ValueError("unsupported op for bytecode")
        out.append(STORE)
        out.append(slot)
    _push(graph["output"], out)
    out.append(HALT)
    return bytes(out)


def disassemble(code: bytes) -> list[tuple[str, ...]]:
    """Strict decode to (name, operand) tuples; reject junk."""
    if not isinstance(code, (bytes, bytearray)):
        raise ValueError("code must be bytes")
    prog: list[tuple[str, ...]] = []
    pos = 0
    while pos < len(code):
        op = code[pos]
        pos += 1
        if op not in NAMES:
            raise ValueError("unknown opcode")
        if prog and prog[-1][0] == "HALT":
            raise ValueError("trailing bytes after HALT")
        if op in HAS_OPERAND:
            need = HAS_OPERAND[op]
            if pos + need > len(code):
                raise ValueError("truncated operand")
            raw = code[pos:pos + need]
            pos += need
            if op == PUSH_IMM16:
                prog.append((NAMES[op], str(int.from_bytes(
                    raw, "little", signed=True))))
            else:
                prog.append((NAMES[op], str(raw[0])))
        else:
            prog.append((NAMES[op],))
    if not prog or prog[-1][0] != "HALT":
        raise ValueError("missing HALT")
    return prog


def run_vm(code: bytes, x: int, y: int) -> int:
    """Execute a byte program; result as signed 16-bit."""
    prog = disassemble(code)
    stack: list[int] = []
    slots = [0] * 256
    inputs = {"PUSH_X": x & MASK, "PUSH_Y": y & MASK}
    for name, *rest in prog:
        if name in inputs:
            stack.append(inputs[name])
        elif name == "PUSH_IMM16":
            stack.append(int(rest[0]) & MASK)
        elif name == "FETCH":
            stack.append(slots[int(rest[0])])
        elif name == "STORE":
            if not stack:
                raise ValueError("stack underflow")
            slots[int(rest[0])] = stack.pop()
        elif name in ("ADD", "SUB", "MUL"):
            if len(stack) < 2:
                raise ValueError("stack underflow")
            right = stack.pop()
            left = stack.pop()
            if name == "ADD":
                stack.append((left + right) & MASK)
            elif name == "SUB":
                stack.append((left - right) & MASK)
            else:
                stack.append((left * right) & MASK)
        elif name == "HALT":
            break
        if len(stack) > STACK_LIMIT:
            raise ValueError("stack overflow")
    if not stack:
        raise ValueError("empty stack at HALT")
    return wrap16(stack[-1])


def binary_holds(task: dict[str, Any], code: bytes) -> bool:
    """Binary oracle: VM behavior on train plus held-out."""
    try:
        disassemble(code)
        rows = list(task["train"]) + list(task["held_out"])
        return all(run_vm(code, case[0], case[1]) == case[2]
                   for case in rows)
    except (KeyError, TypeError, ValueError, IndexError,
            OverflowError):
        return False
