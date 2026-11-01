"""Lower learner-constructed statement trees directly to target source."""
from __future__ import annotations

from rsi_programmer.statement_syntax import Block, Expr, parse
from rsi_programmer.utility_source import emit_python

SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "rem": "%",
           "gt": ">", "ge": ">=", "lt": "<", "le": "<=",
           "eq": "==", "ne": "!="}


def _expr(expr: Expr, language: str) -> str:
    names = {"zero": "0", "one": "1", "two": "2", "minus_one": "-1",
             "length": "len(items)" if language == "python" else "count"}
    if len(expr) == 1:
        return names.get(expr[0], expr[0])
    left, right = _expr(expr[1], language), _expr(expr[2], language)
    if expr[0] == "rem" and language == "python":
        return f"_remainder({left}, {right})"
    return f"({left} {SYMBOLS[expr[0]]} {right})"


def _block(block: Block, language: str, level: int) -> list[str]:
    result: list[str] = []
    indent = "    " * level
    ending = ";" if language == "curlee" else ""
    for statement in block:
        kind = statement[0]
        if kind in ("set", "return"):
            action = "acc = " if kind == "set" else "return "
            result.append(indent + action + _expr(statement[1], language)
                          + ending)
        elif kind == "if":
            condition = _expr(statement[1], language)
            if language == "python":
                result.append(indent + f"if {condition}:")
            else:
                result.append(indent + f"if ({condition}) {{")
            result += _block(statement[2], language, level + 1)
            if language == "curlee":
                result.append(indent + "}")
        else:
            if language == "python":
                result.append(indent + "for value in items:")
            else:
                result += [indent + "for (i in 0..count)",
                           indent + "    [ invariant count >= 0 && "
                           "count <= 256; ] {",
                           indent + "    if (i < 0 || i >= 256) "
                           "{ return 0; }",
                           indent + "    let value: Int = input[i];"]
            result += _block(statement[1], language, level + 1)
            if language == "curlee":
                result.append(indent + "}")
    return result


def python_source(tokens: tuple[str, ...]) -> str:
    """Emit a standalone CLI whose statement structure comes from tokens."""
    block = parse(tokens)
    # Reuse only the existing imports and CLI; replace its complete function.
    plumbing = emit_python(("positive",))
    imports = plumbing[:plumbing.index("def process")]
    main = plumbing[plumbing.index("def main") :]
    lines = ["def _remainder(a: int, b: int) -> int:",
             "    value = abs(a) % abs(b)",
             "    return -value if a < 0 else value", "", "",
             "def process(items: list[int]) -> int:", "    acc = 0",
             *_block(block, "python", 1), "", "", main]
    return imports + "\n".join(lines)


def curlee_source(tokens: tuple[str, ...]) -> str:
    """Emit the same AST using a bounded-array scalar-returning interface."""
    lines = ["fn process(input: &[Int; 256], count: Int) -> Int",
             "    [ requires count >= 0 && count <= 256; ] {",
             "    let acc: Int = 0;", *_block(parse(tokens), "curlee", 1),
             "}", ""]
    return "\n".join(lines)


def curlee_harness(tokens: tuple[str, ...], items: list[int], answer: int
                   ) -> str:
    """Assert a separately computed scalar expected result."""
    if len(items) > 256:
        raise ValueError("CURLEE input capacity exceeded")
    lines = [curlee_source(tokens), "fn main() -> Int {",
             "    let input: [Int; 256] = [0; 256];"]
    lines += [f"    input[{i}] = {value};" for i, value in enumerate(items)]
    lines += [f"    if (process(&input, {len(items)}) != {answer}) {{",
              "        return 1;", "    }", "    return 0;", "}", ""]
    return "\n".join(lines)
