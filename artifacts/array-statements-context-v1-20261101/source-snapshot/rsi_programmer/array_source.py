"""Lower array statements to Python files and bounded CURLEE routines."""
from __future__ import annotations

from rsi_programmer.array_syntax import Block, parse
from rsi_programmer.statement_source import _expr
from rsi_programmer.utility_source import emit_python


def _body(body: Block, language: str, level: int) -> list[str]:
    result: list[str] = []
    indent = "    " * level
    for statement in body:
        if statement[0] == "append":
            expr = _expr(statement[1], language)
            if language == "python":
                result += [indent + f"output.append({expr})",
                           indent + "outlen += 1"]
            else:
                result += [indent + "if (outlen < 0 || outlen >= 512) "
                           "{ return -1; }",
                           indent + f"output[outlen] = {expr};",
                           indent + "outlen = outlen + 1;"]
        else:
            expr = _expr(statement[1], language)
            opening = f"if {expr}:" if language == "python" else (
                f"if ({expr}) {{")
            result.append(indent + opening)
            result += _body(statement[2], language, level + 1)
            if language == "curlee":
                result.append(indent + "}")
    return result


def python_source(tokens: tuple[str, ...]) -> str:
    plumbing = emit_python(("positive",))
    imports = plumbing[:plumbing.index("def process")]
    main = plumbing[plumbing.index("def main"):]
    lines = ["def _remainder(a: int, b: int) -> int:",
             "    value = abs(a) % abs(b)",
             "    return -value if a < 0 else value", "", "",
             "def process(items: list[int]) -> list[int]:",
             "    if len(items) > 256:",
             "        raise ValueError('At most 256 readings required.')",
             "    output = []", "    outlen = 0",
             "    for value in items:",
             *_body(parse(tokens), "python", 2),
             "    return output", "", "", main]
    return imports + "\n".join(lines)


def curlee_source(tokens: tuple[str, ...]) -> str:
    lines = ["fn process(input: &[Int; 256], count: Int,",
             "           output: &mut [Int; 512]) -> Int",
             "    [ requires count >= 0 && count <= 256; ] {",
             "    let outlen: Int = 0;", "    for (i in 0..count)",
             "        [ invariant count >= 0 && count <= 256;",
             "          invariant outlen >= 0 && outlen <= 2 * i; ] {",
             "        if (i < 0 || i >= 256) { return -1; }",
             "        let value: Int = input[i];",
             *_body(parse(tokens), "curlee", 2), "    }",
             "    return outlen;", "}", ""]
    return "\n".join(lines)


def curlee_harness(tokens: tuple[str, ...], items: list[int],
                   expected: list[int]) -> str:
    if len(items) > 256 or len(expected) > 512:
        raise ValueError("array harness capacity exceeded")
    lines = [curlee_source(tokens), "fn main() -> Int {",
             "    let input: [Int; 256] = [0; 256];",
             "    let output: [Int; 512] = [0; 512];"]
    lines += [f"    input[{index}] = {value};"
              for index, value in enumerate(items)]
    lines += [f"    if (process(&input, {len(items)}, &mut output) != "
              f"{len(expected)}) {{ return 1; }}"]
    lines += [f"    if (output[{index}] != {value}) {{ return 1; }}"
              for index, value in enumerate(expected)]
    lines += ["    return 0;", "}", ""]
    return "\n".join(lines)
