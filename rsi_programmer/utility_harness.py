"""Independent expected-value assertions around emitted CURLEE routines."""
from __future__ import annotations

from rsi_programmer.utility_source import emit_curlee


def curlee_harness(program: tuple[str, ...], inputs: list[list[int]],
                   expected: list[list[int]]) -> str:
    """Assert supplied oracle values without computing them in target code."""
    if len(inputs) != len(expected) or not inputs:
        raise ValueError("nonempty input and expected arrays must match")
    if any(len(values) > 256 for values in inputs + expected):
        raise ValueError("CURLEE processing capacity is 256 integers")
    lines = [emit_curlee(program),
             "fn correct(input: &[Int; 256], count: Int,",
             "           expected: &[Int; 256], expected_count: Int) -> Bool",
             "    [ requires count >= 0 && count <= 256;",
             "      requires expected_count >= 0 && expected_count <= 256; ]",
             "{",
             "    let output: [Int; 256] = [0; 256];",
             "    let size: Int = process(&input, count, &mut output);",
             "    if (size != expected_count) { return false; }",
             "    for (i in 0..expected_count)",
             "        [ invariant expected_count >= 0 && "
             "expected_count <= 256; ] {",
             "        if (i < 0 || i >= 256) { return false; }",
             "        if (output[i] != expected[i]) { return false; }",
             "    }", "    return true;", "}", "",
             "fn main() -> Int {",
             "    let input: [Int; 256] = [0; 256];",
             "    let expected: [Int; 256] = [0; 256];"]
    for number, (values, answers) in enumerate(zip(inputs, expected), 1):
        for name, array in (("input", values), ("expected", answers)):
            lines += [f"    {name}[{index}] = {value};"
                      for index, value in enumerate(array)]
        lines += [f"    if (!correct(&input, {len(values)}, &expected, "
                  f"{len(answers)})) {{ return {number}; }}"]
    lines += ["    return 0;", "}", ""]
    return "\n".join(lines)
