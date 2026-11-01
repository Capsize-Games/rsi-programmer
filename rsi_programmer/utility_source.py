"""Compile a learned composition into standalone data-processing code."""
from __future__ import annotations

from rsi_programmer.utility_ops import OPS


def emit_python(program: tuple[str, ...]) -> str:
    """Emit the selected processing steps plus fixed command-line plumbing."""
    if not program or any(op not in OPS for op in program):
        raise ValueError("supported learned utility program required")
    lines = ["import argparse", "import json", "from pathlib import Path",
             "", "", "def process(items: list[int]) -> list[int]:",
             "    values = list(items)"]
    filters = {"positive": "value > 0", "nonnegative": "value >= 0",
               "even": "value % 2 == 0", "odd": "value % 2 != 0"}
    maps = {"absolute": "abs(value)", "negate": "-value",
            "double": "2 * value", "increment": "value + 1"}
    for op in program:
        if op in filters:
            lines += ["    next_values = []", "    for value in values:",
                      f"        if {filters[op]}:",
                      "            next_values.append(value)",
                      "    values = next_values"]
        elif op in maps:
            lines += ["    next_values = []", "    for value in values:",
                      f"        next_values.append({maps[op]})",
                      "    values = next_values"]
        elif op == "unique":
            lines += ["    next_values = []", "    seen = set()",
                      "    for value in values:",
                      "        if value not in seen:",
                      "            seen.add(value)",
                      "            next_values.append(value)",
                      "    values = next_values"]
        elif op == "prefix_sum":
            lines += ["    next_values = []", "    total = 0",
                      "    for value in values:", "        total += value",
                      "        next_values.append(total)",
                      "    values = next_values"]
        elif op == "reverse":
            lines.append("    values = list(reversed(values))")
        else:
            descending = ", reverse=True" if op == "descending" else ""
            lines.append(f"    values = sorted(values{descending})")
    lines += ["    return values", "", "", "def main() -> None:",
              "    parser = argparse.ArgumentParser(",
              "        description='Process a JSON list of integer readings.'"
              ")",
              "    parser.add_argument('input', type=Path)",
              "    parser.add_argument('output', type=Path)",
              "    args = parser.parse_args()",
              "    if args.input.resolve() == args.output.resolve():",
              "        parser.error('Input and output files must differ.')",
              "    try:",
              "        values = json.loads(args.input.read_text())",
              "        if not isinstance(values, list) or any(",
              "                type(value) is not int for value in values):",
              "            raise ValueError("
              "'Expected a JSON list of integers.')",
              "    except (OSError, ValueError) as error:",
              "        parser.error(str(error))",
              "    args.output.write_text(json.dumps(process(values)) "
              "+ '\\n')",
              "", "", "if __name__ == '__main__':", "    main()", ""]
    return "\n".join(lines)


def emit_curlee(program: tuple[str, ...]) -> str:
    """Emit processing for at most 256 signed integer readings."""
    if not program or any(op not in OPS for op in program):
        raise ValueError("supported learned utility program required")
    lines = ["fn process(input: &[Int; 256], count: Int,",
             "           output: &mut [Int; 256]) -> Int",
             "    [ requires count >= 0 && count <= 256; ] {",
             "    if (count < 0 || count > 256) { return -1; }"]
    previous = "input"
    previous_length = "count"
    filters = {"positive": "value > 0", "nonnegative": "value >= 0",
               "even": "value % 2 == 0", "odd": "value % 2 != 0"}
    maps = {"negate": "0 - value", "double": "2 * value",
            "increment": "value + 1"}
    for index, op in enumerate(program):
        current, length = f"s{index}", f"len{index}"
        lines += [f"    let {current}: [Int; 256] = [0; 256];",
                  f"    let {length}: Int = 0;"]
        if op == "prefix_sum":
            lines.append(f"    let total{index}: Int = 0;")
        lines += [f"    for (i{index} in 0..{previous_length})",
                  f"        [ invariant {previous_length} >= 0 && "
                  f"{previous_length} <= 256;",
                  f"          invariant {length} >= 0 && "
                  f"{length} <= i{index}; ] {{",
                  f"        if (i{index} < 0 || i{index} >= 256 || "
                  f"{length} < 0 || {length} >= 256) {{ return -1; }}",
                  f"        let value: Int = {previous}[i{index}];"]
        append = [f"{current}[{length}] = value;",
                  f"{length} = {length} + 1;"]
        if op in filters:
            lines += [f"        if ({filters[op]}) {{",
                      *("            " + line for line in append),
                      "        }"]
        elif op in maps:
            lines += [f"        {current}[{length}] = {maps[op]};",
                      f"        {length} = {length} + 1;"]
        elif op == "absolute":
            lines += ["        if (value < 0) {",
                      f"            {current}[{length}] = 0 - value;",
                      "        } else {",
                      f"            {current}[{length}] = value;",
                      "        }", f"        {length} = {length} + 1;"]
        elif op == "unique":
            lines += ["        let duplicate: Bool = false;",
                      f"        for (j{index} in 0..{length})",
                      f"            [ invariant {length} >= 0 && "
                      f"{length} <= i{index};",
                      f"              invariant i{index} >= 0 && "
                      f"i{index} < {previous_length};",
                      f"              invariant {previous_length} "
                      "<= 256; ] {",
                      f"            if (j{index} < 0 || j{index} >= 256) "
                      "{ return -1; }",
                      f"            if ({current}[j{index}] == value) {{",
                      "                duplicate = true;", "            }",
                      "        }", "        if (!duplicate) {",
                      *("            " + line for line in append),
                      "        }"]
        elif op == "prefix_sum":
            lines += [f"        total{index} = total{index} + value;",
                      f"        {current}[{length}] = total{index};",
                      f"        {length} = {length} + 1;"]
        elif op == "reverse":
            lines += [f"        let at: Int = "
                      f"{previous_length} - i{index} - 1;",
                      "        if (at < 0 || at >= 256) { return -1; }",
                      f"        {current}[{length}] = {previous}[at];",
                      f"        {length} = {length} + 1;"]
        else:
            lines += ["        " + line for line in append]
        lines.append("    }")
        if op in ("ascending", "descending"):
            comparison = ">" if op == "ascending" else "<"
            lines += [f"    if ({length} > 1) {{",
                      f"    for (a{index} in 0..{length})",
                      f"        [ invariant {length} > 1 && "
                      f"{length} <= 256; ] {{",
                      f"        for (b{index} in 0..({length} - 1))",
                      f"            [ invariant {length} > 1 && "
                      f"{length} <= 256; ] {{",
                      f"            if (b{index} < 0 || b{index} >= 255) "
                      "{ return -1; }",
                      f"            let first: Int = {current}[b{index}];",
                      f"            let second: Int = "
                      f"{current}[b{index} + 1];",
                      f"            if (first {comparison} second) {{",
                      f"                {current}[b{index}] = second;",
                      f"                {current}[b{index} + 1] = first;",
                      "            }", "        }", "    }", "    }"]
        previous = current
        previous_length = length
    lines += [f"    for (i in 0..{previous_length})",
              f"        [ invariant {previous_length} >= 0 && "
              f"{previous_length} <= 256; ] {{",
              "        if (i < 0 || i >= 256) { return -1; }",
              f"        output[i] = {previous}[i];", "    }",
              f"    return {previous_length};", "}", ""]
    return "\n".join(lines)
