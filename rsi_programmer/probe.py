#!/usr/bin/env python3
"""Bounded non-LLM search for a pure, typed computation DAG.

Generated programs are structured data. The lowerer emits source as a report
artifact, but validation uses a small allowlisted interpreter and an
independent held-out oracle; emitted source is never executed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


OPS: dict[str, Callable[[int, int], int]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
}
CONSTS: tuple[int, ...] = (-2, -1, 0, 1, 2, 3, 4, 5)
CMP: dict[str, Callable[[int, int], bool]] = {
    "lt": lambda a, b: a < b,
    "eq": lambda a, b: a == b,
}
SELECT = "select"


def type_of(expr: Expr | str) -> str:
    """Static IR type: Bool for comparisons, Int otherwise."""
    if isinstance(expr, str):
        return "Int"
    if expr.op in CMP:
        return "Bool"
    return "Int"
TASK = "f(x, y) = (x + y) * (x + y) + y"
TRAIN_CASES = ((-3, 1, 5), (-2, 2, 2), (-1, -2, 7), (0, 3, 12),
               (1, 0, 1), (2, -1, 0), (3, 2, 27), (4, -3, -2))
TRAIN = tuple((x, y) for x, y, _expected in TRAIN_CASES)
TRAIN_EXPECTED = tuple(expected for _x, _y, expected in TRAIN_CASES)
HELD_OUT = ((-7, 2), (-4, -1), (-1, 4), (0, -5), (2, 5), (5, -2), (8, 1))
MAX_TREE_OPS = 4


@dataclass(frozen=True)
class Expr:
    op: str
    left: Expr | str | int
    right: Expr | None = None
    size: int = 0
    third: Expr | None = None


def terminals() -> tuple[Expr, ...]:
    return Expr("input", "x"), Expr("input", "y")


def const_terms(values: tuple[int, ...] = ()) -> tuple[Expr, ...]:
    """Size-zero constant terminals for the given values."""
    return tuple(Expr("const", value, None, 0) for value in values)


def signature(expr: Expr, rows: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    if expr.op == "const":
        value = expr.left
        assert type(value) is int
        return tuple(value for _ in rows)
    if expr.op == "input":
        col = 0 if expr.left == "x" else 1
        return tuple(row[col] for row in rows)
    if expr.op in CMP:
        left = signature(expr.left, rows)  # type: ignore[arg-type]
        right = signature(expr.right, rows)  # type: ignore[arg-type]
        fn = CMP[expr.op]
        return tuple(fn(a, b) for a, b in zip(left, right))
    if expr.op == SELECT:
        cond = signature(expr.left, rows)  # type: ignore[arg-type]
        first = signature(expr.right, rows)  # type: ignore[arg-type]
        assert expr.third is not None
        second = signature(expr.third, rows)
        return tuple(a if c else b for c, a, b in
                     zip(cond, first, second))
    left = signature(expr.left, rows)  # type: ignore[arg-type]
    right = signature(expr.right, rows)  # type: ignore[arg-type]
    fn = OPS[expr.op]
    return tuple(fn(a, b) for a, b in zip(left, right))


def enumerate_programs(rows: tuple[tuple[int, int], ...], expected: tuple[int, ...], max_ops: int,
                       root_op: str | None = None, deduplicate: bool = True,
                       consts: tuple[int, ...] = (),
                       branch: bool = False):
    """Enumerate typed integer expression trees by node count.

    Each binary Apply consumes two Int values and returns Int. The output is
    later hash-consed into a DAG, so repeated subexpressions share one node.
    Constant terminals join level zero only when consts is non-empty, and
    Bool comparisons plus ternary select join only when branch is set, so
    frozen families keep their exact historical search order by default.
    """
    levels: dict[int, list[Expr]] = {
        0: list(terminals()) + list(const_terms(consts))}
    blevels: dict[int, list[Expr]] = {0: []}
    tried = 0
    for size in range(1, max_ops + 1):
        current: list[Expr] = []
        current_seen: set[tuple[str, tuple[int, ...]]] = set()
        for left_size in range(size):
            right_size = size - left_size - 1
            for left, right in itertools.product(levels[left_size], levels[right_size]):
                for op in OPS:
                    tried += 1
                    expr = Expr(op, left, right, size)
                    if deduplicate:
                        sig = signature(expr, rows)
                        key = (op, sig)
                        if key in current_seen:
                            continue
                        current_seen.add(key)
                    # Keep per-layer candidates in deterministic grammar order.
                    current.append(expr)
        bcurrent: list[Expr] = []
        if branch:
            bseen: set[tuple[str, tuple[int, ...]]] = set()
            for left_size in range(size):
                right_size = size - left_size - 1
                pairs = itertools.product(levels[left_size],
                                          levels[right_size])
                for left, right in pairs:
                    for op in CMP:
                        tried += 1
                        expr = Expr(op, left, right, size)
                        if deduplicate:
                            key = (op, signature(expr, rows))
                            if key in bseen:
                                continue
                            bseen.add(key)
                        bcurrent.append(expr)
            for cond_size in range(size):
                for first_size in range(size - cond_size):
                    second_size = size - cond_size - first_size - 1
                    triples = itertools.product(
                        blevels[cond_size], levels[first_size],
                        levels[second_size])
                    for cond, first, second in triples:
                        tried += 1
                        expr = Expr(SELECT, cond, first, size,
                                    third=second)
                        if deduplicate:
                            key = (SELECT, signature(expr, rows))
                            if key in current_seen:
                                continue
                            current_seen.add(key)
                        current.append(expr)
        levels[size] = current
        blevels[size] = bcurrent
        for expr in current:
            if (root_op is None or expr.op == root_op
                    or (branch and root_op == SELECT
                        and expr.op == SELECT)):
                if signature(expr, rows) == expected:
                    return expr, tried, _volume(levels, blevels)
    return None, tried, _volume(levels, blevels)


def _volume(levels: dict[int, list[Expr]],
            blevels: dict[int, list[Expr]]) -> int:
    """Retained expressions across both typed level tables."""
    return sum(map(len, levels.values())) + sum(
        map(len, blevels.values()))


def lower_to_dag(expr: Expr) -> dict:
    nodes: list[dict] = []
    intern: dict[tuple, int] = {}

    def visit(term: Expr | str) -> int | str:
        if isinstance(term, str):
            if term not in {"x", "y"}:
                raise ValueError("unknown input")
            return term
        if term.op == "const":
            if type(term.left) is not int or term.left not in CONSTS:
                raise ValueError("constant out of range")
            key = ("const", term.left)
            if key not in intern:
                intern[key] = len(nodes)
                nodes.append({"op": "const", "type": "Int",
                              "args": [], "value": term.left})
            return intern[key]
        if term.op == "input":
            return visit(term.left)
        if term.op in CMP:
            left = visit(term.left)  # type: ignore[arg-type]
            right = visit(term.right)  # type: ignore[arg-type]
            key = (term.op, left, right)
            if key not in intern:
                intern[key] = len(nodes)
                nodes.append({"op": term.op, "type": "Bool",
                              "args": [left, right]})
            return intern[key]
        if term.op == SELECT:
            assert term.third is not None
            cond = visit(term.left)  # type: ignore[arg-type]
            first = visit(term.right)  # type: ignore[arg-type]
            second = visit(term.third)
            key = (term.op, cond, first, second)
            if key not in intern:
                intern[key] = len(nodes)
                nodes.append({"op": term.op, "type": "Int",
                              "args": [cond, first, second]})
            return intern[key]
        left, right = visit(term.left), visit(term.right)  # type: ignore[arg-type]
        key = (term.op, left, right)
        if key not in intern:
            intern[key] = len(nodes)
            nodes.append({"op": term.op, "type": "Int", "args": [left, right]})
        return intern[key]

    output = visit(expr)
    return {"input_types": {"x": "Int", "y": "Int"}, "nodes": nodes, "output": output}


def validate_graph(graph: dict) -> None:
    if set(graph) != {"input_types", "nodes", "output"}:
        raise ValueError("malformed graph fields")
    if graph["input_types"] != {"x": "Int", "y": "Int"} or not isinstance(graph["nodes"], list):
        raise ValueError("malformed inputs or node table")
    types: list[str] = []
    for index, node in enumerate(graph["nodes"]):
        if not isinstance(node, dict):
            raise ValueError("malformed node")
        if node.get("op") == "const":
            if set(node) != {"op", "type", "args", "value"}:
                raise ValueError("malformed constant")
            if node["type"] != "Int" or node["args"] != []:
                raise ValueError("constant signature mismatch")
            if (type(node["value"]) is not int
                    or node["value"] not in CONSTS):
                raise ValueError("constant out of range")
            types.append("Int")
            continue
        if set(node) != {"op", "type", "args"}:
            raise ValueError("malformed node")
        if node["op"] not in OPS and node["op"] not in CMP \
                and node["op"] != SELECT:
            raise ValueError("unknown operation")
        args = node["args"]
        if node["op"] in CMP:
            want, kinds, ntype = 2, ("Int", "Int"), "Bool"
        elif node["op"] == SELECT:
            want, kinds, ntype = 3, ("Bool", "Int", "Int"), "Int"
        else:
            want, kinds, ntype = 2, ("Int", "Int"), "Int"
        if node["type"] != ntype or not isinstance(args, list) \
                or len(args) != want:
            raise ValueError("operation signature mismatch")
        for arg, kind in zip(args, kinds):
            if isinstance(arg, str):
                if arg in {"x", "y"} and kind == "Int":
                    continue
                raise ValueError("unknown input reference")
            if type(arg) is not int or arg < 0 or arg >= index:
                raise ValueError("invalid, cyclic, or forward reference")
            if types[arg] != kind:
                raise ValueError("operand type mismatch")
        types.append(node["type"])
    out = graph["output"]
    if isinstance(out, str):
        if out not in {"x", "y"}:
            raise ValueError("unknown output input reference")
    elif type(out) is not int or not 0 <= out < len(types):
        raise ValueError("invalid output reference")


def interpret(graph: dict, x: int, y: int) -> int:
    validate_graph(graph)
    out = graph["output"]
    if not isinstance(out, str) and graph["nodes"][out]["type"] != "Int":
        raise ValueError("program output must be Int")
    values: list[int] = []
    inputs = {"x": x, "y": y}
    for node in graph["nodes"]:
        if node["op"] == "const":
            values.append(node["value"])
            continue
        args = [inputs[arg] if isinstance(arg, str) else values[arg] for arg in node["args"]]
        if node["op"] in CMP:
            values.append(CMP[node["op"]](*args))
        elif node["op"] == SELECT:
            values.append(args[1] if args[0] else args[2])
        else:
            values.append(OPS[node["op"]](*args))
    output = graph["output"]
    return inputs[output] if isinstance(output, str) else values[output]


def oracle_accepts(graph: dict, rows: tuple[tuple[int, int], ...]) -> bool:
    """Check outputs against the task contract, independently of graph search."""
    return all(interpret(graph, x, y) == (x + y) * (x + y) + y for x, y in rows)


def render_source(graph: dict) -> str:
    validate_graph(graph)
    lines = ["def synthesized(x: int, y: int) -> int:"]
    for i, node in enumerate(graph["nodes"]):
        if node["op"] == "const":
            lines.append(f"    t{i} = ({node['value']})")
            continue
        args = [arg if isinstance(arg, str) else f"t{arg}" for arg in node["args"]]
        if node["op"] in CMP:
            symbol = {"lt": "<", "eq": "=="}[node["op"]]
            lines.append(f"    t{i} = ({args[0]} {symbol} {args[1]})")
            continue
        if node["op"] == SELECT:
            lines.append(f"    t{i} = ({args[1]} if {args[0]}"
                         f" else {args[2]})")
            continue
        symbol = {"add": "+", "sub": "-", "mul": "*"}[node["op"]]
        lines.append(f"    t{i} = ({args[0]} {symbol} {args[1]})")
    result = graph["output"]
    lines.append(f"    return {result if isinstance(result, str) else f't{result}'}")
    return "\n".join(lines)


def _partition_worker(args: tuple[tuple[tuple[int, int], ...], tuple[int, ...], int, str]):
    rows, expected, max_ops, root_op = args
    expr, tried, unique = enumerate_programs(rows, expected, max_ops, root_op=root_op, deduplicate=True)
    return expr, tried, unique


def _usage() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    status = Path("/proc/self/status")
    memory = {}
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith(("VmHWM:", "VmRSS:")):
                key, value, _unit = line.split()
                memory[key.rstrip(":").lower() + "_kib"] = int(value)
    return {"user_cpu_s": usage.ru_utime, "system_cpu_s": usage.ru_stime,
            "max_rss_kib": usage.ru_maxrss, **memory}


def _negative_controls(graph: dict) -> dict[str, bool]:
    results = {}
    cases = {
        "unknown_op": lambda g: g["nodes"].__setitem__(0, {"op": "shell", "type": "Int", "args": ["x", "y"]}),
        "cycle": lambda g: g["nodes"].__setitem__(0, {"op": "add", "type": "Int", "args": [0, "y"]}),
        "malformed": lambda g: g["nodes"].__setitem__(0, {"op": "add", "type": "Int", "args": ["x"]}),
        "unhashable_reference": lambda g: g["nodes"].__setitem__(0, {"op": "add", "type": "Int", "args": [["x"], "y"]}),
        "const_with_args": lambda g: g["nodes"].__setitem__(0, {"op": "const", "type": "Int", "args": ["x"], "value": 1}),
        "const_out_of_range": lambda g: g["nodes"].__setitem__(0, {"op": "const", "type": "Int", "args": [], "value": 99}),
    }
    for name, corrupt in cases.items():
        candidate = json.loads(json.dumps(graph))
        corrupt(candidate)
        try:
            validate_graph(candidate)
        except ValueError:
            results[name] = True
        else:
            results[name] = False
    wrong_graph = {"input_types": {"x": "Int", "y": "Int"}, "nodes": [], "output": "x"}
    results["wrong_semantics"] = not oracle_accepts(wrong_graph, HELD_OUT)
    return results


def run() -> dict:
    before = _usage()
    start = time.perf_counter()
    expr, considered, unique = enumerate_programs(TRAIN, TRAIN_EXPECTED, MAX_TREE_OPS, deduplicate=True)
    wall = time.perf_counter() - start
    usage = _usage()
    result: dict = {
        "task": TASK,
        "train_rows": len(TRAIN),
        "held_out_rows": len(HELD_OUT),
        "max_tree_ops": MAX_TREE_OPS,
        "search": {"candidate_expressions_considered": considered,
                   "retained_expressions": unique,
                   "wall_s": wall,
                   "time_to_first_training_fit_s": wall if expr is not None else None,
                   "cpu_s": (usage["user_cpu_s"] - before["user_cpu_s"])
                            + (usage["system_cpu_s"] - before["system_cpu_s"]),
                   "max_rss_kib_process_high_water": usage.get("vmhwm_kib", usage["max_rss_kib"]),
                   "rss_kib_at_measurement": usage.get("vmrss_kib")},
        "enumerative_baseline": {},
        "parallel_partition": {},
    }
    if expr is None:
        result["status"] = "no training-fit candidate within bound"
        return result
    graph = lower_to_dag(expr)
    source = render_source(graph)
    train_ok = oracle_accepts(graph, TRAIN)
    held_values = [{"x": x, "y": y, "actual": interpret(graph, x, y),
                    "oracle": (x + y) * (x + y) + y} for x, y in HELD_OUT]
    held_ok = oracle_accepts(graph, HELD_OUT)

    baseline_before = _usage()
    start = time.perf_counter()
    baseline_expr, base_considered, base_retained = enumerate_programs(
        TRAIN, TRAIN_EXPECTED, MAX_TREE_OPS, deduplicate=False)
    base_wall = time.perf_counter() - start
    baseline_after = _usage()
    baseline_graph = lower_to_dag(baseline_expr) if baseline_expr else None
    baseline_held_out_exact = bool(baseline_graph and oracle_accepts(baseline_graph, HELD_OUT))
    result["enumerative_baseline"] = {
        "candidate_expressions_considered": base_considered,
        "retained_expressions": base_retained,
        "wall_s": base_wall,
        "cpu_s": (baseline_after["user_cpu_s"] - baseline_before["user_cpu_s"])
                 + (baseline_after["system_cpu_s"] - baseline_before["system_cpu_s"]),
        "status": "found" if baseline_expr else "not found",
        "held_out_exact": baseline_held_out_exact,
        "dag_node_count_after_hash_consing": len(baseline_graph["nodes"]) if baseline_graph else None,
        "selected_graph_semantics_match_search": bool(
            baseline_expr and signature(baseline_expr, TRAIN) == signature(expr, TRAIN)),
    }

    child_usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        parts = list(pool.map(_partition_worker,
                              [(TRAIN, TRAIN_EXPECTED, MAX_TREE_OPS, op) for op in OPS]))
    parallel_wall = time.perf_counter() - start
    child_usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu_delta = (
        child_usage_after.ru_utime - child_usage_before.ru_utime
        + child_usage_after.ru_stime - child_usage_before.ru_stime
    )
    partition_checks = []
    for root_op, part in zip(OPS, parts):
        candidate = part[0]
        correct = False
        if candidate is not None:
            candidate_graph = lower_to_dag(candidate)
            correct = candidate.op == root_op and oracle_accepts(candidate_graph, HELD_OUT)
        partition_checks.append({"root_op": root_op, "found": candidate is not None,
                                 "selected_root_matches_partition": candidate is None or candidate.op == root_op,
                                 "held_out_exact": correct if candidate is not None else None})
    result["parallel_partition"] = {
        "workers": 2,
        "root_operation_partitions": len(OPS),
        "wall_s": parallel_wall,
        "candidate_expressions_considered_including_repeated_subsearch": sum(p[1] for p in parts),
        "child_cpu_s": child_cpu_delta,
        "max_child_rss_kib": child_usage_after.ru_maxrss,
        "partition_checks": partition_checks,
        "status": "found" if any(p[0] for p in parts) else "not found",
        "interpretation": "cold process startup and repeated subsearch are included",
    }
    result.update({
        "status": "accepted" if train_ok and held_ok else "failed exact check",
        "selected_expression_tree_ops": expr.size,
        "dag_node_count_after_hash_consing": len(graph["nodes"]),
        "lowered_source_artifact": source,
        "generated_source_executed": False,
        "train_exact": train_ok,
        "held_out_exact": held_ok,
        "held_out_results": held_values,
        "negative_controls_rejected": _negative_controls(graph),
        "oracle": "independent direct formula (x+y)*(x+y)+y; not derived from candidate graph",
        "resource_note": "max RSS is process high-water and includes Python runtime; child CPU is the RUSAGE_CHILDREN delta across the measured pool; child high-water RSS is aggregate",
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run()
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else report)
    return 0 if report.get("status") == "accepted" and all(report["negative_controls_rejected"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
