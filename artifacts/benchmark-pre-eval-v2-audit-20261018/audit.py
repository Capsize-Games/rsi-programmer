#!/usr/bin/env python3
"""Independent structural and semantic audit for benchmark v2 inputs.

This module deliberately does not import the corpus generator, benchmark
runner, or their polynomial helpers.  Its polynomial arithmetic is a small,
separate exact implementation used to check the frozen JSON bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "fixtures/benchmark_pre_eval_v2.json"
CORPUS_PATH = ROOT / "fixtures/benchmark_pre_eval_corpus_v2.json"
V1_CORPUS_PATH = ROOT / "fixtures/benchmark_pre_eval_corpus_v1.json"
GENERATOR_PATH = ROOT / "scripts/generate_benchmark_pre_eval_v2.py"
CORPUS_MODULE_PATH = ROOT / "rsi_programmer/benchmark_v2_corpus.py"
PINNED_PROTOCOL_SHA256 = (
    "cf83dc077dc7960d4e576d898d842484655ac2c61d0e36128da6cda5ae8e7015"
)
PINNED_CORPUS_SHA256 = (
    "a8fd6add90b34da732da8dc96f2c0fafcabc2f017a17cdef4b31303eec4ba921"
)
PINNED_GENERATOR_SHA256 = {
    "scripts/generate_benchmark_pre_eval_v2.py": (
        "30cf448a5762710958df1ca96d7631b442e32638dd9236f63e297018d3076b31"
    ),
    "rsi_programmer/benchmark_v2_corpus.py": (
        "13c49ca629ce51e766ffdd3615a925a601ede4c9cff9a713710d159698f6526f"
    ),
}
FAMILIES = ("balanced", "repeated_subtree", "skewed")
SPLITS = ("train", "eval")
GRID = set(itertools.product(range(-32, 33), repeat=2))

# A polynomial maps (x exponent, y exponent) to a nonzero integer coefficient.
Polynomial = dict[tuple[int, int], int]


def digest(path: Path) -> str:
    """Hash file bytes exactly as frozen."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def poly_add(left: Polynomial, right: Polynomial,
             sign: int = 1) -> Polynomial:
    """Return left + sign*right, dropping zero coefficients."""
    result = dict(left)
    for power, coefficient in right.items():
        value = result.get(power, 0) + sign * coefficient
        if value:
            result[power] = value
        else:
            result.pop(power, None)
    return result


def poly_mul(left: Polynomial, right: Polynomial) -> Polynomial:
    """Return the exact product of two integer polynomials."""
    result: Polynomial = {}
    for (lx, ly), lc in left.items():
        for (rx, ry), rc in right.items():
            power = (lx + rx, ly + ry)
            result[power] = result.get(power, 0) + lc * rc
    return {power: coefficient for power, coefficient in result.items()
            if coefficient}


def expression_polynomial(node: Any) -> Polynomial:
    """Parse and evaluate a strict binary expression AST exactly."""
    if not isinstance(node, dict) or "op" not in node:
        raise ValueError("expression node must be an object with op")
    operator = node["op"]
    if operator == "input":
        if set(node) != {"op", "name"} or node["name"] not in {"x", "y"}:
            raise ValueError("malformed input node")
        return {(1, 0): 1} if node["name"] == "x" else {(0, 1): 1}
    if operator not in {"add", "sub", "mul"}:
        raise ValueError("unknown operator")
    if set(node) != {"op", "left", "right"}:
        raise ValueError("malformed binary node")
    left = expression_polynomial(node["left"])
    right = expression_polynomial(node["right"])
    if operator == "add":
        return poly_add(left, right)
    if operator == "sub":
        return poly_add(left, right, -1)
    return poly_mul(left, right)


def expression_key(node: Any) -> str:
    """Render the grammar in a deterministic fully parenthesized form."""
    if not isinstance(node, dict):
        raise ValueError("malformed expression node")
    if node.get("op") == "input":
        if set(node) != {"op", "name"} or node.get("name") not in {"x", "y"}:
            raise ValueError("malformed input node")
        return str(node["name"])
    if node.get("op") not in {"add", "sub", "mul"} or set(node) != {
            "op", "left", "right"}:
        raise ValueError("malformed binary node")
    return (f"{node['op']}({expression_key(node['left'])},"
            f"{expression_key(node['right'])})")


def operation_count(node: Any) -> int:
    """Count syntax-tree binary operators, without DAG sharing."""
    if not isinstance(node, dict):
        raise ValueError("malformed expression node")
    if node.get("op") == "input":
        if set(node) != {"op", "name"} or node.get("name") not in {"x", "y"}:
            raise ValueError("malformed input node")
        return 0
    if node.get("op") not in {"add", "sub", "mul"} or set(node) != {
            "op", "left", "right"}:
        raise ValueError("malformed binary node")
    return (1 + operation_count(node["left"])
            + operation_count(node["right"]))


def canonical_polynomial(poly: Polynomial) -> list[list[int]]:
    """Serialize coefficients in ascending (x power, y power) order."""
    return [[x_power, y_power, coefficient]
            for (x_power, y_power), coefficient in sorted(poly.items())]


def polynomial_degree(poly: Polynomial) -> int:
    """Return total degree, or -1 for the zero polynomial."""
    return max((x_power + y_power for x_power, y_power in poly), default=-1)


def coefficient_l1(poly: Polynomial) -> int:
    """Return the sum of absolute values of canonical coefficients."""
    return sum(abs(value) for value in poly.values())


def difficulty_floor(poly: Polynomial) -> bool:
    """Apply the predeclared algebraic floor, independent of any search."""
    degree = polynomial_degree(poly)
    norm = coefficient_l1(poly)
    return (degree >= 2 and (degree >= 5 or
                             (degree == 4 and norm > 1) or
                             (degree == 3 and norm > 2) or
                             (degree == 2 and norm > 4)))


def evaluate(poly: Polynomial, x_value: int, y_value: int) -> int:
    """Evaluate an exact polynomial at one integer pair."""
    return sum(coefficient * x_value ** x_power * y_value ** y_power
               for (x_power, y_power), coefficient in poly.items())


def family_matches(family: str, node: Any) -> bool:
    """Check the advertised syntax family independently."""
    if not isinstance(node, dict) or node.get("op") == "input":
        return False
    left = node.get("left")
    right = node.get("right")
    if family == "balanced":
        return operation_count(left) == operation_count(right)
    if family == "repeated_subtree":
        counts = Counter(subtree_keys(node))
        return any(count >= 2 for count in counts.values())
    if family == "skewed":
        return abs(operation_count(left) - operation_count(right)) >= 2
    return False


def subtree_keys(node: Any) -> list[str]:
    """Return every nonleaf subtree key, including the whole expression."""
    if node.get("op") == "input":
        return []
    return ([expression_key(node)] + subtree_keys(node["left"])
            + subtree_keys(node["right"]))


def seeded_digest(seed: int, value: str) -> str:
    """Recompute the documented seed-prefixed SHA256 ranking."""
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def syntax_shapes(size: int) -> list[dict[str, Any]]:
    """Enumerate ordered binary operator shapes of one size."""
    if size == 0:
        return [{"op": "input", "name": "_"}]
    result = []
    for left_size in range(size):
        right_size = size - left_size - 1
        for left in syntax_shapes(left_size):
            for right in syntax_shapes(right_size):
                for operator in ("add", "sub", "mul"):
                    result.append({"op": operator, "left": left,
                                   "right": right})
    return result


def labeled_expressions(shape: Any) -> list[dict[str, Any]]:
    """Enumerate all x/y assignments for the leaves of a syntax shape."""
    if shape["op"] == "input":
        return [{"op": "input", "name": name} for name in ("x", "y")]
    left_options = labeled_expressions(shape["left"])
    right_options = labeled_expressions(shape["right"])
    return [{"op": shape["op"], "left": left, "right": right}
            for left in left_options for right in right_options]


def independent_candidate_pools() -> tuple[
        dict[str, list[Any]], dict[str, int]]:
    """Rebuild family pools to verify sufficiency and selection provenance."""
    reservation_order = ("repeated_subtree", "balanced", "skewed")
    seeds = {"repeated_subtree": 82071, "balanced": 82072,
             "skewed": 82073}
    pools: dict[str, list[Any]] = {family: [] for family in FAMILIES}
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    rejected = Counter()
    for family in reservation_order:
        candidates = []
        for size in (4, 5):
            for shape in syntax_shapes(size):
                for node in labeled_expressions(shape):
                    if family_matches(family, node):
                        candidates.append(node)
        candidates.sort(key=lambda node: seeded_digest(
            seeds[family], expression_key(node)))
        for node in candidates:
            poly = expression_polynomial(node)
            key = tuple(tuple(term) for term in canonical_polynomial(poly))
            if key in seen:
                rejected["semantic_duplicates"] += 1
                continue
            if not difficulty_floor(poly):
                rejected["difficulty_floor"] += 1
                continue
            seen.add(key)
            pools[family].append(node)
    return pools, dict(rejected)


def check_3_op_floor_exhaustively() -> dict[str, Any]:
    """Enumerate the tiny <=3-op DSL to challenge the algebraic bound."""
    terminals = ({"op": "input", "name": "x"},
                 {"op": "input", "name": "y"})
    by_size: dict[int, list[dict[str, Any]]] = {0: list(terminals)}
    for size in range(1, 4):
        expressions: list[dict[str, Any]] = []
        for left_size in range(size):
            right_size = size - left_size - 1
            for left in by_size[left_size]:
                for right in by_size[right_size]:
                    for operator in ("add", "sub", "mul"):
                        expressions.append({"op": operator, "left": left,
                                            "right": right})
        by_size[size] = expressions
    counterexamples = []
    checked = 0
    for size in range(4):
        for node in by_size[size]:
            checked += 1
            poly = expression_polynomial(node)
            if difficulty_floor(poly):
                counterexamples.append({
                    "program": expression_key(node),
                    "polynomial": canonical_polynomial(poly),
                })
    return {"trees_checked": checked,
            "counterexamples": counterexamples,
            "bound_holds": not counterexamples}


def audit_corpus(corpus: Any) -> dict[str, Any]:
    """Audit counts, metadata, exact semantics, uniqueness, and row design."""
    errors: list[str] = []
    if not isinstance(corpus, dict):
        return {"ok": False, "errors": ["corpus root is not an object"]}
    splits = corpus.get("splits")
    if not isinstance(splits, dict):
        return {"ok": False, "errors": ["splits is not an object"]}
    expected_seeds = {"families": {"repeated_subtree": 82071,
                                   "balanced": 82072, "skewed": 82073},
                      "split": 82074, "inputs": 82075}
    if corpus.get("seeds") != expected_seeds:
        errors.append("seed metadata differs from frozen design")
    if corpus.get("corpus_id") != "benchmark-pre-eval-v2":
        errors.append("corpus id differs from v2")
    if corpus.get("generation_acceptance") != (
            "all selected functions retained; no solver or search-outcome "
            "filtering"):
        errors.append(
            "generation acceptance does not rule out outcome filters")
    if corpus.get("dsl") != {
            "operators": ["add", "sub", "mul"],
            "terminals": ["x", "y"], "max_tree_ops": 5}:
        errors.append("DSL metadata differs from frozen design")
    if corpus.get("generation_audit", {}).get("input_grid") != [-32, 32]:
        errors.append("input grid metadata differs from [-32, 32]")
    semantic: dict[tuple[tuple[int, int, int], ...], str] = {}
    ids: set[str] = set()
    expr_keys: set[str] = set()
    coordinates: set[tuple[int, int]] = set()
    counts: Counter[tuple[str, str]] = Counter()
    tasks_by_id: dict[str, dict[str, Any]] = {}
    task_total = 0
    for split in SPLITS:
        tasks = splits.get(split)
        if not isinstance(tasks, list):
            errors.append(f"missing task list for {split}")
            continue
        expected = 24 if split == "train" else 48
        if len(tasks) != expected:
            errors.append(f"{split} task count {len(tasks)} != {expected}")
        for task in tasks:
            task_total += 1
            if not isinstance(task, dict):
                errors.append(f"{split} has non-object task")
                continue
            task_id = task.get("id")
            family = task.get("family")
            if not isinstance(task_id, str) or task_id in ids:
                errors.append(f"invalid or duplicate task id: {task_id!r}")
            else:
                ids.add(task_id)
                tasks_by_id[task_id] = task
            if family not in FAMILIES:
                errors.append(f"{task_id}: unknown family {family!r}")
                continue
            counts[(split, family)] += 1
            generator = task.get("generator")
            if not isinstance(generator, dict):
                errors.append(f"{task_id}: generator metadata missing")
                continue
            node = generator.get("program")
            try:
                key = expression_key(node)
                poly = expression_polynomial(node)
                ops = operation_count(node)
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(f"{task_id}: malformed witness ({exc})")
                continue
            if key in expr_keys:
                errors.append(f"{task_id}: duplicate syntax witness")
            expr_keys.add(key)
            pkey = tuple(tuple(term) for term in canonical_polynomial(poly))
            if pkey in semantic:
                prior = semantic[pkey]
                errors.append(f"{task_id}: semantic duplicate of {prior}")
            else:
                semantic[pkey] = task_id
            if ops not in {4, 5}:
                errors.append(f"{task_id}: witness has {ops} operations")
            if task.get("max_tree_ops") != 5:
                errors.append(f"{task_id}: max_tree_ops is not 5")
            if not family_matches(family, node):
                errors.append(f"{task_id}: witness does not match {family}")
            degree = polynomial_degree(poly)
            norm = coefficient_l1(poly)
            if not difficulty_floor(poly):
                errors.append(
                    f"{task_id}: witness misses static difficulty floor")
            expected_metadata = {
                "seed": {"repeated_subtree": 82071, "balanced": 82072,
                         "skewed": 82073}[family],
                "program_key": key,
                "canonical_polynomial": canonical_polynomial(poly),
                "tree_ops": ops,
                "witness_ops": ops,
                "degree": degree,
                "coefficient_l1": norm,
            }
            for field, expected_value in expected_metadata.items():
                if generator.get(field) != expected_value:
                    errors.append(f"{task_id}: incorrect {field} metadata")
            task_coordinates: list[tuple[int, int]] = []
            observed_coordinates: set[tuple[int, int]] = set()
            held_coordinates: set[tuple[int, int]] = set()
            for row_kind in ("train", "held_out"):
                rows = task.get(row_kind)
                if not isinstance(rows, list) or len(rows) != 12:
                    errors.append(
                        f"{task_id}: {row_kind} must contain 12 rows")
                    continue
                for row in rows:
                    if (not isinstance(row, list) or len(row) != 3 or
                            any(isinstance(value, bool) or
                                not isinstance(value, int) for value in row)):
                        errors.append(f"{task_id}: malformed {row_kind} row")
                        continue
                    x_value, y_value, observed = row
                    pair = (x_value, y_value)
                    if pair not in GRID:
                        errors.append(f"{task_id}: input outside common grid")
                    if pair in coordinates:
                        errors.append(f"{task_id}: globally reused input pair")
                    coordinates.add(pair)
                    task_coordinates.append(pair)
                    if row_kind == "train":
                        observed_coordinates.add(pair)
                    else:
                        held_coordinates.add(pair)
                    if evaluate(poly, x_value, y_value) != observed:
                        errors.append(
                            f"{task_id}: output mismatches exact witness")
            if len(task_coordinates) == 24:
                if len(set(task_coordinates)) != 24:
                    errors.append(f"{task_id}: repeated row coordinate")
                if observed_coordinates & held_coordinates:
                    errors.append(f"{task_id}: observed and held rows overlap")
    for split, expected in (("train", 8), ("eval", 16)):
        for family in FAMILIES:
            if counts[(split, family)] != expected:
                actual = counts[(split, family)]
                errors.append(f"{split}/{family} count {actual} != {expected}")
    expected_pools, expected_rejections = independent_candidate_pools()
    generation_audit = corpus.get("generation_audit", {})
    if not isinstance(generation_audit, dict):
        errors.append("generation audit metadata is malformed")
        generation_audit = {}
    recorded_pool_counts = generation_audit.get("candidate_pool_counts", {})
    for family in ("repeated_subtree", "balanced", "skewed"):
        actual = len(expected_pools[family])
        recorded = recorded_pool_counts.get(family)
        if actual < 24:
            errors.append(
                f"independent {family} pool has only {actual} functions")
        if recorded != actual:
            errors.append(
                f"recorded {family} pool count {recorded} != {actual}")
    if generation_audit.get("rejected_counts") != expected_rejections:
        errors.append("recorded generation rejection counts differ")
    expected_tasks: dict[str, dict[str, Any]] = {}
    selected: list[tuple[str, Any]] = []
    split_seed = 82074
    for family in ("repeated_subtree", "balanced", "skewed"):
        ranked = sorted(expected_pools[family], key=lambda node:
                        seeded_digest(split_seed,
                                      f"{family}:{expression_key(node)}"))
        four_ops = [node for node in ranked if operation_count(node) == 4]
        five_ops = [node for node in ranked if operation_count(node) == 5]
        four_quota = min(8, len(four_ops))
        chosen = four_ops[:four_quota] + five_ops[:24 - four_quota]
        if len(chosen) < 24:
            chosen += four_ops[four_quota:four_quota + 24 - len(chosen)]
        selected.extend((family, node) for node in chosen)
    selected_sizes = Counter(operation_count(node) for _, node in selected)
    expected_size_counts = {"4": selected_sizes[4], "5": selected_sizes[5]}
    if generation_audit.get("selected_witness_ops") != expected_size_counts:
        errors.append(
            "selected witness-size counts differ from reconstruction")
    split_order = sorted(selected, key=lambda item: seeded_digest(
        split_seed, f"split:{item[0]}:{expression_key(item[1])}"))
    family_split_counts = Counter()
    split_task_counts = Counter()
    input_order = sorted(GRID, key=lambda pair: seeded_digest(
        82075, f"{pair[0]}:{pair[1]}"))
    for index, (family, node) in enumerate(split_order):
        split = ("train" if family_split_counts[("train", family)] < 8
                 else "eval")
        family_split_counts[(split, family)] += 1
        split_task_counts[split] += 1
        task_id = f"v2_{split}_{split_task_counts[split]:02d}"
        expected_tasks[task_id] = {
            "family": family,
            "program_key": expression_key(node),
            "input_pairs": input_order[index * 24:(index + 1) * 24],
        }
    for task_id, expected in expected_tasks.items():
        task = tasks_by_id.get(task_id)
        if task is None:
            errors.append(f"missing deterministically selected task {task_id}")
            continue
        generator = task.get("generator")
        if not isinstance(generator, dict):
            errors.append(f"{task_id}: missing generator metadata")
            continue
        witness = generator.get("program")
        try:
            actual_key = expression_key(witness)
        except (ValueError, TypeError, KeyError):
            continue
        if task.get("family") != expected["family"]:
            errors.append(f"{task_id}: family differs from seeded selection")
        if actual_key != expected["program_key"]:
            errors.append(f"{task_id}: witness differs from seeded selection")
        observed = task.get("train", [])
        held = task.get("held_out", [])
        if not isinstance(observed, list) or not isinstance(held, list):
            continue
        actual_pairs = []
        for row in observed + held:
            if isinstance(row, list) and len(row) == 3:
                actual_pairs.append((row[0], row[1]))
        if actual_pairs != expected["input_pairs"]:
            errors.append(f"{task_id}: rows differ from global seeded shuffle")
    bound = check_3_op_floor_exhaustively()
    if not bound["bound_holds"]:
        errors.append("static floor admits a <=3-op expression")
    return {
        "ok": not errors,
        "errors": errors,
        "task_total": task_total,
        "unique_task_ids": len(ids),
        "unique_expression_syntax": len(expr_keys),
        "unique_exact_functions": len(semantic),
        "independently_rebuilt_pool_counts": {
            family: len(expected_pools[family])
            for family in ("repeated_subtree", "balanced", "skewed")},
        "globally_unique_input_pairs": len(coordinates),
        "family_counts": {f"{split}/{family}": counts[(split, family)]
                          for split in SPLITS for family in FAMILIES},
        "static_floor_proof": bound,
    }


def exact_function_keys(corpus: Any) -> set[
        tuple[tuple[int, int, int], ...]]:
    """Extract exact polynomial keys from a corpus with this auditor."""
    result = set()
    for split in SPLITS:
        for task in corpus["splits"][split]:
            result.add(tuple(tuple(term) for term in canonical_polynomial(
                expression_polynomial(task["generator"]["program"]))))
    return result


def audit_protocol(protocol: Any, protocol_hash: str,
                   corpus_hash: str) -> dict[str, Any]:
    """Check frozen protocol bytes and independently pinned source digests."""
    errors = []
    if not isinstance(protocol, dict):
        return {"ok": False, "errors": ["protocol root is not an object"]}
    expected_sources = protocol.get("corpus", {}).get("generator_source", {})
    actual_sources = {
        "scripts/generate_benchmark_pre_eval_v2.py": digest(GENERATOR_PATH),
        "rsi_programmer/benchmark_v2_corpus.py": digest(CORPUS_MODULE_PATH),
    }
    if protocol_hash != PINNED_PROTOCOL_SHA256:
        errors.append("protocol bytes differ from independent pinned SHA256")
    if corpus_hash != PINNED_CORPUS_SHA256:
        errors.append("corpus bytes differ from independent pinned SHA256")
    if protocol.get("status") != "frozen-before-measurement":
        errors.append("protocol is not marked frozen-before-measurement")
    baseline = "14af0d7de26ba1256fadf3b2722513d3bf9577fc"
    if protocol.get("baseline_revision") != baseline:
        errors.append("protocol baseline revision differs")
    if protocol.get("corpus", {}).get("sha256") != corpus_hash:
        errors.append("protocol does not pin the corpus bytes")
    if expected_sources != PINNED_GENERATOR_SHA256:
        errors.append("protocol generator pins differ from independent pins")
    if actual_sources != PINNED_GENERATOR_SHA256:
        errors.append("current generator source bytes differ from pins")
    expected_training = {
        "task_split": "train", "training_task_count": 24,
        "training_rows_only": True, "positive_cap": 16,
        "negative_cap": 256, "sampling_seed": 82078,
        "epochs": 40, "learning_rate": 0.05, "seeds": [0, 1, 2],
    }
    training = protocol.get("training", {})
    for key, value in expected_training.items():
        if training.get(key) != value:
            errors.append(f"protocol training field {key} differs")
    evaluation = protocol.get("evaluation", {})
    if evaluation.get("candidate_scan_budgets") != [
            512, 2048, 8192, 32768]:
        errors.append("protocol candidate budgets differ")
    if evaluation.get("tree_bound") != 5:
        errors.append("protocol tree bound differs")
    if evaluation.get("task_count") != 48:
        errors.append("protocol evaluation task count differs")
    if protocol.get("controls") != [
            "streaming", "heuristic", "seeded_random", "untrained_snn",
            "trained_snn", "mlp", "recurrent"]:
        errors.append("protocol controls or their order differ")
    return {"ok": not errors, "errors": errors,
            "protocol_sha256": protocol_hash,
            "corpus_sha256": corpus_hash,
            "generator_source_sha256": actual_sources}


def load_json(path: Path) -> Any:
    """Load UTF-8 JSON from one path."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--write", type=Path,
                        help="write JSON report to this path")
    args = parser.parse_args()
    report: dict[str, Any] = {"protocol_path": str(args.protocol),
                              "corpus_path": str(args.corpus)}
    report["input_sha256"] = {
        "protocol": digest(args.protocol) if args.protocol.exists() else None,
        "corpus": digest(args.corpus) if args.corpus.exists() else None,
        "generator": (digest(GENERATOR_PATH)
                      if GENERATOR_PATH.exists() else None),
        "corpus_module": (digest(CORPUS_MODULE_PATH)
                          if CORPUS_MODULE_PATH.exists() else None),
        "v1_corpus": (digest(V1_CORPUS_PATH)
                      if V1_CORPUS_PATH.exists() else None),
    }
    corpus = load_json(args.corpus)
    report["corpus_audit"] = audit_corpus(corpus)
    if args.protocol.exists():
        protocol = load_json(args.protocol)
        report["protocol_audit"] = audit_protocol(
            protocol, report["input_sha256"]["protocol"],
            report["input_sha256"]["corpus"],
        )
        v1_corpus = load_json(V1_CORPUS_PATH)
        v2_functions = exact_function_keys(corpus)
        v1_functions = exact_function_keys(v1_corpus)
        overlap = sorted(v2_functions & v1_functions)
        report["v1_disjointness"] = {
            "v1_unique_functions": len(v1_functions),
            "v2_unique_functions": len(v2_functions),
            "exact_function_overlap": len(overlap),
            "overlap_examples": [list(map(list, poly))
                                 for poly in overlap[:5]],
            "zero_overlap": not overlap,
        }
        report["ok"] = (
            report["corpus_audit"]["ok"]
            and report["protocol_audit"]["ok"]
            and report["v1_disjointness"]["zero_overlap"]
        )
    else:
        report["ok"] = report["corpus_audit"]["ok"]
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
