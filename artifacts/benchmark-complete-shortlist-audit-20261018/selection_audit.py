#!/usr/bin/env python3
"""Independent stdlib audit of the frozen train-only selector evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SELECTION_PROTOCOL_PATH = (
    ROOT / "fixtures/benchmark_shortlist_selection_v1.json"
)
SELECTION_PROTOCOL_SHA256 = (
    "64eb64fb2b029be4736f9170df8fa4d2956c954d85e8133952f7981a416ac616"
)
V2_PROTOCOL_PATH = ROOT / "fixtures/benchmark_pre_eval_v2.json"
V2_PROTOCOL_SHA256 = (
    "cf83dc077dc7960d4e576d898d842484655ac2c61d0e36128da6cda5ae8e7015"
)
CORPUS_PATH = ROOT / "fixtures/benchmark_pre_eval_corpus_v2.json"
CORPUS_SHA256 = (
    "a8fd6add90b34da732da8dc96f2c0fafcabc2f017a17cdef4b31303eec4ba921"
)
REFERENCE_PATH = (
    ROOT / "artifacts/benchmark-pre-eval-v2-20261018/training.json"
)
REFERENCE_SHA256 = (
    "47b02c532f5a11faa4808f5ae9646ba976ab8a56d80a0aadfcaf062239d82b6b"
)
SELECTION_SCRIPT_PATH = ROOT / "scripts/select_shortlist_method.py"
SELECTION_SCRIPT_SHA256 = (
    "75c54033cbd794c5e9b134d77b28e683a63ddf19a0e56ef7e33a0d65497fcc2e"
)
SELECTION_DIR = ROOT / "artifacts/benchmark-shortlist-selection-run-20261018"
EXTERNAL_RECEIPT_PATH = (
    ROOT / "artifacts/benchmark-shortlist-validation-20261018/selection.json"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def polynomial(node: Any) -> dict[tuple[int, int], int]:
    if not isinstance(node, dict):
        raise ValueError("malformed expression")
    op = node.get("op")
    if op == "input":
        name = node.get("name")
        if name == "x":
            return {(1, 0): 1}
        if name == "y":
            return {(0, 1): 1}
        raise ValueError("unknown expression input")
    if op not in {"add", "sub", "mul"}:
        raise ValueError("unknown expression operator")
    left, right = polynomial(node.get("left")), polynomial(node.get("right"))
    if op in {"add", "sub"}:
        out = dict(left)
        sign = 1 if op == "add" else -1
        for power, value in right.items():
            out[power] = out.get(power, 0) + sign * value
    else:
        out: dict[tuple[int, int], int] = defaultdict(int)
        for (lx, ly), lv in left.items():
            for (rx, ry), rv in right.items():
                out[lx + rx, ly + ry] += lv * rv
    return {key: value for key, value in out.items() if value}


def graph_polynomial(graph: Any) -> dict[tuple[int, int], int]:
    if not isinstance(graph, dict) or set(graph) != {
        "input_types",
        "nodes",
        "output",
    }:
        raise ValueError("malformed graph")
    if graph["input_types"] != {"x": "Int", "y": "Int"}:
        raise ValueError("unexpected graph inputs")
    values: list[dict[tuple[int, int], int]] = []
    for i, node in enumerate(graph["nodes"]):
        if (
            not isinstance(node, dict)
            or set(node) != {"op", "type", "args"}
            or node["type"] != "Int"
            or node["op"] not in {"add", "sub", "mul"}
        ):
            raise ValueError("malformed graph node")
        args = node["args"]
        if not isinstance(args, list) or len(args) != 2:
            raise ValueError("graph nodes require two arguments")
        operands = []
        for ref in args:
            if ref == "x":
                operands.append({(1, 0): 1})
            elif ref == "y":
                operands.append({(0, 1): 1})
            elif type(ref) is int and 0 <= ref < i:
                operands.append(values[ref])
            else:
                raise ValueError("invalid or forward graph reference")
        left, right = operands
        if node["op"] in {"add", "sub"}:
            out = dict(left)
            sign = 1 if node["op"] == "add" else -1
            for power, value in right.items():
                out[power] = out.get(power, 0) + sign * value
        else:
            out = defaultdict(int)
            for (lx, ly), lv in left.items():
                for (rx, ry), rv in right.items():
                    out[lx + rx, ly + ry] += lv * rv
        values.append({key: value for key, value in out.items() if value})
    output = graph["output"]
    if output == "x":
        return {(1, 0): 1}
    if output == "y":
        return {(0, 1): 1}
    if type(output) is int and 0 <= output < len(values):
        return values[output]
    raise ValueError("invalid graph output")


def eval_graph(graph: dict[str, Any], x: int, y: int) -> int:
    values = []
    for node in graph["nodes"]:
        args = [
            x if ref == "x" else y if ref == "y" else values[ref]
            for ref in node["args"]
        ]
        if node["op"] == "add":
            values.append(args[0] + args[1])
        elif node["op"] == "sub":
            values.append(args[0] - args[1])
        else:
            values.append(args[0] * args[1])
    output = graph["output"]
    return x if output == "x" else y if output == "y" else values[output]


def tree_ops(graph: dict[str, Any]) -> int:
    def count(ref: Any) -> int:
        if ref in ("x", "y"):
            return 0
        if type(ref) is not int or not 0 <= ref < len(graph["nodes"]):
            raise ValueError("invalid graph reference")
        return 1 + sum(count(arg) for arg in graph["nodes"][ref]["args"])

    return count(graph["output"])


def selected_schedule(tasks: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    controls = ("cheap_only", "heuristic", "trained_snn", "mlp", "recurrent")
    settings = ((64, 4), (256, 16))
    sequence = []
    for task in tasks:
        for repeat in (0, 1):
            entries = [("streaming", "streaming", 0, 0)]
            entries.extend(
                ("complete_shortlist", control, window, quota)
                for window, quota in settings
                for control in controls
            )
            ordered = sorted(
                entries,
                key=lambda entry: hashlib.sha256(
                    (
                        f"82091:{task['id']}:{entry[0]}:{entry[1]}:"
                        f"{entry[2]}:{entry[3]}"
                    ).encode()
                ).digest(),
            )
            if repeat:
                ordered.reverse()
            sequence.extend(
                (str(task["id"]), repeat, *entry) for entry in ordered
            )
    return sequence


def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        row[k]
        for k in (
            "task_id",
            "repeat",
            "engine",
            "control",
            "window_size",
            "shortlist_size",
        )
    )


def behavior_without_time(row: dict[str, Any]) -> dict[str, Any]:
    search = row.get("search") or {}
    stable = {
        key: value
        for key, value in search.items()
        if not key.endswith(("_seconds", "_ns")) and key != "graph"
    }
    return {
        "graph": row.get("graph"),
        "search": stable,
        "validation": row.get("validation"),
    }


def audit(bundle: Path = SELECTION_DIR) -> dict[str, Any]:
    errors: list[str] = []
    protocol_bytes = SELECTION_PROTOCOL_PATH.read_bytes()
    v2_bytes = V2_PROTOCOL_PATH.read_bytes()
    corpus_bytes = CORPUS_PATH.read_bytes()
    reference_bytes = REFERENCE_PATH.read_bytes()
    if sha256(protocol_bytes) != SELECTION_PROTOCOL_SHA256:
        raise ValueError("selection protocol hash mismatch")
    if sha256(v2_bytes) != V2_PROTOCOL_SHA256:
        raise ValueError("v2 protocol hash mismatch")
    if sha256(corpus_bytes) != CORPUS_SHA256:
        raise ValueError("v2 corpus hash mismatch")
    if sha256(reference_bytes) != REFERENCE_SHA256:
        raise ValueError("frozen model reference hash mismatch")
    if sha256(SELECTION_SCRIPT_PATH.read_bytes()) != SELECTION_SCRIPT_SHA256:
        raise ValueError("frozen selection script hash mismatch")
    protocol, v2, corpus, reference = (
        json.loads(protocol_bytes),
        json.loads(v2_bytes),
        json.loads(corpus_bytes),
        json.loads(reference_bytes),
    )
    manifest = read_json(bundle / "manifest.json")
    selection = read_json(bundle / "selection.json")
    training = read_json(bundle / "training.json")
    external = read_json(EXTERNAL_RECEIPT_PATH)
    raw_bytes = (bundle / "raw.jsonl").read_bytes()
    training_bytes = (bundle / "training.json").read_bytes()
    selection_bytes = (bundle / "selection.json").read_bytes()
    expected_files = {
        "raw.jsonl": sha256(raw_bytes),
        "training.json": sha256(training_bytes),
        "selection.json": sha256(selection_bytes),
    }
    if manifest.get("files_sha256") != expected_files:
        errors.append("selection manifest file hashes differ from bytes")
    if manifest.get("protocol_sha256") != SELECTION_PROTOCOL_SHA256:
        errors.append("selection run protocol hash differs")
    if selection.get("protocol_sha256") != SELECTION_PROTOCOL_SHA256:
        errors.append("selection report protocol hash differs")
    if selection.get("corpus_sha256") != CORPUS_SHA256:
        errors.append("selection corpus hash differs")
    if selection.get("v2_protocol_sha256") != V2_PROTOCOL_SHA256:
        errors.append("selection v2 protocol hash differs")
    if selection.get("model_reference_sha256") != REFERENCE_SHA256:
        errors.append("selection model reference hash differs")
    if selection.get("selection_script_sha256") != SELECTION_SCRIPT_SHA256:
        errors.append("selection report script hash differs")
    if selection.get("raw_jsonl_sha256") != sha256(raw_bytes):
        errors.append("selection raw hash differs")
    if selection.get("training_json_sha256") != sha256(training_bytes):
        errors.append("selection training hash differs")
    if manifest.get("selection_json_sha256") != sha256(selection_bytes):
        errors.append("manifest selection hash differs")
    train_tasks = corpus["splits"]["train"]
    eval_ids = {str(task["id"]) for task in corpus["splits"]["eval"]}
    train_ids = {str(task["id"]) for task in train_tasks}
    rows: list[dict[str, Any]] = []
    keyset = set()
    sequence = []
    expected_poly = {
        str(task["id"]): polynomial(task["generator"]["program"])
        for task in train_tasks
    }
    task_map = {str(task["id"]): task for task in train_tasks}
    for line_number, line in enumerate(raw_bytes.splitlines(), 1):
        try:
            row = json.loads(line)
            key = selection_key(row)
            sequence.append(key)
            if key in keyset:
                errors.append(f"duplicate selection key at line {line_number}")
            keyset.add(key)
            rows.append(row)
            task_id = str(row["task_id"])
            if task_id not in train_ids or task_id in eval_ids:
                errors.append(f"non-training task used at line {line_number}")
            if row.get("seed") != 0 or row.get("root_signature_cap") != 131072:
                errors.append(
                    f"selection seed/cap mismatch at line {line_number}"
                )
            if row.get("error"):
                errors.append(f"selection row error at line {line_number}")
            if type(row.get("accepted")) is not bool:
                errors.append(f"invalid accepted flag at line {line_number}")
            if (
                type(row.get("attempt_cpu_ns")) is not int
                or row["attempt_cpu_ns"] < 0
            ):
                errors.append(f"invalid CPU time at line {line_number}")
            if (
                type(row.get("attempt_wall_ns")) is not int
                or row["attempt_wall_ns"] < 0
            ):
                errors.append(f"invalid wall time at line {line_number}")
            search = row.get("search") or {}
            if search.get("signature_evaluations", 0) > 131072:
                errors.append(
                    f"selection root cap exceeded at line {line_number}"
                )
            graph = row.get("graph")
            expected_digest = ""
            actual_digest = ""
            checks = {
                "graph_valid": False,
                "observed_rows_valid": False,
                "exact_polynomial_valid": False,
            }
            if graph is not None:
                try:
                    graph_poly = graph_polynomial(graph)
                    expected_triples = tuple(
                        (x, y, value)
                        for (x, y), value in sorted(
                            expected_poly[task_id].items()
                        )
                    )
                    actual_triples = tuple(
                        (x, y, value)
                        for (x, y), value in sorted(graph_poly.items())
                    )
                    expected_digest = sha256(
                        json.dumps(
                            expected_triples, separators=(",", ":")
                        ).encode()
                    )
                    actual_digest = sha256(
                        json.dumps(
                            actual_triples, separators=(",", ":")
                        ).encode()
                    )
                    checks["graph_valid"] = (
                        tree_ops(graph) <= 5 and len(graph["nodes"]) <= 5
                    )
                    checks["observed_rows_valid"] = all(
                        eval_graph(graph, int(x), int(y)) == int(target)
                        for x, y, target in task_map[task_id]["train"]
                    )
                    checks["exact_polynomial_valid"] = (
                        graph_poly == expected_poly[task_id]
                    )
                except (TypeError, ValueError, KeyError, IndexError):
                    pass
            for flag, value in checks.items():
                if row.get("validation", {}).get(flag) is not value:
                    errors.append(
                        f"independent {flag} mismatch at line {line_number}"
                    )
            if graph is not None and (
                row.get("validation", {}).get("expected_polynomial_sha256")
                != expected_digest
                or row.get("validation", {}).get("actual_polynomial_sha256")
                != actual_digest
            ):
                errors.append(
                    f"polynomial hash mismatch at line {line_number}"
                )
            should_accept = all(checks.values())
            if row.get("accepted") is not should_accept:
                errors.append(f"accepted flag mismatch at line {line_number}")
            if row.get("behavior_sha256") != sha256(
                json.dumps(
                    behavior_without_time(row),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ):
                errors.append(
                    f"behavior fingerprint mismatch at line {line_number}"
                )
        except Exception as exc:
            errors.append(
                f"malformed selection row {line_number}: "
                f"{type(exc).__name__}: {exc}"
            )
    expected_sequence = selected_schedule(train_tasks)
    if sequence != expected_sequence:
        errors.append("selection raw order differs from frozen schedule")
    if len(rows) != 528 or keyset != set(expected_sequence):
        errors.append("selection raw matrix incomplete or unexpected")

    repeated: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        repeat_key = tuple(
            row[key]
            for key in (
                "task_id",
                "engine",
                "control",
                "window_size",
                "shortlist_size",
            )
        )
        repeated[repeat_key].append(row)
    for repeat_key, attempts in repeated.items():
        by_repeat = {row["repeat"]: row for row in attempts}
        if set(by_repeat) != {0, 1} or behavior_without_time(
            by_repeat[0]
        ) != behavior_without_time(by_repeat[1]):
            errors.append(
                f"selection repeat behavior differs for {repeat_key}"
            )

    by_config: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    streams: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["engine"] == "streaming":
            streams[str(row["task_id"])].append(row)
        else:
            by_config[
                (row["window_size"], row["shortlist_size"], row["control"])
            ].append(row)
    stream_count = sum(
        bool(row["accepted"]) for group in streams.values() for row in group
    )
    stream_rows = [row for group in streams.values() for row in group]
    settings = ((64, 4), (256, 16))
    controls = ("cheap_only", "heuristic", "trained_snn", "mlp", "recurrent")
    computed_methods = []
    for window, quota in settings:
        control_rows = {}
        eligible = True
        all_cpu = []
        for control in controls:
            group = by_config[(window, quota, control)]
            accepted = sum(bool(row["accepted"]) for row in group)
            mean_cpu = statistics.mean(row["attempt_cpu_ns"] for row in group)
            streaming_accepted = stream_count
            control_rows[control] = {
                "attempts": len(group),
                "accepted": accepted,
                "streaming_accepted": streaming_accepted,
                "coverage_fraction": accepted / len(group) if group else None,
                "mean_attempt_cpu_ns": mean_cpu,
            }
            eligible = (
                eligible
                and len(group) == 48
                and accepted >= streaming_accepted
            )
            all_cpu.extend(row["attempt_cpu_ns"] for row in group)
        computed_methods.append(
            {
                "window_size": window,
                "shortlist_size": quota,
                "eligible": eligible,
                "mean_attempt_cpu_ns_across_five_controls": statistics.mean(
                    all_cpu
                ),
                "controls": control_rows,
            }
        )
    eligible_methods = [
        method for method in computed_methods if method["eligible"]
    ]
    selected_method = (
        min(
            eligible_methods,
            key=lambda method: (
                method["mean_attempt_cpu_ns_across_five_controls"],
                method["window_size"],
            ),
        )
        if eligible_methods
        else None
    )
    observed_selection = selection.get("selected")
    expected_selected = (
        {
            "window_size": selected_method["window_size"],
            "shortlist_size": selected_method["shortlist_size"],
            "cheap_row_index": 0,
        }
        if selected_method
        else None
    )
    observed_pair = (
        {
            key: observed_selection[key]
            for key in ("window_size", "shortlist_size")
        }
        if isinstance(observed_selection, dict)
        else None
    )
    if observed_pair != (
        expected_selected
        and {
            key: expected_selected[key]
            for key in ("window_size", "shortlist_size")
        }
    ):
        errors.append("selected parameter pair differs from recomputed rule")
    reported_methods = selection.get("methods", [])
    if len(reported_methods) != len(computed_methods):
        errors.append("selection method summary row count differs")
    for reported, computed in zip(reported_methods, computed_methods):
        if (
            reported.get("window_size") != computed["window_size"]
            or reported.get("shortlist_size") != computed["shortlist_size"]
            or reported.get("eligible") is not computed["eligible"]
            or reported.get("mean_attempt_cpu_ns_across_five_controls")
            != computed["mean_attempt_cpu_ns_across_five_controls"]
        ):
            errors.append("selection method summary differs from raw rows")
        for control, counts in computed["controls"].items():
            summary = reported.get("controls", {}).get(control, {})
            for key in ("attempts", "accepted", "streaming_accepted"):
                if summary.get(key) != counts[key]:
                    errors.append(
                        f"selection summary {control}.{key} differs "
                        "from raw rows"
                    )
            if (
                summary.get("mean_attempt_cpu_ns")
                != counts["mean_attempt_cpu_ns"]
            ):
                errors.append(
                    f"selection summary {control} CPU mean differs "
                    "from raw rows"
                )
    if (
        selection.get("status") != "complete"
        or manifest.get("status") != "complete"
    ):
        errors.append("selection run is not complete")
    if (
        selection.get("matrix_valid") is not True
        or manifest.get("matrix_valid") is not True
    ):
        errors.append("selection matrix validity claim is false")
    if (
        selection.get("source_unchanged") is not True
        or manifest.get("source_unchanged") is not True
    ):
        errors.append("selection source stability claim is false")
    if selection.get("source_before") != selection.get("source_after"):
        errors.append("selection source snapshots differ")
    if manifest.get("source_before") != manifest.get("source_after"):
        errors.append("manifest source snapshots differ")
    if (
        training.get("status") != "complete"
        or training.get("evaluation_fits") != 0
    ):
        errors.append("selection training record incomplete or has eval fits")
    if training.get("training_task_ids") != [
        str(t["id"]) for t in train_tasks
    ]:
        errors.append(
            "selection training task list differs from frozen train split"
        )
    if training.get("seed") != 0 or training.get("epochs") != 40:
        errors.append("selection training seed or epoch count differs")
    expected_groups = training.get("training_groups", {}).get("groups", [])
    if len(expected_groups) != 24 or any(
        group.get("status") != "ok" for group in expected_groups
    ):
        errors.append("selection training groups incomplete")
    if [group.get("task_id") for group in expected_groups] != [
        str(task["id"]) for task in train_tasks
    ]:
        errors.append(
            "selection group task order differs from frozen train split"
        )
    for group in expected_groups:
        counts = group.get("label_counts", {})
        selected_counts = group.get("selected_counts", {})
        if (
            group.get("positive_cap") != 16
            or group.get("negative_cap") != 256
            or group.get("sampling_seed") != 82078
            or selected_counts.get("positive")
            != min(counts.get("positive", -1), 16)
            or selected_counts.get("negative")
            != min(counts.get("negative", -1), 256)
        ):
            errors.append(
                f"selection group caps/counts invalid: {group.get('task_id')}"
            )
    if not training.get("model_weight_hashes_match_v2_reference"):
        errors.append("selection model hash checkpoint failed")
    seed0 = next(item for item in reference["seeds"] if item["seed"] == 0)
    for control in ("heuristic", "trained_snn", "mlp", "recurrent"):
        expected_weight = seed0["controls"][control][
            "weights_after_fit_sha256"
        ]
        actual_weight = (
            training.get("models", {})
            .get(control, {})
            .get("weights_after_fit_sha256")
        )
        if actual_weight != expected_weight:
            errors.append(f"selection fitted weights differ for {control}")
    if training.get("frozen_model_fingerprints") != selection.get(
        "final_model_fingerprints"
    ):
        errors.append("frozen selection weight fingerprints differ")
    if training.get("frozen_model_state_fingerprints") != selection.get(
        "final_model_state_fingerprints"
    ):
        errors.append("frozen selection state fingerprints differ")
    for control in ("heuristic", "trained_snn", "mlp", "recurrent"):
        model = training.get("models", {}).get(control, {})
        if model.get("fingerprint_after_fit") != training.get(
            "frozen_model_state_fingerprints", {}
        ).get(control):
            errors.append(f"frozen model state mismatch for {control}")
    streaming_report = selection.get("streaming", {})
    if (
        streaming_report.get("attempts") != len(stream_rows)
        or streaming_report.get("accepted") != stream_count
        or streaming_report.get("mean_attempt_cpu_ns")
        != statistics.mean(row["attempt_cpu_ns"] for row in stream_rows)
    ):
        errors.append(
            "streaming selection reference summary differs from raw rows"
        )
    if selection.get("repeats_behavior_identical") is not True:
        errors.append("selection report claims nondeterministic repeats")

    if external.get("exit_code") != 0:
        errors.append("external selection command failed")
    if (
        external.get("child_total_cpu_seconds") is None
        or external.get("cli_wall_seconds") is None
    ):
        errors.append("external full-process selection receipt is incomplete")
    external_cpu = round(external["child_total_cpu_seconds"] * 1_000_000_000)
    external_wall = round(external["cli_wall_seconds"] * 1_000_000_000)
    protocol_evidence = json.loads(
        (ROOT / "fixtures/benchmark_complete_shortlist_v1.json").read_bytes()
    )["selection_evidence"]
    for key, expected in (
        (
            "path",
            "artifacts/benchmark-shortlist-selection-run-20261018/"
            "manifest.json",
        ),
        ("sha256", sha256((bundle / "manifest.json").read_bytes())),
        (
            "execution_path",
            "artifacts/benchmark-shortlist-validation-20261018/selection.json",
        ),
        ("execution_sha256", sha256(EXTERNAL_RECEIPT_PATH.read_bytes())),
        ("actual_attempts", 528),
        ("status", "complete"),
        ("selected", expected_selected),
        ("execution_charge_cpu_ns", external_cpu),
        ("execution_charge_wall_ns", external_wall),
    ):
        if protocol_evidence.get(key) != expected:
            errors.append(f"main protocol selection_evidence.{key} differs")
    if protocol_evidence.get("files_sha256") != expected_files:
        errors.append("main protocol selection evidence file hashes differ")
    if protocol["selection_script"]["sha256"] != SELECTION_SCRIPT_SHA256:
        errors.append("selection protocol does not pin the audited script")
    if selection.get("selected") != observed_pair:
        errors.append("selection output does not pin recomputed setting")
    return {
        "report_type": "independent-train-only-shortlist-selection-audit",
        "selection_protocol_sha256": SELECTION_PROTOCOL_SHA256,
        "main_protocol_sha256": sha256(
            (
                ROOT / "fixtures/benchmark_complete_shortlist_v1.json"
            ).read_bytes()
        ),
        "raw_rows": len(rows),
        "expected_raw_rows": len(expected_sequence),
        "streaming_accepted": stream_count,
        "methods": computed_methods,
        "selected": expected_selected,
        "external_selection_charge_cpu_ns": external_cpu,
        "external_selection_charge_wall_ns": external_wall,
        "errors": errors,
        "passed": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=SELECTION_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.bundle)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
