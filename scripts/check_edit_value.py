"""Verify the frozen edit-value task bank and checker-side references."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from rsi_programmer.array_curriculum import probes, teachers
from rsi_programmer.array_execution import execute
from rsi_programmer.array_tasks import TASKS, oracle

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "fixtures/edit_value_v1"


def _keep_nonpositive(values: list[int]) -> list[int]:
    return [value for value in values if value <= 0]


def _absolute_value(values: list[int]) -> list[int]:
    return [abs(value) for value in values]


def _duplicate_positive(values: list[int]) -> list[int]:
    return [part for value in values if value > 0
            for part in (value, value)]


def _add_output_index(values: list[int]) -> list[int]:
    output: list[int] = []
    for value in values:
        if value > 0:
            output.append(value + len(output))
    return output


def _first_original_then_negate(values: list[int]) -> list[int]:
    output: list[int] = []
    for value in values:
        if len(output) == 0:
            output.append(value)
        if len(output) > 0:
            output.append(-value)
    return output


def _negative_magnitude_else_one(values: list[int]) -> list[int]:
    return [abs(value) if value < 0 else 1 for value in values]


REFERENCES: dict[str, Callable[[list[int]], list[int]]] = {
    "keep_nonpositive": _keep_nonpositive,
    "absolute_value": _absolute_value,
    "duplicate_positive": _duplicate_positive,
    "add_output_index_for_positive": _add_output_index,
    "first_original_then_negate": _first_original_then_negate,
    "negative_magnitude_if_negative_else_one":
        _negative_magnitude_else_one,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pre_fit() -> dict[str, object]:
    protocol = json.loads((BANK / "protocol.json").read_text())
    examples = json.loads((BANK / "fresh_examples.json").read_text())
    hashes = json.loads((BANK / "sha256.json").read_text())[
        "files_sha256"]
    if protocol.get("status") != "frozen-before-edit-value-solver-trial":
        raise ValueError("task bank was not frozen before solver trials")
    if len(protocol["fresh_tasks"]) != 6 or len(examples) != 6:
        raise ValueError("expected six frozen fresh tasks")
    if protocol["development_tasks"] != list(TASKS):
        raise ValueError("development tasks must remain separate")
    if protocol["retention_tasks"]:
        raise ValueError("unqualified prior tasks cannot be called retention")
    if protocol["baseline_recheck_tasks"] != list(TASKS):
        raise ValueError("original tasks must stay in baseline recheck")
    if any(row["task_family"] != "bounded_integer_list"
           for row in protocol["fresh_tasks"]):
        raise ValueError("fresh tasks may not claim new language families")
    descriptors = {row["task_id"]: row for row in protocol["fresh_tasks"]}
    example_map = {row["task_id"]: row for row in examples}
    if set(descriptors) != set(example_map):
        raise ValueError("task descriptor and case IDs differ")
    rows = probes()
    known = {tuple(tuple(execute(program, values)) for values in rows)
             for program in teachers()}
    known.update(tuple(tuple(oracle(name, values)) for values in rows)
                 for name in TASKS)
    signatures: set[tuple[tuple[int, ...], ...]] = set()
    for task_id, descriptor in descriptors.items():
        reference = REFERENCES[descriptor["reference"]]
        tokens = tuple(descriptor["witness"])
        observed_signature = tuple(tuple(reference(values)) for values in rows)
        witness_signature = tuple(tuple(execute(tokens, values))
                                  for values in rows)
        if observed_signature != witness_signature:
            raise ValueError(f"reference and witness disagree: {task_id}")
        if observed_signature in known or observed_signature in signatures:
            raise ValueError(
                f"fresh task signature is not distinct: {task_id}")
        signatures.add(observed_signature)
        visible = example_map[task_id]["examples"]
        if len(visible) != 5:
            raise ValueError(f"wrong visible case count: {task_id}")
        for pair in visible:
            if pair["output"] != reference(pair["input"]):
                raise ValueError(f"reference mismatch in examples: {task_id}")

    for name in ("protocol.json", "fresh_examples.json"):
        if sha256(BANK / name) != hashes.get(name):
            raise ValueError(f"frozen fixture hash mismatch: {name}")

    return {"fresh_tasks": len(descriptors),
            "fresh_signatures_distinct": len(signatures),
            "scope": protocol["domain"],
            "protocol_sha256": sha256(BANK / "protocol.json"),
            "examples_sha256": sha256(BANK / "fresh_examples.json")}


def verify() -> dict[str, object]:
    result = verify_pre_fit()
    examples = json.loads((BANK / "fresh_examples.json").read_text())
    cases = json.loads((BANK / "held_cases.json").read_text())
    hashes = json.loads((BANK / "sha256.json").read_text())[
        "files_sha256"]
    if len(cases) != 6:
        raise ValueError("expected six held tasks")
    descriptors = {row["task_id"]: row for row in json.loads(
        (BANK / "protocol.json").read_text())["fresh_tasks"]}
    references = {row["task_id"]: REFERENCES[row["reference"]]
                  for row in descriptors.values()}
    example_map = {row["task_id"]: row for row in examples}
    case_map = {row["task_id"]: row for row in cases}
    if set(descriptors) != set(case_map):
        raise ValueError("task descriptor and held IDs differ")
    for task_id, task_cases in case_map.items():
        reference = references[task_id]
        visible = example_map[task_id]["examples"]
        held_inputs = task_cases["held_inputs"]
        held_expected = task_cases["expected"]
        if len(held_inputs) != 20:
            raise ValueError(f"wrong held case count: {task_id}")
        if len(held_inputs) != len(held_expected):
            raise ValueError(f"held input/output lengths differ: {task_id}")
        if {tuple(pair["input"]) for pair in visible} & {
                tuple(values) for values in held_inputs}:
            raise ValueError(f"held rows overlap examples: {task_id}")
        for values, expected in zip(held_inputs, held_expected):
            if expected != reference(values):
                raise ValueError(f"reference mismatch in held rows: {task_id}")
    if sha256(BANK / "held_cases.json") != hashes.get("held_cases.json"):
        raise ValueError("frozen fixture hash mismatch: held_cases.json")
    result["held_cases_per_task"] = 20
    result["held_cases_sha256"] = sha256(BANK / "held_cases.json")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
