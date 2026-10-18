"""Render post-run shortlist cost and work results from frozen outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WORK_AXES = {
    "cheap_row_operations": "cheap_candidate_row_operations",
    "root_calls": "signature_evaluations",
    "policy_scored_features": "scored_descriptors",
    "dedup_signature_evaluations": "duplicate_signature_evaluations",
    "descriptor_replays": "descriptor_replays",
}
COUNTS = ("cheap_only", "streaming", "full_cached", "complete_shortlist")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from one completed run file."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read the completed JSON-lines observations."""
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def config_key(engine: str, control: str) -> tuple[str, str]:
    return engine, control


def point_rows(
    raw: list[dict[str, Any]],
    engine: str,
    control: str,
    budget: int,
) -> list[dict[str, Any]]:
    """Select a complete combined-repeat cell from raw observations."""
    return [
        row for row in raw
        if row.get("engine") == engine
        and row.get("control") == control
        and row.get("budget") == budget
    ]


def verify_cell(rows: list[dict[str, Any]], expected: int) -> None:
    """Reject sparse, duplicate, failed, or censored full-success cells."""
    keys = {
        (row.get("seed"), row.get("task_id"), row.get("repeat"))
        for row in rows
    }
    if len(rows) != expected or len(keys) != expected:
        raise ValueError("first-success raw cell is incomplete or duplicated")
    if any(not row.get("accepted") or row.get("censored") for row in rows):
        raise ValueError("summary full success disagrees with raw outcomes")
    if any(row.get("error") for row in rows):
        raise ValueError("first-success raw cell contains an error")


def render(run_dir: Path) -> str:
    """Build a text report; call only after the run is complete."""
    aggregate = read_json(run_dir / "aggregate.json")
    training = read_json(run_dir / "training.json")
    raw = read_jsonl(run_dir / "raw.jsonl")
    memory = read_jsonl(run_dir / "memory.jsonl")
    if training.get("status") != "complete":
        raise ValueError("training record is not complete")
    seeds = training.get("seeds", [])
    if not seeds:
        raise ValueError("training record has no fitted seeds")
    expected_attempts = len(seeds) * 48 * 2
    full_success = {
        (row["engine"], row["control"]): row
        for row in aggregate["matched_success_cpu_cost"]
        if row["repeat"] is None and row["target"] == 1.0
    }
    configs = [
        ("streaming", "streaming"),
        ("cheap_only", "cheap_only"),
        *[(engine, control)
          for control in ("heuristic", "trained_snn", "mlp", "recurrent")
          for engine in ("full_cached", "complete_shortlist")],
    ]
    lines = [
        "First full-success cost and work",
        "Costs use the summary's full-success budget and complete raw cell.",
        "N costs include the frozen production setup and applicable selector charge.",
        "Work totals cover all seed/task/repeat attempts at the first full-success cap.",
        "",
        ("engine,control,budget,mean_attempt_cpu_ns,N48_cpu_ns,"
         "N480_cpu_ns,N4800_cpu_ns,cheap_row_ops/root_calls/"
         "policy_features/dedup/replays_per_attempt"),
    ]
    for engine, control in configs:
        point = full_success.get(config_key(engine, control))
        if point is None or point["status"] != "observed":
            lines.append(f"{engine},{control},NO_FULL_SUCCESS")
            continue
        budget = point["budget"]
        rows = point_rows(raw, engine, control, budget)
        verify_cell(rows, expected_attempts)
        work_totals = point.get("work_totals_by_axis")
        if not isinstance(work_totals, dict):
            raise ValueError("full-success summary lacks work totals")
        raw_work = {
            key: sum(row["work"].get(axis, 0) for row in rows)
            for key, axis in WORK_AXES.items()
        }
        if raw_work != {key: work_totals.get(axis)
                        for key, axis in WORK_AXES.items()}:
            raise ValueError("summary work totals differ from raw rows")
        per_attempt = [
            round(raw_work[key] / expected_attempts)
            for key in WORK_AXES
        ]
        n_costs = point["training_amortized_cpu_ns_per_task_by_N"]
        mean_attempt = round(point["mean_attempt_cpu_ns"])
        lines.append(
            f"{engine},{control},{budget},{mean_attempt},"
            f"{round(n_costs['48'])},{round(n_costs['480'])},"
            f"{round(n_costs['4800'])}," + "/".join(map(str, per_attempt))
        )
    memory_expected = 48 * 10 * 2
    memory_keys = {
        (row.get("memory_mode"), row.get("task_id"), row.get("engine"),
         row.get("control"), row.get("budget"))
        for row in memory
    }
    memory_complete = (
        len(memory) == memory_expected
        and len(memory_keys) == memory_expected
        and all(row.get("budget") == 32768 for row in memory)
    )
    lines.extend([
        "",
        "Memory coverage: "
        + ("complete at cap 32768 only (RSS and tracemalloc; 960 rows)."
           if memory_complete else
           f"incomplete or mismatched ({len(memory)} rows; expected 960 at cap 32768)."),
        "Memory results do not establish a memory curve across root caps.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = render(args.run_dir)
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
