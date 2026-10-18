"""Compare frozen legacy control rows across observed-context bundles."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "fixtures" / "online_graph_v1.json"
SCHEDULE_PATH = ROOT / "fixtures" / "online_counterbalanced_v1.json"
CORPUS_SHA256 = (
    "a4f9af2dfc0d8c4539d700b032dcc223c82aa272f02497902024d9f371096140")
SCHEDULE_SHA256 = (
    "0ffbd7eadac93aa63b8df68d9ba250cd4da5a55a178e8339f9505d81638af3f1")
CONTROL_ARMS = (
    "persistent_snn", "frozen_snn", "episode_reset_snn",
    "online_nonspiking", "grammar_streaming",
)
BUDGETS = (144, 432)
SEEDS = (0, 1, 2)
EXPECTED_ROWS = 8640
EXPECTED_NEW_ROWS = 13824
ATTEMPT_FIELDS = (
    "action", "features", "graph", "predictions", "error", "reward",
    "observed_exact",
)
ROW_FIELDS = (
    "graph", "observed_solved", "accepted", "held_correct",
    "polynomial_correct", "held_rows_executed", "candidate_evaluations",
    "generated_candidates", "interpreter_rows", "node_row_operations",
    "graph_validation_calls", "decision_calls", "feedback_calls",
    "budget_exhausted", "grammar_exhausted", "first_lane_correct",
    "rng_draw_offset_before", "rng_draw_offset_after",
)
IDENTITY_FIELDS = (
    "order_id", "phase_sequence", "stream_index", "phase_position",
    "phase_ordinal", "arm", "seed", "budget", "task_id", "phase",
    "root",
)


def _sha256_file(path: Path) -> str:
    """Hash a file incrementally without retaining its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object and reject non-object documents."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _expected_rows() -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield the frozen ordered identity and task for each legacy control."""
    corpus_bytes = CORPUS_PATH.read_bytes()
    schedule_bytes = SCHEDULE_PATH.read_bytes()
    if hashlib.sha256(corpus_bytes).hexdigest() != CORPUS_SHA256:
        raise ValueError("frozen graph corpus digest mismatch")
    if hashlib.sha256(schedule_bytes).hexdigest() != SCHEDULE_SHA256:
        raise ValueError("frozen counterbalanced schedule digest mismatch")
    corpus = json.loads(corpus_bytes)
    schedule = json.loads(schedule_bytes)["protocol"]
    tasks = {task["id"]: task for task in corpus["tasks"]}
    orders = schedule["orders"]
    if (len(tasks) != 48 or len(orders) != 6
            or tuple(schedule["arms"]) != CONTROL_ARMS
            or tuple(schedule["budgets"]) != BUDGETS
            or tuple(schedule["model_seeds"]) != SEEDS):
        raise ValueError("frozen comparison schedule dimensions changed")
    for order in orders:
        phase_sequence = order["phase_sequence"]
        task_ids = [task["id"] for phase in phase_sequence
                    for task in corpus["tasks"] if task["phase"] == phase]
        if order["task_ids"] != task_ids:
            raise ValueError("frozen task order is inconsistent with corpus")
        for budget in BUDGETS:
            for seed in SEEDS:
                for arm in CONTROL_ARMS:
                    for stream_index, task_id in enumerate(task_ids):
                        task = tasks[task_id]
                        expected = {
                            "order_id": order["order_id"],
                            "phase_sequence": phase_sequence,
                            "stream_index": stream_index,
                            "phase_position": stream_index // 16,
                            "phase_ordinal": stream_index % 16,
                            "arm": arm,
                            "seed": seed,
                            "budget": budget,
                            "task_id": task_id,
                            "phase": task["phase"],
                            "root": task["root"],
                        }
                        yield expected, task


def _iter_rows(path: Path, *, controls_only: bool,
               counts: dict[str, int]) -> Iterator[dict[str, Any]]:
    """Stream JSONL rows, optionally selecting only the frozen controls."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            counts["total"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON row {line_number} in {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object row {line_number} in {path}")
            if not controls_only or row.get("arm") in CONTROL_ARMS:
                counts["selected"] += 1
                yield row


def _compare_attempts(
    old: dict[str, Any], new: dict[str, Any], row_number: int
) -> None:
    """Compare all frozen per-attempt action and execution outputs."""
    old_attempts = old.get("attempts")
    new_attempts = new.get("attempts")
    if (not isinstance(old_attempts, list)
            or not isinstance(new_attempts, list)):
        raise ValueError(f"attempts must be arrays at row {row_number}")
    if len(old_attempts) != len(new_attempts):
        raise ValueError(f"attempt count differs at row {row_number}")
    for attempt_index, (old_attempt, new_attempt) in enumerate(
            zip(old_attempts, new_attempts)):
        if (not isinstance(old_attempt, dict)
                or not isinstance(new_attempt, dict)):
            raise ValueError(
                f"invalid attempt object at row {row_number}, "
                f"attempt {attempt_index}")
        for field in ATTEMPT_FIELDS:
            if field not in old_attempt or field not in new_attempt:
                raise ValueError(
                    f"missing attempt {field} at row {row_number}, "
                    f"attempt {attempt_index}")
            if old_attempt[field] != new_attempt[field]:
                raise ValueError(
                    f"attempt {field} differs at row {row_number}, "
                    f"attempt {attempt_index}")


def _compare_row(
    old: dict[str, Any],
    new: dict[str, Any],
    expected: dict[str, Any],
    row_number: int,
) -> None:
    """Compare matched control identity, readout state and outcomes."""
    for field in IDENTITY_FIELDS:
        if field not in old or old[field] != expected[field]:
            raise ValueError(f"old {field} violates frozen schedule at row "
                             f"{row_number}")
        if field not in new or new[field] != expected[field]:
            raise ValueError(f"new {field} violates frozen schedule at row "
                             f"{row_number}")
    for field in ROW_FIELDS:
        if field not in old or field not in new:
            raise ValueError(f"missing {field} at row {row_number}")
        if old[field] != new[field]:
            raise ValueError(f"{field} differs at row {row_number}")
    for old_field, new_field in (
            ("weight_before", "readout_hash_before"),
            ("weight_after", "readout_hash_after")):
        if old_field not in old or new_field not in new:
            raise ValueError(
                f"missing readout hash at row {row_number}")
        if old[old_field] != new[new_field]:
            raise ValueError(f"{old_field}/{new_field} differs at row "
                             f"{row_number}")
    _compare_attempts(old, new, row_number)


def compare_row_streams(
    old_rows: Iterable[dict[str, Any]],
    new_rows: Iterable[dict[str, Any]],
    *,
    expected_count: int = EXPECTED_ROWS,
) -> int:
    """Compare matched row iterables against the frozen identity schedule."""
    expected_iter = _expected_rows()
    sentinel = object()
    old_iter = iter(old_rows)
    new_iter = iter(new_rows)
    compared = 0
    for row_number in range(1, expected_count + 1):
        old = next(old_iter, sentinel)
        new = next(new_iter, sentinel)
        if old is sentinel or new is sentinel:
            raise ValueError(f"missing control row {row_number}")
        expected, _task = next(expected_iter)
        _compare_row(old, new, expected, row_number)
        compared += 1
    if next(old_iter, sentinel) is not sentinel:
        raise ValueError("extra old control row")
    if next(new_iter, sentinel) is not sentinel:
        raise ValueError("extra new control row")
    if expected_count == EXPECTED_ROWS:
        try:
            next(expected_iter)
        except StopIteration:
            pass
        else:
            raise ValueError(
                "frozen schedule has more rows than expected_count")
    return compared


def compare_bundles(old_bundle: Path, new_bundle: Path) -> dict[str, Any]:
    """Verify complete bundle provenance and compare all legacy controls."""
    old_bundle = old_bundle.resolve()
    new_bundle = new_bundle.resolve()
    old_manifest = _load_json(old_bundle / "manifest.json")
    new_manifest = _load_json(new_bundle / "manifest.json")
    old_rows_path = old_bundle / "episodes.jsonl.gz"
    new_rows_path = new_bundle / "episodes.jsonl.gz"
    old_raw_sha = _sha256_file(old_rows_path)
    new_raw_sha = _sha256_file(new_rows_path)
    if old_raw_sha != old_manifest.get("raw_sha256"):
        raise ValueError("old bundle raw checksum mismatch")
    new_pinned_sha = new_manifest.get(
        "episodes_raw_sha256", new_manifest.get("raw_sha256"))
    if new_raw_sha != new_pinned_sha:
        raise ValueError("new bundle raw checksum mismatch")
    if old_manifest.get("episode_rows") != EXPECTED_ROWS:
        raise ValueError("old bundle declared row count is not 8640")
    if new_manifest.get("episode_rows") != EXPECTED_NEW_ROWS:
        raise ValueError("new bundle declared row count is not 13824")
    if (old_manifest.get("source_unchanged") is not True
            or new_manifest.get("source_unchanged") is not True):
        raise ValueError("a bundle did not attest unchanged sources")

    old_counts = {"total": 0, "selected": 0}
    new_counts = {"total": 0, "selected": 0}
    old_rows = _iter_rows(old_rows_path, controls_only=False,
                          counts=old_counts)
    new_rows = _iter_rows(new_rows_path, controls_only=True,
                          counts=new_counts)
    matched = compare_row_streams(old_rows, new_rows)
    if old_counts["total"] != EXPECTED_ROWS:
        raise ValueError("old raw row count mismatch")
    if (new_counts["total"] != EXPECTED_NEW_ROWS
            or new_counts["selected"] != EXPECTED_ROWS):
        raise ValueError("new raw/control row count mismatch")
    return {
        "schema_version": 1,
        "verification_status": "exact_control_equivalence",
        "matched_control_rows": matched,
        "control_arms": list(CONTROL_ARMS),
        "input_bundles": {
            "old": {
                "path": str(old_bundle),
                "manifest_sha256": _sha256_file(
                    old_bundle / "manifest.json"),
                "raw_sha256": old_raw_sha,
                "declared_episode_rows": old_manifest["episode_rows"],
                "source_head": old_manifest.get("git_head"),
            },
            "new": {
                "path": str(new_bundle),
                "manifest_sha256": _sha256_file(
                    new_bundle / "manifest.json"),
                "raw_sha256": new_raw_sha,
                "declared_episode_rows": new_manifest["episode_rows"],
                "source_head": new_manifest.get("git_head"),
            },
        },
        "compared_fields": {
            "identity": list(IDENTITY_FIELDS),
            "legacy_readout_hash_mapping": {
                "weight_before": "readout_hash_before",
                "weight_after": "readout_hash_after",
            },
            "row_outcomes": list(ROW_FIELDS),
            "attempts": list(ATTEMPT_FIELDS),
        },
        "excluded_fields": [
            "CPU and wall timings", "new plastic-encoder hashes",
            "additional plasticity and context work counters",
            "new-only plastic controls and retention rows",
        ],
        "frozen_fixture_sha256": {
            "corpus": CORPUS_SHA256,
            "schedule": SCHEDULE_SHA256,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the comparator and emit a JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-bundle", type=Path, required=True)
    parser.add_argument("--new-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    receipt = compare_bundles(arguments.old_bundle, arguments.new_bundle)
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        sys.stdout.write(encoded)
    else:
        arguments.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
