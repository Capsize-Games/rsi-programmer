"""Scoped corruption controls for the plastic-encoder auditor.

These controls use one runner-generated episode prefix and one retention
probe-group prefix. They establish that semantic corruption is rejected
before the expected incomplete-matrix EOF check; they do not claim a full
bundle audit. All temporary prefix copies live under the system temporary
directory. The frozen measurement bundle is read-only.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from rsi_programmer.online_benchmark import load_corpus, source_manifest
from rsi_programmer.online_counterbalanced import load_schedule
from rsi_programmer.online_plastic_benchmark import (
    _attempt_row,
    load_protocol,
    policy_for,
    probe_retention,
)
from scripts.audit_online_graph import verify_source_hashes
from scripts.audit_online_plastic_encoder import (
    PROTOCOL_SHA256,
    ReferencePlasticPolicy,
    _audit_retention,
    _audit_task_rows,
)

ROOT = Path(__file__).resolve().parents[2]


def _write_jsonl_gzip(path: Path, row: dict[str, Any]) -> None:
    """Replace one temporary prefix stream with one JSON record."""
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _expect_error(label: str, expected: str,
                  function: Callable[[], Any]) -> dict[str, str]:
    """Require a semantic error substring before the prefix EOF error."""
    try:
        function()
    except ValueError as error:
        message = str(error)
        if expected not in message:
            raise AssertionError(
                f"{label}: expected {expected!r}, got {message!r}") from error
        return {"control": label, "status": "rejected", "error": message}
    raise AssertionError(f"{label}: corrupted prefix was accepted")


def _episode_controls(directory: Path) -> list[dict[str, str]]:
    """Build one valid first episode, then audit independent corrupt copies."""
    protocol = load_protocol()
    corpus = load_corpus()
    schedule = load_schedule()
    tasks = corpus["tasks"]
    task_by_id = {task["id"]: task for task in tasks}
    orders = schedule["orders"]
    first_order = orders[0]
    first_task = task_by_id[first_order["task_ids"][0]]
    policy = policy_for("persistent_snn", 0)
    if policy is None:
        raise AssertionError("neural control unexpectedly has no policy")
    row, _offset = _attempt_row(
        first_task,
        order_id=first_order["order_id"],
        phase_sequence=list(first_order["phase_sequence"]),
        stream_index=0,
        arm="persistent_snn",
        seed=0,
        budget=144,
        policy=policy,
        draw_offset=0,
    )
    path = directory / "episodes.jsonl.gz"
    controls = []

    _write_jsonl_gzip(path, row)
    controls.append(_expect_error(
        "valid_episode_prefix", "incomplete episode matrix 1/13824",
        lambda: _audit_task_rows(directory, protocol, tasks, orders)))

    corrupt = copy.deepcopy(row)
    corrupt["attempts"][0]["predictions"][0] += 1
    _write_jsonl_gzip(path, corrupt)
    controls.append(_expect_error(
        "forged_observed_prediction", "observed predictions mismatch",
        lambda: _audit_task_rows(directory, protocol, tasks, orders)))

    corrupt = copy.deepcopy(row)
    corrupt["attempts"][0]["reward"] = (
        0.123 if corrupt["attempts"][0]["reward"] != 0.123 else 0.321)
    _write_jsonl_gzip(path, corrupt)
    controls.append(_expect_error(
        "forged_observed_reward", "attempt reward mismatch",
        lambda: _audit_task_rows(directory, protocol, tasks, orders)))

    corrupt = copy.deepcopy(row)
    corrupt["encoder_hash_before"] = "f" * 64
    _write_jsonl_gzip(path, corrupt)
    controls.append(_expect_error(
        "forged_encoder_hash", "pre-task encoder hash mismatch",
        lambda: _audit_task_rows(directory, protocol, tasks, orders)))

    corrupt = copy.deepcopy(row)
    corrupt["context_work"]["observation_scalars"] += 1
    _write_jsonl_gzip(path, corrupt)
    controls.append(_expect_error(
        "forged_context_work", "observed context setup counters mismatch",
        lambda: _audit_task_rows(directory, protocol, tasks, orders)))
    return controls


def _retention_controls(directory: Path) -> list[dict[str, str]]:
    """Build one valid 48-context group and corrupt copied prefix rows."""
    corpus = load_corpus()
    schedule = load_schedule()
    tasks = corpus["tasks"]
    first_order = schedule["orders"][0]
    runtime = policy_for("persistent_snn", 0)
    if runtime is None:
        raise AssertionError("neural control unexpectedly has no policy")
    row = probe_retention(
        runtime,
        tasks,
        order_id=first_order["order_id"],
        phase_sequence=list(first_order["phase_sequence"]),
        budget=144,
        seed=0,
        arm="persistent_snn",
        completed_tasks=0,
    )
    reference = ReferencePlasticPolicy("persistent_snn", 0)
    key = (first_order["order_id"], 144, 0, "persistent_snn", 0)
    boundary_states = {key: reference}
    path = directory / "retention.jsonl.gz"
    controls = []

    def audit(candidate: dict[str, Any]) -> None:
        _write_jsonl_gzip(path, candidate)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _audit_retention(
            directory, boundary_states, tasks,
            {"retention_raw_sha256": digest},
        )

    valid = _expect_error(
        "valid_retention_prefix", "retention matrix incomplete 1/1008",
        lambda: audit(row))
    valid["status"] = "replayed_prefix"
    controls.append(valid)

    corrupt = copy.deepcopy(row)
    corrupt["contexts"][0]["rates"][0] += 0.1
    controls.append(_expect_error(
        "forged_retention_rate", "retention context probe",
        lambda: audit(corrupt)))

    corrupt = copy.deepcopy(row)
    corrupt["contexts"][0]["probabilities"][0] += 0.1
    controls.append(_expect_error(
        "forged_retention_probability", "retention context probe",
        lambda: audit(corrupt)))

    corrupt = copy.deepcopy(row)
    corrupt["source_state_sha256_before"] = "0" * 64
    controls.append(_expect_error(
        "forged_retention_source_hash", "retention source before mismatch",
        lambda: audit(corrupt)))

    corrupt = copy.deepcopy(row)
    corrupt["explicit_probe_work"]["neuron_ticks"] -= 1
    controls.append(_expect_error(
        "forged_retention_work", "retention explicit work missing",
        lambda: audit(corrupt)))
    return controls


def run_controls() -> dict[str, Any]:
    """Run scoped semantic corruption controls without reading measurements."""
    protocol = load_protocol()
    if protocol["matrix"]["episode_rows"] != 13824:
        raise AssertionError("frozen protocol parser did not return")
    with tempfile.TemporaryDirectory(
            prefix="online-plastic-negative-controls-") as temporary:
        directory = Path(temporary)
        controls = _episode_controls(directory)
        controls.extend(_retention_controls(directory))

    manifest = source_manifest()
    manifest.pop("rsi_programmer/online_plastic_encoder.py")
    try:
        verify_source_hashes(manifest, ROOT)
    except ValueError as error:
        controls.append({
            "control": "missing_source_manifest_entry",
            "status": "rejected",
            "error": str(error),
        })
    else:
        raise AssertionError("source manifest omission was accepted")

    return {
        "status": "passed",
        "scope": "One valid episode row and one valid 48-probe retention "
                 "group; semantic corruption is checked before matrix EOF. "
                 "This is not a full-bundle audit.",
        "protocol_sha256": PROTOCOL_SHA256,
        "controls": controls,
    }


if __name__ == "__main__":
    print(json.dumps(run_controls(), sort_keys=True, indent=2))
