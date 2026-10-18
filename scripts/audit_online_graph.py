"""Independently audit a frozen online graph-search result bundle."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from rsi_programmer.online_graph import (
    GRAMMAR_SIZE,
    LANE_SIZE,
    OPERATORS,
    iter_candidates,
)
from rsi_programmer.online_spiking import OnlineSpikingPolicy
from rsi_programmer.probe import interpret

ROOT = Path(__file__).resolve().parents[1]
CORPUS_SHA256 = (
    "a4f9af2dfc0d8c4539d700b032dcc223c82aa272f02497902024d9f371096140"
)
ARMS = ("persistent_snn", "frozen_snn", "episode_reset_snn",
        "online_nonspiking", "grammar_streaming")
SEEDS = (0, 1, 2)
BUDGETS = (144, 432)
TOLERANCE = 1e-12


def _sha256(payload: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def _close(actual: Any, expected: float, label: str) -> None:
    """Reject nonnumeric, nonfinite, or mismatched measured values."""
    if (isinstance(actual, bool) or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(float(actual), expected,
                                rel_tol=TOLERANCE, abs_tol=TOLERANCE)):
        raise ValueError(f"{label} mismatch")


def _polynomial(graph: dict[str, Any]) -> tuple[tuple[int, int, int], ...]:
    """Expand arithmetic graph nodes independently of the engine helper."""
    values: dict[str | int, dict[tuple[int, int], int]] = {
        "x": {(1, 0): 1}, "y": {(0, 1): 1}}
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("graph nodes missing")
    for index, node in enumerate(nodes):
        if node.get("op") == "const":
            values[index] = {(0, 0): node["value"]}
            continue
        if node.get("op") not in OPERATORS:
            raise ValueError("unsupported graph operation")
        left, right = (values[arg] for arg in node["args"])
        result: dict[tuple[int, int], int] = {}
        if node["op"] == "mul":
            for (xp, yp), value in left.items():
                for (xq, yq), other in right.items():
                    key = xp + xq, yp + yq
                    result[key] = result.get(key, 0) + value * other
        else:
            result.update(left)
            sign = 1 if node["op"] == "add" else -1
            for key, value in right.items():
                result[key] = result.get(key, 0) + sign * value
        values[index] = {key: value for key, value in result.items()
                         if value}
    output = values[graph["output"]]
    return tuple((xp, yp, value)
                 for (xp, yp), value in sorted(output.items()) if value)


def _policy(arm: str, seed: int) -> OnlineSpikingPolicy | None:
    """Construct the protocol-matched policy for deterministic replay."""
    if arm == "grammar_streaming":
        return None
    return OnlineSpikingPolicy(seed=seed, learning=arm != "frozen_snn",
                               spiking=arm != "online_nonspiking")


def verify_source_hashes(source_files: dict[str, str],
                         repo_root: Path) -> None:
    """Check the complete expected source set and its pinned digests."""
    suffixes = {".py", ".json", ".js"}
    expected_paths: set[str] = set()
    for directory in ("rsi_programmer", "tests", "scripts", "fixtures"):
        expected_paths.update(
            str(path.relative_to(repo_root))
            for path in (repo_root / directory).rglob("*")
            if path.is_file() and path.suffix in suffixes)
    expected_paths.update(name for name in (
        "requirements.txt", "justfile", ".github/workflows/ci.yml")
        if (repo_root / name).is_file())
    if set(source_files) != expected_paths:
        missing = sorted(expected_paths - set(source_files))
        extra = sorted(set(source_files) - expected_paths)
        raise ValueError(
            f"source manifest path set mismatch; missing={missing}, "
            f"extra={extra}")
    for name, expected in source_files.items():
        source = (repo_root / name).resolve()
        if repo_root.resolve() not in source.parents or not source.is_file():
            raise ValueError("invalid or missing pinned source path")
        if _sha256(source.read_bytes()) != expected:
            raise ValueError(f"source hash mismatch: {name}")


def _gates(rows: list[dict[str, Any]]) -> dict[str, bool]:
    """Recompute the frozen aggregate and per-seed acceptance gates."""
    def aggregate(budget: int, arm: str, key: str) -> float:
        return sum(float(row[key]) for row in rows
                   if row["budget"] == budget and row["arm"] == arm)

    def first_better(arm: str, per_seed: bool = True) -> bool:
        ours = aggregate(144, "persistent_snn", "first_lane_correct")
        other = aggregate(144, arm, "first_lane_correct")
        better_each = all(
            next(row["first_lane_correct"] for row in rows
                 if (row["budget"], row["arm"], row["seed"])
                 == (144, "persistent_snn", seed))
            > next(row["first_lane_correct"] for row in rows
                   if (row["budget"], row["arm"], row["seed"])
                   == (144, arm, seed)) for seed in SEEDS)
        return ours - other >= 0.10 * 144 and (better_each or not per_seed)

    def accepts_at_least(budget: int, arm: str) -> bool:
        return (aggregate(budget, "persistent_snn", "accepted")
                >= aggregate(budget, arm, "accepted"))

    def cpu_at_most(budget: int, arm: str) -> bool:
        return (aggregate(budget, "persistent_snn", "total_cpu_seconds")
                <= aggregate(budget, arm, "total_cpu_seconds"))

    adaptation = all(
        first_better(arm) and accepts_at_least(144, arm)
        for arm in ("frozen_snn", "episode_reset_snn"))
    specific = (adaptation and first_better("online_nonspiking", False)
                and accepts_at_least(144, "online_nonspiking")
                and cpu_at_most(144, "online_nonspiking"))
    streaming = all(
        accepts_at_least(budget, "grammar_streaming")
        and cpu_at_most(budget, "grammar_streaming")
        for budget in BUDGETS)
    return {"adaptation": adaptation, "snn_specific": specific,
            "streaming_deployment": streaming}


def _phase_counts(rows: list[dict[str, Any]]) -> list[dict[str, int]]:
    """Recompute acceptance and first-choice counts for each corpus phase."""
    return [{
        "phase": phase,
        "accepted": sum(bool(row["accepted"]) for row in rows
                         if row["phase"] == phase),
        "first_lane_correct": sum(bool(row["first_lane_correct"])
                                   for row in rows
                                   if row["phase"] == phase),
    } for phase in range(3)]


def audit_episode_row(row: dict[str, Any], task: dict[str, Any],
                      budget: int,
                      policy: OnlineSpikingPolicy | None) -> None:
    """Replay one task, including controller choices and all check outcomes."""
    if (row.get("task_id") != task["id"] or row.get("phase") != task["phase"]
            or row.get("root") != task["root"] or row.get("budget") != budget):
        raise ValueError("episode identity mismatch")
    _validate_cpu_fields(row)
    observed = tuple(tuple(item) for item in task["observed"])
    held = tuple(tuple(item) for item in task["held"])
    expected_poly = tuple(tuple(term) for term in task["polynomial"])
    attempts = row.get("attempts")
    if not isinstance(attempts, list) or not 0 < len(attempts) <= budget:
        raise ValueError("invalid attempt list or budget use")
    lane_candidates = {operator: iter_candidates(operator)
                       for operator in OPERATORS}
    cursors = {operator: 0 for operator in OPERATORS}
    previous_action: str | None = None
    previous_error = 1.0
    exact_graph: dict[str, Any] | None = None
    before_metrics = policy.metrics() if policy else {}
    if policy:
        policy.start_episode(task["id"])
    row_operations = 0
    for index, attempt in enumerate(attempts):
        available = tuple(op for op in OPERATORS
                          if cursors[op] < LANE_SIZE)
        if not available:
            raise ValueError("attempt after all grammar lanes exhausted")
        features = (1.0, *(float(previous_action == op)
                           for op in OPERATORS), previous_error,
                    index / budget)
        if attempt.get("features") != list(features):
            raise ValueError("policy feature history mismatch")
        if policy:
            action = policy.choose(features, available)
            if action != attempt.get("action"):
                raise ValueError("controller choice does not replay")
        else:
            action = next(op for offset in range(3)
                          if (op := OPERATORS[(index + offset) % 3])
                          in available)
            if action != attempt.get("action"):
                raise ValueError("streaming lane order mismatch")
        if action not in available:
            raise ValueError("attempt used unavailable lane")
        candidate = next(lane_candidates[action])
        if candidate != attempt.get("graph"):
            raise ValueError("candidate is not next in grammar lane")
        cursors[action] += 1
        predictions = tuple(interpret(candidate, x, y)
                            for x, y, _want in observed)
        if list(predictions) != attempt.get("predictions"):
            raise ValueError("observed predictions mismatch")
        error = sum(min(1.0, abs(got - row_data[2])
                        / (1 + abs(row_data[2])))
                    for got, row_data in zip(predictions, observed))
        error /= len(observed)
        _close(attempt.get("error"), error, "attempt error")
        reward = 1.0 - error
        _close(attempt.get("reward"), reward, "attempt reward")
        exact = all(got == row_data[2]
                    for got, row_data in zip(predictions, observed))
        if attempt.get("observed_exact") is not exact:
            raise ValueError("observed exact flag mismatch")
        if policy:
            policy.observe(reward)
        row_operations += len(observed) * 3
        previous_action, previous_error = action, error
        if exact:
            if index != len(attempts) - 1:
                raise ValueError(
                    "attempts continued after exact observed match")
            exact_graph = candidate
    if policy:
        policy.end_episode()
    if row.get("graph") != exact_graph:
        raise ValueError("returned observed solution mismatch")
    if row.get("observed_solved") is not (exact_graph is not None):
        raise ValueError("observed solve flag mismatch")
    count = len(attempts)
    expected_counts = {
        "candidate_evaluations": count, "generated_candidates": count,
        "interpreter_rows": count * len(observed),
        "node_row_operations": row_operations,
        "graph_validation_calls": count * len(observed),
        "decision_calls": count if policy else 0,
        "feedback_calls": count if policy else 0,
        "budget_exhausted": exact_graph is None and count == budget,
        "grammar_exhausted": exact_graph is None and count == GRAMMAR_SIZE,
    }
    for key, expected in expected_counts.items():
        if row.get(key) != expected:
            raise ValueError(f"episode {key} mismatch")
    outputs = tuple(interpret(exact_graph, x, y)
                    for x, y, _want in held) if exact_graph else ()
    held_correct = bool(exact_graph) and outputs == tuple(
        want for _x, _y, want in held)
    poly_correct = (bool(exact_graph)
                    and _polynomial(exact_graph) == expected_poly)
    verdict = {
        "accepted": held_correct and poly_correct,
        "held_correct": held_correct,
        "polynomial_correct": poly_correct,
        "held_rows_executed": len(held) if exact_graph else 0,
    }
    for key, expected in verdict.items():
        mismatch = (row.get(key) is not expected
                    if isinstance(expected, bool)
                    else row.get(key) != expected)
        if mismatch:
            raise ValueError(f"independent checker {key} mismatch")
    first_op = attempts[0]["action"]
    if row.get("first_lane_correct") is not (first_op == task["root"]):
        raise ValueError("first lane correctness mismatch")
    if policy:
        after_metrics = policy.metrics()
        work = {key: after_metrics[key] - before_metrics[key]
                for key in after_metrics if type(after_metrics[key]) is int}
        if row.get("policy_work") != work:
            raise ValueError("policy work counters mismatch")
        if row.get("weight_after") != after_metrics["weight_hash"]:
            raise ValueError("post-episode weight hash mismatch")


def _validate_cpu_fields(row: dict[str, Any]) -> None:
    """Validate timing measurements and their nested CPU accounting."""
    names = ("episode_cpu_seconds", "episode_wall_seconds",
             "checker_cpu_seconds", "reset_cpu_seconds",
             "policy_cpu_seconds", "update_cpu_seconds")
    for name in names:
        value = row.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or value < 0):
            raise ValueError(f"invalid timing field {name}")
    nested = row["policy_cpu_seconds"] + row["update_cpu_seconds"]
    if nested > row["episode_cpu_seconds"] + 1e-9:
        raise ValueError("policy/update CPU exceeds episode CPU")


def audit_bundle(bundle: Path, repo_root: Path = ROOT) -> dict[str, Any]:
    """Validate provenance and replay every preregistered matrix row."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    corpus_path = repo_root / "fixtures" / "online_graph_v1.json"
    corpus_bytes = corpus_path.read_bytes()
    if _sha256(corpus_bytes) != CORPUS_SHA256:
        raise ValueError("frozen corpus SHA mismatch")
    if manifest.get("corpus_sha256") != CORPUS_SHA256:
        raise ValueError("manifest corpus SHA mismatch")
    if manifest.get("source_unchanged") is not True:
        raise ValueError("runner did not attest stable source")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("source hash manifest missing")
    verify_source_hashes(source_files, repo_root)
    raw_path = bundle / "episodes.jsonl.gz"
    raw_bytes = raw_path.read_bytes()
    if _sha256(raw_bytes) != manifest.get("raw_sha256"):
        raise ValueError("compressed episode stream SHA mismatch")
    corpus = json.loads(corpus_bytes)
    tasks = corpus["tasks"]
    if len(tasks) != 48 or len({task["id"] for task in tasks}) != 48:
        raise ValueError("frozen corpus task set invalid")
    expected_sequence = [(budget, arm, seed, task["id"])
                         for budget in BUDGETS for seed in SEEDS
                         for arm in ARMS for task in tasks]
    expected_rows = set(expected_sequence)
    if manifest.get("episode_rows") != len(expected_sequence):
        raise ValueError("manifest episode row count mismatch")
    seen: set[tuple[int, str, int, str]] = set()
    policies: dict[tuple[int, str, int], OnlineSpikingPolicy | None] = {}
    prior_hashes: dict[tuple[int, str, int], str] = {}
    compact_rows: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    task_by_id = {task["id"]: task for task in tasks}
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            key = (row.get("budget"), row.get("arm"), row.get("seed"),
                   row.get("task_id"))
            if (key not in expected_rows or key in seen
                    or len(seen) >= len(expected_sequence)
                    or key != expected_sequence[len(seen)]):
                raise ValueError(f"unexpected or duplicate row {line_number}")
            seen.add(key)
            budget, arm, seed, task_id = key
            group = budget, arm, seed
            policy = policies.get(group)
            if group not in policies:
                policy = _policy(arm, seed)
                policies[group] = policy
            if policy and arm == "episode_reset_snn":
                policy.reset_learning()
            before_hash = policy.weight_hash() if policy else None
            if row.get("weight_before") != before_hash:
                raise ValueError("pre-episode weight hash mismatch")
            if policy and arm == "persistent_snn" and group in prior_hashes:
                if before_hash != prior_hashes[group]:
                    raise ValueError("persistent weight chain mismatch")
            audit_episode_row(row, task_by_id[task_id], budget, policy)
            compact_rows.setdefault(group, []).append({
                "accepted": row.get("accepted"),
                "observed_solved": row.get("observed_solved"),
                "first_lane_correct": row.get("first_lane_correct"),
                "candidate_evaluations": row.get("candidate_evaluations"),
                "node_row_operations": row.get("node_row_operations"),
                "phase": row.get("phase"),
                "episode_cpu_seconds": row.get("episode_cpu_seconds"),
                "checker_cpu_seconds": row.get("checker_cpu_seconds"),
                "reset_cpu_seconds": row.get("reset_cpu_seconds"),
                "policy_cpu_seconds": row.get("policy_cpu_seconds"),
                "update_cpu_seconds": row.get("update_cpu_seconds"),
            })
            if policy:
                after_hash = policy.weight_hash()
                if arm == "frozen_snn" and after_hash != before_hash:
                    raise ValueError("frozen control changed readout")
                if arm == "episode_reset_snn":
                    initial = OnlineSpikingPolicy(seed=seed).weight_hash()
                    if before_hash != initial:
                        raise ValueError("reset control did not reset weights")
                prior_hashes[group] = after_hash
    if seen != expected_rows:
        raise ValueError(
            f"incomplete matrix: {len(seen)}/{len(expected_rows)}")
    summary = json.loads((bundle / "summary.json").read_text())
    summary_rows = summary.get("per_seed")
    if not isinstance(summary_rows, list) or len(summary_rows) != 30:
        raise ValueError("per-seed summary matrix invalid")
    expected_groups = {(budget, arm, seed) for budget in BUDGETS
                       for arm in ARMS for seed in SEEDS}
    actual_groups = {(row.get("budget"), row.get("arm"), row.get("seed"))
                     for row in summary_rows}
    if actual_groups != expected_groups:
        raise ValueError("per-seed summary keys are invalid")
    setup = manifest.get("setup_cpu_seconds")
    if not isinstance(setup, dict):
        raise ValueError("setup CPU accounting missing")
    for item in summary_rows:
        group = item["budget"], item["arm"], item["seed"]
        group_rows = compact_rows.get(group, [])
        budget, arm, seed = group
        setup_key = f"{budget}:{arm}:{seed}"
        if setup_key not in setup:
            raise ValueError("setup CPU entry missing")
        if (isinstance(setup[setup_key], bool)
                or not isinstance(setup[setup_key], (int, float))
                or not math.isfinite(setup[setup_key])
                or setup[setup_key] < 0):
            raise ValueError("invalid setup CPU value")
        total_cpu = setup[setup_key] + sum(
            row["episode_cpu_seconds"] + row["checker_cpu_seconds"]
            + row["reset_cpu_seconds"] for row in group_rows)
        checks = {
            "tasks": len(group_rows),
            "accepted": sum(bool(row["accepted"]) for row in group_rows),
            "first_lane_correct": sum(bool(row["first_lane_correct"])
                                       for row in group_rows),
            "candidate_evaluations": sum(row["candidate_evaluations"]
                                         for row in group_rows),
            "node_row_operations": sum(row["node_row_operations"]
                                       for row in group_rows),
        }
        checks["observed_only_false_positives"] = sum(
            1 for row in group_rows
            if row["observed_solved"] and not row["accepted"])
        for key, value in checks.items():
            if item.get(key) != value:
                raise ValueError(f"summary {key} mismatch")
        phase_counts = item.get("phase_counts")
        expected_phases = _phase_counts(group_rows)
        if phase_counts != expected_phases:
            raise ValueError("summary phase counts mismatch")
        _close(item.get("total_cpu_seconds"), total_cpu,
               "summary total CPU")
        _close(item.get("policy_cpu_seconds"), sum(
            row["policy_cpu_seconds"] for row in group_rows),
            "summary policy CPU")
        _close(item.get("update_cpu_seconds"), sum(
            row["update_cpu_seconds"] for row in group_rows),
            "summary update CPU")
    expected_gates = _gates(summary_rows)
    if summary.get("gates") != expected_gates:
        raise ValueError("summary acceptance gates mismatch")
    return {"status": "passed", "rows": len(seen),
            "tasks_per_cell": len(tasks), "corpus_sha256": CORPUS_SHA256,
            "raw_sha256": _sha256(raw_bytes)}


def main() -> None:
    """Run the independent audit for a completed result bundle."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_bundle(args.bundle), sort_keys=True))


if __name__ == "__main__":
    main()
