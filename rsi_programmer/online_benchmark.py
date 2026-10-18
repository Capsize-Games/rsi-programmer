"""Frozen, test-before-feedback pilot for online graph controllers."""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from rsi_programmer.online_graph import check_result, run_episode
from rsi_programmer.online_spiking import OnlineSpikingPolicy

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "fixtures" / "online_graph_v1.json"
CORPUS_SHA256 = (
    "a4f9af2dfc0d8c4539d700b032dcc223c82aa272f02497902024d9f371096140")
ARMS = ("persistent_snn", "frozen_snn", "episode_reset_snn",
        "online_nonspiking", "grammar_streaming")
SEEDS = (0, 1, 2)
BUDGETS = (144, 432)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_manifest() -> dict[str, str]:
    """Bind runtime, tests, scripts, inputs and workflow without artifacts."""
    paths: set[Path] = set()
    for directory in ("rsi_programmer", "tests", "scripts", "fixtures"):
        paths.update(path for path in (ROOT / directory).rglob("*")
                     if path.suffix in {".py", ".json", ".js"})
    paths.update(ROOT / name for name in (
        "requirements.txt", "justfile", ".github/workflows/ci.yml"))
    return {str(path.relative_to(ROOT)): _digest(path.read_bytes())
            for path in sorted(paths)}


def load_corpus(path: Path = CORPUS) -> dict[str, Any]:
    """Reject any changes to the preregistered corpus/protocol bytes."""
    payload = path.read_bytes()
    if _digest(payload) != CORPUS_SHA256:
        raise ValueError("frozen online corpus/protocol hash mismatch")
    corpus = json.loads(payload)
    protocol = corpus["protocol"]
    if (len(corpus["tasks"]) != 48
            or tuple(protocol["arms"]) != ARMS
            or tuple(protocol["model_seeds"]) != SEEDS
            or protocol["primary_budget"] != BUDGETS[0]
            or protocol["secondary_budget"] != BUDGETS[1]
            or protocol["policy"]["eligibility_decay"] != 0.0):
        raise ValueError("online protocol does not match runner")
    return corpus


def _policy(arm: str, seed: int) -> OnlineSpikingPolicy | None:
    if arm == "grammar_streaming":
        return None
    if arm not in ARMS:
        raise ValueError("unknown control")
    return OnlineSpikingPolicy(seed=seed, learning=arm != "frozen_snn",
                               spiking=arm != "online_nonspiking")


def summarize(rows: list[dict[str, Any]],
              setup: dict[str, float]) -> dict[str, Any]:
    """Retain failures and report seeds separately from the shared tasks."""
    groups: dict[tuple[int, str, int], list[dict[str, Any]]]
    groups = defaultdict(list)
    for row in rows:
        groups[row["budget"], row["arm"], row["seed"]].append(row)
    summaries = []
    for (budget, arm, seed), items in sorted(groups.items()):
        total_cpu = setup[f"{budget}:{arm}:{seed}"] + sum(
            item["episode_cpu_seconds"] + item["checker_cpu_seconds"]
            + item["reset_cpu_seconds"] for item in items)
        summaries.append({
            "budget": budget, "arm": arm, "seed": seed,
            "tasks": len(items),
            "accepted": sum(item["accepted"] for item in items),
            "observed_only_false_positives": sum(
                item["observed_solved"] and not item["accepted"]
                for item in items),
            "first_lane_correct": sum(item["first_lane_correct"]
                                      for item in items),
            "candidate_evaluations": sum(item["candidate_evaluations"]
                                         for item in items),
            "node_row_operations": sum(item["node_row_operations"]
                                       for item in items),
            "total_cpu_seconds": total_cpu,
            "policy_cpu_seconds": sum(item["policy_cpu_seconds"]
                                      for item in items),
            "update_cpu_seconds": sum(item["update_cpu_seconds"]
                                      for item in items),
            "phase_counts": [{
                "phase": phase,
                "accepted": sum(item["accepted"] for item in items
                                if item["phase"] == phase),
                "first_lane_correct": sum(item["first_lane_correct"]
                                          for item in items
                                          if item["phase"] == phase),
            } for phase in range(3)],
        })

    def aggregate(budget: int, arm: str, key: str) -> float:
        return sum(row[key] for row in summaries
                   if row["budget"] == budget and row["arm"] == arm)

    def first_better(arm: str, require_each_seed: bool = True) -> bool:
        ours = aggregate(144, "persistent_snn", "first_lane_correct")
        other = aggregate(144, arm, "first_lane_correct")
        per_seed = all(
            next(row["first_lane_correct"] for row in summaries
                 if (row["budget"], row["arm"], row["seed"])
                 == (144, "persistent_snn", seed))
            > next(row["first_lane_correct"] for row in summaries
                   if (row["budget"], row["arm"], row["seed"])
                   == (144, arm, seed)) for seed in SEEDS)
        return (ours - other >= 0.10 * 144
                and (per_seed or not require_each_seed))

    def coverage_at_least(budget: int, arm: str) -> bool:
        return (aggregate(budget, "persistent_snn", "accepted")
                >= aggregate(budget, arm, "accepted"))

    def cpu_at_most(budget: int, arm: str) -> bool:
        return (aggregate(budget, "persistent_snn", "total_cpu_seconds")
                <= aggregate(budget, arm, "total_cpu_seconds"))

    adaptation = all(first_better(arm) and coverage_at_least(144, arm)
                     for arm in ("frozen_snn", "episode_reset_snn"))
    specific = (adaptation and first_better("online_nonspiking", False)
                and coverage_at_least(144, "online_nonspiking")
                and cpu_at_most(144, "online_nonspiking"))
    deployment = all(coverage_at_least(budget, "grammar_streaming")
                     and cpu_at_most(budget, "grammar_streaming")
                     for budget in BUDGETS)
    return {"per_seed": summaries, "gates": {
        "adaptation": adaptation, "snn_specific": specific,
        "streaming_deployment": deployment},
        "scope": "48 shared synthetic tasks; three model seeds; pilot only"}


def run_benchmark(output: Path) -> dict[str, Any]:
    """Run the entire frozen matrix once; never overwrite an old result."""
    corpus = load_corpus()
    output.mkdir(parents=True, exist_ok=False)
    sources_before = source_manifest()
    run_started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    setup: dict[str, float] = {}
    with gzip.open(output / "episodes.jsonl.gz", "wt",
                   encoding="utf-8") as raw:
        for budget in BUDGETS:
            for seed in SEEDS:
                for arm in ARMS:
                    tick = time.process_time()
                    policy = _policy(arm, seed)
                    setup[f"{budget}:{arm}:{seed}"] = (
                        time.process_time() - tick)
                    for task in corpus["tasks"]:
                        tick = time.process_time()
                        if policy is not None and arm == "episode_reset_snn":
                            policy.reset_learning()
                        reset_cpu = time.process_time() - tick
                        before = policy.metrics() if policy else {}
                        observed = tuple(tuple(row) for row
                                         in task["observed"])
                        out = run_episode(policy, observed, budget=budget,
                                          episode_id=task["id"])
                        after = policy.metrics() if policy else {}
                        work = {key: after[key] - before[key]
                                for key in after
                                if type(after[key]) is int}
                        held = tuple(tuple(row) for row in task["held"])
                        poly = tuple(tuple(term) for term
                                     in task["polynomial"])
                        verdict = check_result(out["graph"], held, poly)
                        row = {"arm": arm, "seed": seed,
                               "budget": budget,
                               "task_id": task["id"], "phase": task["phase"],
                               "root": task["root"],
                               "first_lane_correct": (
                                   bool(out["attempts"])
                                   and out["attempts"][0]["action"]
                                   == task["root"]),
                               "policy_work": work,
                               "weight_before": before.get("weight_hash"),
                               "weight_after": after.get("weight_hash"),
                               "reset_cpu_seconds": reset_cpu,
                               "model_state_json_bytes": len(json.dumps(
                                   policy.snapshot())) if policy else 0,
                               **out, **verdict}
                        raw.write(json.dumps(row, sort_keys=True,
                                             allow_nan=False) + "\n")
                        all_rows.append({key: value for key, value
                                         in row.items() if key != "attempts"})
                    print(f"completed budget={budget} seed={seed} arm={arm}",
                          flush=True)
    sources_after = source_manifest()
    if sources_before != sources_after:
        raise RuntimeError("source changed during online benchmark")
    summary = summarize(all_rows, setup)
    metadata = {
        "corpus_sha256": CORPUS_SHA256,
        "source_files": sources_before,
        "source_unchanged": True,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "snntorch")},
        "episode_rows": len(all_rows),
        "setup_cpu_seconds": setup,
        "wall_seconds": time.perf_counter() - run_started,
        "whole_process_peak_rss_kib": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss,
        "rss_scope": "cumulative process maximum; no per-arm memory claim",
        "cpu_scope": "setup/reset + episode including traces + held checker; "
                     "excludes output compression and provenance hashing",
        "raw_sha256": _digest((output / "episodes.jsonl.gz").read_bytes()),
    }
    for name, data in (("summary.json", summary), ("manifest.json", metadata)):
        (output / name).write_text(json.dumps(data, indent=2,
                                             sort_keys=True) + "\n")
    return summary
