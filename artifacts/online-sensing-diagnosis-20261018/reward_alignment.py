"""Audit reward alignment across the frozen graph-search grammar."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CORPUS_SHA256 = (
    "a4f9af2dfc0d8c4539d700b032dcc223c82aa272f02497902024d9f371096140")
EXPECTED_GRAPH_SHA256 = (
    "399a4512a7749b05875adf15f4fc795c03f6d87f1aee1b53f4f8b7a9d1b8a6be")
EXPECTED_GENERATOR_SHA256 = (
    "5c4ae63c98868d5a36e0c303e4b363ae95373e376787e9586328fb0953d64421")
OPERATORS = ("add", "sub", "mul")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_inputs() -> tuple[dict[str, Any], str, str, str]:
    """Load verified corpus and source pins from this checkout."""
    sys.path.insert(0, str(ROOT))
    from rsi_programmer.online_benchmark import load_corpus

    corpus_path = ROOT / "fixtures" / "online_graph_v1.json"
    graph_path = ROOT / "rsi_programmer" / "online_graph.py"
    generator_path = ROOT / "rsi_programmer" / "online_corpus.py"
    corpus_hash = sha256_file(corpus_path)
    graph_hash = sha256_file(graph_path)
    generator_hash = sha256_file(generator_path)
    if (corpus_hash != EXPECTED_CORPUS_SHA256
            or graph_hash != EXPECTED_GRAPH_SHA256
            or generator_hash != EXPECTED_GENERATOR_SHA256):
        raise ValueError("frozen corpus or source hash changed")
    corpus = load_corpus()
    return corpus, corpus_hash, graph_hash, generator_hash


def audit() -> dict[str, Any]:
    """Compare uniform full-lane rewards with exact solvability."""
    sys.path.insert(0, str(ROOT))
    from rsi_programmer.online_graph import graph_polynomial
    from rsi_programmer.online_graph import iter_candidates
    from rsi_programmer.probe import interpret

    corpus, corpus_hash, graph_hash, generator_hash = load_frozen_inputs()
    lanes = {root: list(iter_candidates(root)) for root in OPERATORS}
    lane_sizes = {root: len(candidates)
                  for root, candidates in lanes.items()}
    if lane_sizes != {root: 144 for root in OPERATORS}:
        raise ValueError("frozen grammar lane sizes changed")
    results: list[dict[str, Any]] = []
    alternative_beats: dict[str, int] = {root: 0 for root in OPERATORS}
    winning_tasks = 0

    for task in corpus["tasks"]:
        rows = tuple(tuple(row) for row in task["observed"])
        target = tuple(tuple(term) for term in task["polynomial"])
        means: dict[str, float] = {}
        solve_positions: dict[str, int | None] = {}
        polynomial_match_lanes: list[str] = []
        for root, candidates in lanes.items():
            rewards: list[float] = []
            observed_solve: int | None = None
            for position, graph in enumerate(candidates, 1):
                if graph_polynomial(graph) == target:
                    polynomial_match_lanes.append(root)
                predictions = [interpret(graph, x, y)
                               for x, y, _want in rows]
                error = sum(
                    min(1.0, abs(got - want) / (1 + abs(want)))
                    for got, (_x, _y, want) in zip(predictions, rows)
                ) / len(rows)
                rewards.append(1.0 - error)
                if (observed_solve is None
                        and all(got == row[2]
                                for got, row in zip(predictions, rows))):
                    observed_solve = position
            means[root] = sum(rewards) / len(rewards)
            solve_positions[root] = observed_solve

        true_root = str(task["root"])
        if set(polynomial_match_lanes) != {true_root}:
            raise ValueError(
                f"{task['id']}: target lane matches "
                f"{sorted(set(polynomial_match_lanes))}; "
                f"corpus root is {true_root}")
        if solve_positions[true_root] is None:
            raise ValueError("target missing from its observed solvable lane")
        if any(solve_positions[root] is not None
               for root in OPERATORS if root != true_root):
            raise ValueError("observed rows solve a non-target lane")

        beaten = [root for root in OPERATORS if root != true_root
                  and means[root] > means[true_root]]
        winning_tasks += int(bool(beaten))
        for root in beaten:
            alternative_beats[root] += 1
        results.append({
            "task_id": task["id"],
            "phase": task["phase"],
            "solvable_root": true_root,
            "mean_reward_by_root": means,
            "first_observed_exact_position_by_root": solve_positions,
            "non_solvable_lanes_beating_solvable_mean": beaten,
        })

    return {
        "schema_version": 1,
        "corpus_sha256": corpus_hash,
        "online_graph_source_sha256": graph_hash,
        "online_corpus_source_sha256": generator_hash,
        "task_count": len(results),
        "lane_size": lane_sizes,
        "reward_formula": (
            "1 - mean(min(1, abs(prediction-target)/(1+abs(target))))"),
        "full_lane_uniform_mean_reward": True,
        "tasks_with_non_solvable_lane_higher_mean": winning_tasks,
        "alternative_lane_win_counts_over_all_tasks": alternative_beats,
        "interpretation_limit": (
            "Uniform full-lane means are not actual sequential exposures "
            "and do not establish a causal effect. Candidate order is "
            "deterministic; actual exposure stops at the first exact "
            "observed match or at the episode budget."),
        "task_results": results,
    }


def main() -> None:
    """Write the audit result to the requested path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)
                           + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "tasks": result["task_count"],
        "tasks_with_non_solvable_lane_higher_mean": result[
            "tasks_with_non_solvable_lane_higher_mean"],
        "alternative_lane_win_counts_over_all_tasks": result[
            "alternative_lane_win_counts_over_all_tasks"],
        "hashes": {
            "corpus": result["corpus_sha256"],
            "online_graph": result["online_graph_source_sha256"],
            "online_corpus": result["online_corpus_source_sha256"],
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
