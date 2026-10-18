#!/usr/bin/env python3
"""Paired train/held-out ablations for author and beam ranking (#70)."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rsi_programmer.author import EXTRA, TaskRow, author_search
from rsi_programmer.author import beam_search, refine_author, train_author
from rsi_programmer.policy import SearchPolicy
from rsi_programmer.probe import interpret, validate_graph
from rsi_programmer.traces import TraceRecorder

DEFAULT_TRAIN = ROOT / "fixtures" / "rsi_shadow_tasks.json"
DEFAULT_EVAL = (ROOT / "fixtures" / "rsi_shadow_tasks.json",
                ROOT / "fixtures" / "rsi_novel_tasks.json")
VARIANTS = ("full", "no_exact_flag", "no_open_gap", "no_both", "no_neural")
SUPERVISION_MODES = ("canonical", "reachability")
DEFAULT_BUDGET = 128


class ZeroScorer:
    """Use the classical bonuses with a zero neural ranking contribution."""

    @staticmethod
    def scores(feats: list[list[float]]) -> list[float]:
        return [0.0] * len(feats)


def _tasks(path: Path) -> tuple[list[TaskRow], list[dict[str, Any]]]:
    """Load train rows and preserve frozen held-out expectations."""
    family = cast(dict[str, Any], TraceRecorder(str(path)).load_family())
    tasks = cast(list[dict[str, Any]], family["tasks"])
    search: list[TaskRow] = []
    for task in tasks:
        train = cast(list[list[int]], task["train"])
        rows = tuple((case[0], case[1]) for case in train)
        target = tuple(case[2] for case in train)
        bound = int(task.get("max_tree_ops", 4))
        search.append((str(task["id"]), rows, target, bound))
    return search, tasks


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_ref(path: Path) -> str:
    """Use portable repository-relative fixture names in saved output."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(
            "fixture paths must be inside the repository") from exc


def _source_hash() -> str:
    """Hash implementation inputs used by the author measurement."""
    paths = sorted((ROOT / "rsi_programmer").glob("*.py"))
    paths.append(ROOT / "scripts" / "measure_author_ablation.py")
    paths.sort()
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fixture_hashes(paths: list[Path]) -> dict[str, str]:
    """Hash all fixture files by their portable repository-relative names."""
    return {
        _fixture_ref(path): _sha(path)
        for path in sorted(set(paths), key=lambda item: _fixture_ref(item))
    }


def _package_roots(package_name: str) -> list[Path]:
    """Resolve every imported package location without publishing paths."""
    package = importlib.import_module(package_name)
    locations = getattr(package, "__path__", None)
    if locations is None:
        file_name = getattr(package, "__file__", None)
        if file_name is None:
            raise RuntimeError(
                f"cannot locate imported package {package_name}")
        locations = [str(Path(file_name).parent)]
    roots = sorted({Path(item).resolve() for item in locations})
    if not roots:
        raise RuntimeError(
            f"imported package {package_name} has no source path")
    return roots


def _package_content(roots: list[Path]) -> tuple[str, int]:
    """Hash imported Python sources using package-relative identifiers."""
    files: list[tuple[str, Path]] = []
    for root_index, root in enumerate(roots):
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            files.append((f"{root_index}:{relative}", path))
    files.sort(key=lambda item: item[0])
    if not files:
        raise RuntimeError("imported dependency has no Python source files")
    digest = hashlib.sha256()
    for identifier, path in files:
        digest.update(identifier.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), len(files)


def _dependency_git(roots: list[Path]) -> dict[str, Any]:
    """Return commit and dirty status when a package root is in a Git tree."""
    git_root: Path | None = None
    for root in roots:
        for candidate in (root, *root.parents):
            if (candidate / ".git").exists():
                git_root = candidate
                break
        if git_root is not None:
            break
    if git_root is None:
        return {"git_commit": None, "git_dirty": None}
    commit = subprocess.run(
        ["git", "-C", str(git_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(git_root), "status", "--porcelain",
         "--untracked-files=all"], check=True, capture_output=True,
        text=True).stdout
    return {"git_commit": commit, "git_dirty": bool(status)}


def _dependency_provenance(package_name: str) -> dict[str, Any]:
    """Describe imported dependency content without exposing machine paths."""
    roots = _package_roots(package_name)
    content_hash, file_count = _package_content(roots)
    return {
        "package": package_name,
        "content_sha256": content_hash,
        "python_file_count": file_count,
        **_dependency_git(roots),
    }


def _assert_provenance_unchanged(
        label: str, before: dict[str, str], after: dict[str, str]
        ) -> None:
    """Fail if files used during measurement changed before reporting."""
    if before != after:
        raise RuntimeError(f"{label} changed during measurement")


def _weights(base: list[float], variant: str) -> list[float]:
    """Return a paired bonus ablation of the fitted full configuration."""
    result = list(base)
    indices = {name: EXTRA.index(name) for name in EXTRA}
    if variant in ("no_exact_flag", "no_both"):
        result[indices["exact_flag"]] = 0.0
    if variant in ("no_open_gap", "no_both"):
        result[indices["open_gap"]] = 0.0
    return result


def _rows_correct(graph: dict[str, Any] | None,
                  rows: list[tuple[int, int, int]]
                  ) -> tuple[bool, int, int]:
    """Independently check a graph against input/output rows."""
    if graph is None:
        return False, 0, len(rows)
    validate_graph(graph)
    hits = sum(interpret(graph, x, y) == expected
               for x, y, expected in rows)
    return hits == len(rows), hits, len(rows)


def _held_correct(graph: dict[str, Any] | None,
                  task: dict[str, Any]) -> tuple[bool, int, int]:
    """Check a returned graph only against the frozen held-out rows."""
    held = cast(list[list[int]], task["held_out"])
    rows = [(case[0], case[1], case[2]) for case in held]
    return _rows_correct(graph, rows)


def _measure_one(
        policy: Any, variant: str, path: str, task_row: TaskRow,
        task_data: dict[str, Any], weights: list[float], budget: int,
        width: int, topk: int
        ) -> dict[str, Any]:
    """Run one configuration on one task and retain raw outcomes."""
    task_id, rows, target, bound = task_row
    started = time.perf_counter()
    if path == "author":
        result = author_search(
            policy, rows, target, bound, budget, task_id, topk,
            weights, fallback=False)
    else:
        result = beam_search(
            policy, rows, target, bound, budget, task_id, width,
            weights, fallback=False)
    graph = result.get("graph")
    train_cases = [(x, y, expected)
                   for (x, y), expected in zip(rows, target)]
    train_correct, train_hits, train_count = _rows_correct(
        graph, train_cases)
    held_correct, held_hits, held_count = _held_correct(graph, task_data)
    return {
        "seed": None,
        "task": task_id,
        "path": path,
        "variant": variant,
        "weights": list(weights),
        "neural_enabled": variant != "no_neural",
        "solved_train": bool(result["solved"]),
        "train_correct": train_correct,
        "train_hits": train_hits,
        "train_count": train_count,
        "held_correct": held_correct,
        "held_hits": held_hits,
        "held_count": held_count,
        "accepted": train_correct and held_correct,
        "verdict": result["verdict"],
        "signature_evaluations": int(result["signature_evaluations"]),
        "operand_signature_evaluations": int(
            result["operand_signature_evaluations"]),
        "examined": int(result["examined"]),
        "committed_nodes": int(result["committed_nodes"]),
        "output_nodes": int(result["output_nodes"]),
        "budget_unit": result["budget_unit"],
        "wall_s": time.perf_counter() - started,
    }


def measure(args: argparse.Namespace) -> dict[str, Any]:
    """Train on shadow tasks, tune bonuses there, evaluate held-out tasks."""
    source_start = _source_hash()
    train_path = args.train_fixture.resolve()
    train_ref = _fixture_ref(train_path)
    eval_paths = [path.resolve() for path in args.eval_fixtures]
    for path in eval_paths:
        _fixture_ref(path)
    fixture_hashes_start = _fixture_hashes([train_path, *eval_paths])
    dependency_start = _dependency_provenance("spikeforge")
    train_rows, _ = _tasks(train_path)
    evaluation: list[tuple[
        str, Path, list[TaskRow], list[dict[str, Any]]]] = []
    for resolved in eval_paths:
        rows, tasks = _tasks(resolved)
        evaluation.append((str(resolved), resolved, rows, tasks))

    raw: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    for seed in args.seeds:
        for supervision in SUPERVISION_MODES:
            policy = SearchPolicy(seed=seed)
            started = time.perf_counter()
            supervision_report: list[dict[str, Any]] = []
            groups = train_author(
                policy, str(train_path), epochs=args.epochs,
                supervision=supervision,
                supervision_report=supervision_report)
            fitted, tuning = refine_author(
                policy, train_rows, rounds=args.rounds, seed=seed,
                topk=args.topk, budget=args.budget)
            training.append({
                "seed": seed,
                "supervision": supervision,
                "training_feature_label_pairs": groups,
                "supervision_tasks": supervision_report,
                "weights": list(fitted),
                "tuning": tuning,
                "wall_s": time.perf_counter() - started,
            })
            for fixture, _path, tasks, task_data in evaluation:
                by_id = {str(task["id"]): task for task in task_data}
                for path in ("author", "beam"):
                    for variant in VARIANTS:
                        config = _weights(fitted, variant)
                        scorer: Any = ZeroScorer() \
                            if variant == "no_neural" else policy
                        for task_row in tasks:
                            row = _measure_one(
                                scorer, variant, path, task_row,
                                by_id[task_row[0]], config, args.budget,
                                args.width, args.topk)
                            row["seed"] = seed
                            row["supervision"] = supervision
                            row["fixture"] = fixture
                            raw.append(row)

    fixture_hashes_end = _fixture_hashes([train_path, *eval_paths])
    _assert_provenance_unchanged(
        "fixture files", fixture_hashes_start, fixture_hashes_end)
    dependency_end = _dependency_provenance("spikeforge")
    if (dependency_start["content_sha256"]
            != dependency_end["content_sha256"]):
        raise RuntimeError(
            "SpikeForge Python sources changed during measurement")
    source_end = _source_hash()
    if source_start != source_end:
        raise RuntimeError("source files changed during measurement")
    return {
        "schema": "author-ablation-v1",
        "scope": {
            "training": "shadow train rows only",
            "evaluation": ("fixture train rows for search; frozen held-out "
                           "rows for correctness"),
            "selection": "weights tuned only on shadow tasks",
            "shadow_eval": ("seen-task held-out evaluation; shadow train "
                            "rows are used for fitting and selection"),
            "novel_eval": "task-unseen held-out evaluation",
            "fallback": False,
            "paired": "same seeds, tasks, and path-specific budgets",
            "budget_units": {
                "author": "generated_triples",
                "beam": "committed_nodes",
            },
            "equal_work_claim": False,
            "work_cost": ("signature_evaluations are reported as canonical "
                          "work and may be ranker-invariant"),
        },
        "config": {
            "train_fixture": train_ref,
            "eval_fixtures": [_fixture_ref(item[1])
                              for item in evaluation],
            "seeds": args.seeds,
            "epochs": args.epochs,
            "rounds": args.rounds,
            "budget": args.budget,
            "topk": args.topk,
            "beam_width": args.width,
            "variants": list(VARIANTS),
            "supervision_modes": list(SUPERVISION_MODES),
            "bonus_fields": list(EXTRA),
        },
        "provenance": {
            "source_hash_scope": (
                "repository rsi_programmer Python files, measurement script, "
                "imported SpikeForge Python source bytes and fixture bytes"),
            "source_hash_method": (
                "SHA-256 over sorted repository-relative implementation "
                "paths and bytes; SpikeForge has a separate package-relative "
                "content hash; fixtures are hashed before and after"),
            "source_sha256_start": source_start,
            "source_sha256_finish": source_end,
            "spikeforge": dependency_start,
            "spikeforge_content_sha256_finish":
                dependency_end["content_sha256"],
            "fixture_sha256_start": fixture_hashes_start,
            "fixture_sha256_finish": fixture_hashes_end,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "training": training,
        "coverage": {
            "training_family_tasks": len(train_rows),
            "evaluation_families": [
                {"fixture": _fixture_ref(item[1]),
                 "task_count": len(item[2]),
                 "task_ids": [row[0] for row in item[2]]}
                for item in evaluation
            ],
        },
        "results": raw,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-fixture", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-fixture", type=Path, action="append",
                        dest="eval_fixtures")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.eval_fixtures is None:
        args.eval_fixtures = list(DEFAULT_EVAL)
    if args.epochs < 0 or args.rounds < 0 or args.budget < 0:
        parser.error("epochs, rounds, and budget must be non-negative")
    if not args.seeds or args.topk < 1 or args.width < 1:
        parser.error("provide at least one seed and positive topk/width")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    start = time.perf_counter()
    report = measure(args)
    report["wall_s"] = time.perf_counter() - start
    report["invocation"] = [
        "scripts/measure_author_ablation.py", *sys.argv[1:]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "results": len(report["results"]),
                      "wall_s": report["wall_s"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
