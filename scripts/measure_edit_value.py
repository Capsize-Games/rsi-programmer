"""Run one bounded edit-value comparison over the frozen list task bank.

This entry point performs fits and target-language checks. Run it only through
the repository's resource-guarded campaign supervisor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from rsi_programmer.array_curriculum import probes, teachers
from rsi_programmer.array_execution import execute
from rsi_programmer.array_policy import ArrayPolicy
from rsi_programmer.array_tasks import TASKS, oracle
from rsi_programmer.benchmark_policy import state_fingerprint
from rsi_programmer.edit_value_ranker import EditValueRanker
from rsi_programmer.campaign_worker import _execute_task
from rsi_programmer.program_cost import CostClock
from scripts.check_edit_value import BANK, ROOT, verify, verify_pre_fit

BASELINE = ROOT / "weights/campaigns/6b844a6c0bd742d286d7567a/selected.pt"
BASELINE_SHA256 = (
    "16a852d7b2c2ebb37e9926b5ed1dca22c569dd24a5f786fe071268aca67be746")
BASELINE_STATUS = ROOT / "artifacts/campaign-initial-20261102/status.json"
COMPILER = Path("/home/joe/Projects/curlee/build/linux-debug/curlee")
CURLEE_GUARD = Path("/home/joe/Projects/curlee/scripts/run_limited.py")
WEIGHTS_DIR = ROOT / "weights"
SOURCE_PATHS = (
    "rsi_programmer/array_curriculum.py",
    "rsi_programmer/array_execution.py",
    "rsi_programmer/array_partial.py",
    "rsi_programmer/array_policy.py",
    "rsi_programmer/array_source.py",
    "rsi_programmer/array_syntax.py",
    "rsi_programmer/array_tasks.py",
    "rsi_programmer/campaign_worker.py",
    "rsi_programmer/edit_value_ranker.py",
    "rsi_programmer/statement_execution.py",
    "rsi_programmer/statement_syntax.py",
    "rsi_programmer/statement_source.py",
    "rsi_programmer/utility_ops.py",
    "rsi_programmer/utility_policy.py",
    "rsi_programmer/utility_source.py",
    "scripts/check_edit_value.py",
    "scripts/check_statement_curlee.py",
    "scripts/check_useful_curlee.py",
    "scripts/measure_edit_value.py",
    "scripts/measure_useful_program.py",
    "fixtures/edit_value_v1/protocol.json",
    "fixtures/edit_value_v1/fresh_examples.json",
    "fixtures/edit_value_v1/sha256.json",
    "artifacts/campaign-initial-20261102/status.json",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=list) + "\n")
    temporary.replace(path)


def _checkpoint_path(output: Path, source_hashes: dict[str, str]) -> Path:
    identity = json.dumps({"output": str(output.resolve()),
                           "source_sha256": source_hashes},
                          sort_keys=True, separators=(",", ":"))
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return WEIGHTS_DIR / f"edit-value-{suffix}.pt"


def _save_checkpoint(state: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    candidate_sha = _sha(temporary)
    if path.exists():
        if _sha(path) != candidate_sha:
            temporary.unlink()
            raise FileExistsError(
                f"checkpoint path already contains different weights: {path}")
        temporary.unlink()
        return candidate_sha
    temporary.replace(path)
    return candidate_sha


def _signature(program: tuple[str, ...], rows: list[list[int]]) -> str:
    raw = json.dumps([execute(program, values) for values in rows],
                     separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _source_hashes() -> dict[str, str]:
    files = {name: _sha(ROOT / name) for name in SOURCE_PATHS}
    files["baseline_checkpoint"] = _sha(BASELINE)
    files["curlee_compiler"] = _sha(COMPILER)
    files["curlee_resource_guard"] = _sha(CURLEE_GUARD)
    return files


def _ranker_fingerprint(model: EditValueRanker) -> str:
    return state_fingerprint(model.network.state_dict(), {
        "seed": model.seed, "architecture": "edit-value-ranker-v1"})


def _baseline_fingerprint(model: ArrayPolicy) -> str:
    return state_fingerprint(model.state_dict(), {
        "seed": 0, "execution_context": True})


def _baseline_metadata() -> dict[str, Any]:
    status = json.loads(BASELINE_STATUS.read_text())
    relative = BASELINE.relative_to(ROOT).as_posix()
    rows = [row for row in status.get("results", [])
            if row.get("checkpoint") == relative]
    if len(rows) != 1:
        raise ValueError("frozen ArrayPolicy baseline metadata is missing")
    row = rows[0]
    if row.get("checkpoint_sha256") != _sha(BASELINE):
        raise ValueError("baseline metadata checkpoint hash mismatch")
    return {"status_path": str(BASELINE_STATUS.relative_to(ROOT)),
            "status_sha256": _sha(BASELINE_STATUS),
            "checkpoint": relative,
            "checkpoint_sha256": row["checkpoint_sha256"],
            "recipe": row["recipe"],
            "model_fingerprint": row["model_fingerprint"],
            "prior_development_result": row["development"]}


def _corpus(excluded: set[str], limit: int) -> list[dict[str, Any]]:
    fixed = probes()
    result = []
    for program in teachers():
        signature = _signature(program, fixed)
        if signature in excluded:
            continue
        result.append({"signature": signature,
                       "pairs": [(values, execute(program, values))
                                 for values in fixed],
                       "programs": [program]})
        if len(result) == limit:
            break
    return result


def _fresh_signatures(protocol: dict[str, Any],
                      fixed: list[list[int]]) -> set[str]:
    return {_signature(tuple(row["witness"]), fixed)
            for row in protocol["fresh_tasks"]}


def _task_sets() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples = json.loads((BANK / "fresh_examples.json").read_text())
    fresh = [{"task_id": row["task_id"],
              "task_type": row["task_type"],
              "family_id": row["task_family"],
              "examples": row["examples"]} for row in examples]
    development = []
    observed = [[-3, 0, 1, 4, -1], [2, -2, 5, 1], [0, 3, -4, 2]]
    for name in TASKS:
        development.append({"task_id": name,
                            "task_type": "original_development_recheck",
                            "family_id": "bounded_integer_list",
                            "examples":[{"input": values,
                                         "output": oracle(name, values)}
                                        for values in observed]})
    return development, fresh


def run(output: Path, *, budget: int = 5000, beam_width: int = 12,
        training_tasks: int = 64, transitions: int = 2048,
        epochs: int = 30, rollouts: int = 2, rollout_steps: int = 32,
        search_seconds: float = 3.0
        ) -> dict[str, Any]:
    if min(budget, beam_width, training_tasks, transitions, epochs,
           rollouts, rollout_steps, search_seconds) < 1:
        raise ValueError("all trial limits must be positive")
    if (budget > 5000 or beam_width > 12 or training_tasks > 64
            or transitions > 2048 or epochs > 30 or rollouts > 2
            or rollout_steps > 32 or search_seconds > 3.0):
        raise ValueError("trial settings exceed the preregistered bounds")
    torch.set_num_threads(1)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    clock = CostClock()
    initial = verify_pre_fit()
    protocol = json.loads((BANK / "protocol.json").read_text())
    source_hashes = _source_hashes()
    rows = probes()
    excluded = _fresh_signatures(protocol, rows)
    corpus = _corpus(excluded, training_tasks)
    corpus_signatures = {str(row["signature"]) for row in corpus}
    if corpus_signatures & excluded:
        raise ValueError("fresh signatures remain in training corpus")
    if _sha(BASELINE) != BASELINE_SHA256:
        raise ValueError("frozen original ArrayPolicy baseline changed")
    baseline_metadata = _baseline_metadata()
    _write_json(output / "run.json", {
        "status": "fitting", "protocol": initial,
        "source_sha256": source_hashes,
        "original_array_policy_baseline": baseline_metadata,
        "fresh_excluded_signatures": sorted(excluded),
        "training_task_count": len(corpus),
        "fit_limits": {"max_transitions": transitions,
                       "max_tasks": training_tasks, "epochs": epochs,
                       "rollouts": rollouts,
                       "rollout_steps": rollout_steps,
                       "search_seconds": search_seconds, "seed": 0}})
    if _source_hashes() != source_hashes:
        raise ValueError("source changed before fit")
    trained = EditValueRanker(seed=0)
    fit = trained.fit(corpus, excluded_signatures=excluded,
                      epochs=epochs, rollout_count=rollouts,
                      rollout_steps=rollout_steps,
                      max_transitions=transitions,
                      max_tasks=training_tasks)
    fit_fingerprint = _ranker_fingerprint(trained)
    fit_checkpoint = _checkpoint_path(output, source_hashes)
    checkpoint_sha = _save_checkpoint(trained.network.state_dict(),
                                      fit_checkpoint)
    fit_trace_path = output / "fit-traces.json"
    _write_json(fit_trace_path, {"traces": trained.traces})
    fit_artifacts = {
        "model_state_fingerprint": fit_fingerprint,
        "checkpoint_path": str(fit_checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "trace_path": fit_trace_path.name,
        "trace_sha256": _sha(fit_trace_path),
        "fit_cpu_seconds": fit.get("fit_cpu_seconds"),
    }
    _write_json(output / "run.json", {
        "status": "fit_complete", "protocol": initial,
        "source_sha256": source_hashes, "fit": fit,
        "fit_artifacts": fit_artifacts,
        "original_array_policy_baseline": baseline_metadata,
        "fresh_excluded_signatures": sorted(excluded),
        "training_task_count": len(corpus),
        "fit_limits": {"max_transitions": transitions,
                       "max_tasks": training_tasks, "epochs": epochs,
                       "rollouts": rollouts,
                       "rollout_steps": rollout_steps,
                       "search_seconds": search_seconds}})
    baseline = ArrayPolicy(seed=0, execution_context=True)
    baseline.load_state_dict(torch.load(BASELINE, map_location="cpu",
                                        weights_only=True))
    baseline.eval()
    baseline_before = _baseline_fingerprint(baseline)
    if baseline_before != baseline_metadata["model_fingerprint"]:
        raise ValueError("ArrayPolicy fingerprint differs from metadata")
    development, fresh = _task_sets()
    all_tasks = development + fresh
    methods = {"edit_value_ranker": lambda pairs: trained.synthesize(
                   pairs, budget=budget, beam_width=beam_width,
                   max_seconds=search_seconds),
               "untrained_edit_value": lambda pairs:
                   trained.untrained_synthesize(
                       pairs, budget=budget, beam_width=beam_width,
                       max_seconds=search_seconds),
               "classical_diagnostic": lambda pairs:
                   trained.classical_synthesize(
                       pairs, budget=budget, beam_width=beam_width,
                       max_seconds=search_seconds),
               "original_array_policy": lambda pairs: baseline.synthesize(
                   pairs, width=256, seconds=search_seconds,
                   token_choices=8)}
    started = time.monotonic()
    choices: dict[str, list[dict[str, Any]]] = {}
    for label, synthesize in methods.items():
        choices[label] = []
        for task in all_tasks:
            pairs = [(row["input"], row["output"])
                     for row in task["examples"]]
            result = synthesize(pairs)
            tokens = result.get("program", result.get("tokens"))
            choices[label].append({"task_id": task["task_id"],
                                   "task_type": task["task_type"],
                                   "family_id": task["family_id"],
                                   "tokens": tokens,
                                   "search": result})
            with (output / "choices-progress.jsonl").open("a") as stream:
                stream.write(json.dumps({"method": label,
                                         "choice": choices[label][-1],
                                         "held_rows_opened": False},
                                        default=list) + "\n")
    if _source_hashes() != source_hashes:
        raise ValueError("source changed before choices were frozen")
    trained_after = _ranker_fingerprint(trained)
    baseline_after = _baseline_fingerprint(baseline)
    if fit_fingerprint != trained_after or baseline_before != baseline_after:
        raise ValueError("model state changed during inference")
    frozen = {"status": "choices_frozen", "protocol": initial,
              "source_sha256": source_hashes,
              "excluded_fresh_signatures": sorted(excluded),
              "training_signatures_sha256": hashlib.sha256(
                  "\n".join(sorted(corpus_signatures)).encode()).hexdigest(),
              "fit": fit, "methods": choices,
              "fit_artifacts": fit_artifacts,
              "original_array_policy_baseline": baseline_metadata,
              "model_state_sha256": {"edit_value_before": fit_fingerprint,
                                     "edit_value_after": trained_after,
                                     "array_policy_before": baseline_before,
                                     "array_policy_after": baseline_after},
              "cost_before_checks": clock.elapsed(),
              "choices_frozen_seconds": time.monotonic() - started}
    _write_json(output / "choices-frozen.json", frozen)

    # The held-case file is opened only after every method's choices exist.
    held_rows = {row["task_id"]: row for row in json.loads(
        (BANK / "held_cases.json").read_text())}
    execution: dict[str, list[dict[str, Any]]] = {}
    by_id = {row["task_id"]: row for row in fresh}
    for label, rows_for_method in choices.items():
        execution[label] = []
        for choice in rows_for_method:
            fresh_task = choice["task_id"] in by_id
            task = (dict(by_id[choice["task_id"]]) if fresh_task else {
                "task_id": choice["task_id"],
                "family_id": "bounded_integer_list"})
            if fresh_task:
                task.update(held_rows[choice["task_id"]])
            entry = {"label": label, "chosen": {"tokens": choice["tokens"]}}
            record = _execute_task(output, entry, task, 0, COMPILER,
                                   fresh_task)
            execution[label].append(record)
            _write_json(output / "result.partial.json", {
                "status": "backend_checks_in_progress",
                "method": label, "records": execution})
    held_validation = verify()
    if _source_hashes() != source_hashes:
        raise ValueError("source changed during backend checks")
    result = {"status": "completed", "phase": "development",
              "scope": protocol["domain"], "fresh_tasks": [
                  row["task_id"] for row in fresh],
              "baseline_recheck_tasks": [row["task_id"]
                                         for row in development],
              "retention_status": protocol["retention_status"],
              "fit": fit, "source_sha256": source_hashes,
              "fit_artifacts": fit_artifacts,
              "original_array_policy_baseline": baseline_metadata,
              "held_validation": held_validation,
              "cost": clock.elapsed(),
              "cost_scope": ("Whole process and completed children, including "
                             "fit, search and backend checks"),
              "choices": choices, "execution": execution,
              "fresh_held_qualified": {
                  label: sum(bool(row.get("qualified")) for row in records
                             if row["task_id"] in by_id)
                  for label, records in execution.items()},
              "development_recheck_qualified": {
                  label: sum(bool(row.get("qualified")) for row in records
                             if row["task_id"] not in by_id)
                  for label, records in execution.items()},
              "elapsed_wall_seconds": time.monotonic() - started}
    _write_json(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--beam-width", type=int, default=12)
    parser.add_argument("--training-tasks", type=int, default=64)
    parser.add_argument("--transitions", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--search-seconds", type=float, default=3.0)
    args = parser.parse_args()
    result = run(args.output, budget=args.budget,
                 beam_width=args.beam_width,
                 training_tasks=args.training_tasks,
                 transitions=args.transitions, epochs=args.epochs,
                 rollouts=args.rollouts, rollout_steps=args.rollout_steps,
                 search_seconds=args.search_seconds)
    print(json.dumps({"status": result["status"],
                      "fresh_held_qualified": result[
                          "fresh_held_qualified"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
