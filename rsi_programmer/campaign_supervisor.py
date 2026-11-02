"""Durable, bounded supervisor for programmer development campaigns."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ["clip_negative_readings", "cap_positive_readings",
                 "negative_magnitudes"]
DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1, "worker": [sys.executable, "-m",
                                      "scripts.campaign_trial"],
    "development_task_ids": DEFAULT_TASKS,
    "recipes": [{"execution_context": False, "seed": 0, "width": 32,
                 "seconds": 3.0, "token_choices": 4},
                {"execution_context": True, "seed": 0, "width": 32,
                 "seconds": 3.0, "token_choices": 4}],
    "max_trials": 4, "max_wall_seconds": 1800,
    "max_stagnant_trials": 2, "trial_timeout_seconds": 300,
    "retry_count": 1, "retry_backoff_seconds": 3,
    "poll_seconds": 2, "min_available_gib": 9,
    "stop_available_gib": 5, "memory_wait_seconds": 600,
    "audit_task_ids": [], "held_seed": 723031,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest() -> dict[str, str]:
    paths = list((ROOT / "rsi_programmer").glob("array_*.py"))
    paths.extend(ROOT / "rsi_programmer" / name for name in (
        "array_curriculum.py", "utility_policy.py", "statement_source.py",
        "campaign_supervisor.py", "campaign_worker.py",
        "benchmark_policy.py", "program_cost.py"))
    paths.extend(ROOT / "scripts" / name for name in (
        "programmer_campaign.py", "campaign_trial.py",
        "check_statement_curlee.py",
        "check_useful_curlee.py", "measure_useful_program.py"))
    return {str(path.relative_to(ROOT)): _digest(path) for path in paths
            if path.exists()}


def _append_event(directory: Path, kind: str, **details: Any) -> None:
    event = {"time": time.time(), "event": kind, **details}
    with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _available_gib() -> float:
    values = Path("/proc/meminfo").read_text().splitlines()
    available = next(line for line in values
                     if line.startswith("MemAvailable:"))
    return int(available.split()[1]) / (1024 * 1024)


def _status(directory: Path, state: str, **extra: Any) -> dict[str, Any]:
    path = directory / "status.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update({"state": state, "updated_at": time.time(), **extra})
    _atomic_json(path, current)
    return current


def _recipe_key(recipe: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(recipe)).hexdigest()[:16]


def _stop_requested(directory: Path) -> bool:
    return (directory / "STOP").exists()


def _wall_remaining(directory: Path, config: dict[str, Any]) -> float:
    status = json.loads((directory / "status.json").read_text())
    started = status.get("started_at", time.time())
    return max(0.0, config["max_wall_seconds"] - (time.time() - started))


def _run_trial(directory: Path, config: dict[str, Any], spec: dict[str, Any],
               trial_dir: Path, audit: bool = False) -> dict[str, Any]:
    if trial_dir.exists():
        prior = trial_dir / "result.json"
        if prior.exists():
            try:
                recovered = json.loads(prior.read_text())
            except (OSError, json.JSONDecodeError):
                recovered = None
            if isinstance(recovered, dict):
                _append_event(directory, "trial_receipt_recovered",
                              trial=str(trial_dir))
                return recovered
        suffix = 1
        while (directory / f"{trial_dir.name}-attempt-{suffix}").exists():
            suffix += 1
        replacement = directory / f"{trial_dir.name}-attempt-{suffix}"
        _append_event(directory, "incomplete_trial_preserved",
                      trial=str(trial_dir), retry=str(replacement))
        trial_dir = replacement
    trial_dir.mkdir(parents=True, exist_ok=False)
    spec_path = trial_dir / "spec.json"
    _atomic_json(spec_path, spec)
    command = [*config["worker"], "--spec", str(spec_path), "--output",
               str(trial_dir)]
    if audit:
        command.extend(["--audit", "--baseline-recipe",
                        json.dumps(spec["baseline_recipe"], sort_keys=True),
                        "--selected-recipe",
                        json.dumps(spec["selected_recipe"], sort_keys=True)])
    started = time.monotonic()
    attempt = 0
    admission_waits = 0
    while True:
        if _stop_requested(directory):
            return {"status": "stopped", "attempts": attempt}
        if _wall_remaining(directory, config) <= 0:
            return {"status": "wall_limit"}
        memory = _available_gib()
        if memory < config["min_available_gib"]:
            _status(directory, "WAITING_FOR_RESOURCES",
                    available_gib=memory, trial=spec.get("trial_index"))
            time.sleep(config["poll_seconds"])
            admission_waits += 1
            if admission_waits * config["poll_seconds"] >= config[
                    "memory_wait_seconds"]:
                return {"status": "memory_admission_timeout",
                        "available_gib": memory}
            continue
        attempt += 1
        log_path = trial_dir / f"worker-{attempt}.log"
        _append_event(directory, "trial_start", trial=str(trial_dir),
                      attempt=attempt, available_gib=memory)
        with log_path.open("w", encoding="utf-8") as log:
            lock_path = Path("/tmp/rsi-programmer-builder.lock")
            with lock_path.open("a") as build_lock:
                lock_deadline = time.monotonic() + config.get(
                    "builder_lock_wait_seconds", 300)
                while True:
                    try:
                        fcntl.flock(build_lock,
                                    fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        _status(directory, "WAITING_FOR_RESOURCES",
                                reason="builder lock is busy")
                        if (_stop_requested(directory) or
                                time.monotonic() >= lock_deadline or
                                _wall_remaining(directory, config) <= 0):
                            if _wall_remaining(directory, config) <= 0:
                                return {"status": "wall_limit"}
                            return {"status": "builder_lock_timeout"}
                        time.sleep(config["poll_seconds"])
                process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                           stderr=subprocess.STDOUT,
                                           start_new_session=True,
                                           env={**os.environ,
                                                "RSI_CAMPAIGN_LOCK_HELD": "1"})
                deadline = time.monotonic() + config[
                    "trial_timeout_seconds"]
                deadline = min(deadline, time.monotonic() +
                               _wall_remaining(directory, config))
                interrupted: str | None = None
                while process.poll() is None:
                    memory_now = _available_gib()
                    if _stop_requested(directory):
                        interrupted = "stopped"
                    elif memory_now < config["stop_available_gib"]:
                        interrupted = "resource_stop"
                    elif time.monotonic() >= deadline:
                        exhausted = _wall_remaining(directory, config) <= 0
                        interrupted = "wall_limit" if exhausted else "timeout"
                    if interrupted:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        break
                    time.sleep(min(0.5, config["poll_seconds"]))
                return_code = process.wait()
                fcntl.flock(build_lock, fcntl.LOCK_UN)
        if interrupted in ("stopped", "resource_stop"):
            return {"status": interrupted,
                    "reason": "supervisor interrupted worker",
                    "available_gib": _available_gib()}
        if interrupted == "wall_limit":
            return {"status": "wall_limit"}
        if interrupted == "timeout":
            return {"status": "infrastructure_failure",
                    "return_code": 124, "reason": "trial timeout"}
        result_path = trial_dir / "result.json"
        if return_code == 0 and result_path.exists():
            try:
                result = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                result = {"status": "infrastructure_failure",
                          "error": f"invalid result.json: {error}"}
            if not isinstance(result, dict):
                result = {"status": "infrastructure_failure",
                          "error": "result.json must contain an object"}
            if result.get("status") not in ("completed",
                                             "infrastructure_failure"):
                result = {"status": "infrastructure_failure",
                          "error": "worker returned unsupported status"}
            if result.get("status") == "infrastructure_failure":
                _append_event(directory, "worker_result_invalid",
                              trial=str(trial_dir),
                              error=result.get("error"))
                return result
            valid, reason = _validate_worker_result(result, spec)
            if not valid:
                _append_event(directory, "worker_result_invalid",
                              trial=str(trial_dir), error=reason)
                return {"status": "infrastructure_failure",
                        "error": reason}
            result["supervisor_attempt"] = attempt
            _append_event(directory, "trial_end", trial=str(trial_dir),
                          status=result.get("status"),
                          elapsed=time.monotonic() - started)
            return result
        _append_event(directory, "trial_failure", trial=str(trial_dir),
                      attempt=attempt, return_code=return_code,
                      result_exists=result_path.exists())
        if attempt > config["retry_count"]:
            return {"status": "infrastructure_failure",
                    "return_code": return_code,
                    "log": str(log_path.relative_to(directory))}
        time.sleep(config["retry_backoff_seconds"])


def _validate_worker_result(result: dict[str, Any],
                            spec: dict[str, Any]) -> tuple[bool, str]:
    phase = spec["phase"]
    if result.get("phase") != phase:
        return False, "worker phase does not match spec"
    if result.get("trial_index") != spec["trial_index"]:
        return False, "worker trial index does not match spec"
    if result.get("source_sha256") != spec.get("source_sha256"):
        return False, "worker source hashes do not match immutable spec"
    if phase == "development":
        if result.get("recipe") != spec["recipe"]:
            return False, "worker recipe does not match frozen recipe"
        development = result.get("development")
        if not isinstance(development, dict):
            return False, "missing development metrics"
        rows = development.get("per_task")
        expected_ids = spec["development_task_ids"]
        if not isinstance(rows, list) or len(rows) != len(expected_ids):
            return False, "development task results are incomplete"
        if [row.get("task_id") for row in rows
                if isinstance(row, dict)] != expected_ids:
            return False, "development task IDs do not match spec"
        score = development.get("qualified_tasks")
        count = development.get("task_count")
        if type(score) is not int or type(count) is not int:
            return False, "development counts must be integers"
        qualified = [row.get("qualified") for row in rows]
        if any(type(value) is not bool for value in qualified):
            return False, "task qualification must be boolean evidence"
        if count != len(expected_ids) or score != sum(qualified):
            return False, "development summary disagrees with task evidence"
        if score < 0 or score > count:
            return False, "development score is outside task bounds"
        return True, ""
    if phase == "audit":
        if result.get("fresh_task_ids") != spec.get("audit_task_ids"):
            return False, "audit task IDs do not match spec"
        for label in ("baseline", "selected"):
            record = result.get(label)
            if not isinstance(record, dict):
                return False, f"missing {label} audit record"
            recipe_key = ("baseline_recipe" if label == "baseline" else
                          "selected_recipe")
            if record.get("recipe") != spec.get(recipe_key):
                return False, f"{label} recipe does not match audit spec"
            rows = record.get("per_task")
            if not isinstance(rows, list) or len(rows) != len(
                    spec.get("audit_task_ids", [])):
                return False, f"incomplete {label} audit task results"
            score = record.get("qualified_tasks")
            count = record.get("task_count")
            if type(score) is not int or type(count) is not int:
                return False, f"invalid {label} audit summary counts"
            qualified = [row.get("qualified") for row in rows
                         if isinstance(row, dict)]
            if (len(qualified) != len(rows) or
                    any(type(value) is not bool for value in qualified) or
                    score != sum(qualified) or
                    count != len(spec.get("audit_task_ids", []))):
                return False, f"{label} audit summary disagrees with evidence"
            if [row.get("task_id") for row in rows] != spec.get(
                    "audit_task_ids", []):
                return False, f"{label} audit task IDs do not match spec"
        return True, ""
    return False, "unsupported worker phase"
def start(directory: Path, config_path: Path | None) -> int:
    directory.mkdir(parents=True, exist_ok=False)
    config = (json.loads(config_path.read_text()) if config_path else
              DEFAULT_CONFIG.copy())
    _validate_config(config)
    config_hash = hashlib.sha256(_canonical(config)).hexdigest()
    immutable = {"config": config, "config_sha256": config_hash,
                 "source_sha256": _source_manifest(),
                 "created_at": time.time()}
    _atomic_json(directory / "campaign.json", immutable)
    _atomic_json(directory / "status.json", {
        "state": "RUNNING", "started_at": time.time(), "trial": 0,
        "best_score": None, "stagnant_trials": 0})
    _append_event(directory, "campaign_start", config_sha256=config_hash)
    return _execute(directory, config)


def _validate_config(config: dict[str, Any]) -> None:
    required = ("worker", "development_task_ids", "recipes", "max_trials",
                "max_wall_seconds", "max_stagnant_trials",
                "trial_timeout_seconds", "retry_count", "poll_seconds",
                "min_available_gib", "stop_available_gib",
                "memory_wait_seconds", "retry_backoff_seconds")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing config fields: {missing}")
    if not config["recipes"] or not config["development_task_ids"]:
        raise ValueError("recipes and development tasks must be nonempty")
    if config["stop_available_gib"] >= config["min_available_gib"]:
        raise ValueError("stop threshold must be below admission threshold")


def _execute(directory: Path, config: dict[str, Any]) -> int:
    lock_path = directory / "supervisor.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((directory / "status.json").read_text())
        if state.get("state") in (
                "GENERALIZATION_GAIN", "LOCAL_GAIN_ONLY",
                "NO_MEASURED_GAIN", "INFRASTRUCTURE_FAILURE",
                "SOURCE_DRIFT", "WALL_LIMIT"):
            return 0
        started = state.get("started_at", time.time())
        results = state.get("results", [])
        best_score = state.get("best_score")
        stagnant = state.get("stagnant_trials", 0)
        first = len(results)
        for index in range(first, config["max_trials"]):
            manifest = json.loads((directory / "campaign.json").read_text())
            if _source_manifest() != manifest["source_sha256"]:
                _status(directory, "SOURCE_DRIFT", results=results)
                return 1
            if _stop_requested(directory):
                _status(directory, "STOPPED", results=results)
                _append_event(directory, "campaign_stopped")
                return 0
            if time.time() - started >= config["max_wall_seconds"]:
                _status(directory, "WALL_LIMIT", results=results)
                return 0
            recipe = config["recipes"][index % len(config["recipes"])]
            trial_dir = directory / f"trial-{index + 1:03d}"
            spec = {"schema_version": 1, "phase": "development",
                    "trial_index": index + 1, "recipe": recipe,
                    "development_task_ids": config["development_task_ids"],
                    "held_seed": config.get("held_seed", 723031),
                    "search_seconds": config.get("search_seconds", 3.0),
                    "max_cases_per_task": config.get("max_cases_per_task", 40),
                    "source_sha256": _source_manifest()}
            _status(directory, "RUNNING", trial=index + 1, results=results)
            result = _run_trial(directory, config, spec, trial_dir)
            if result.get("status") == "resource_stop":
                _append_event(directory, "trial_resource_stopped",
                              trial=index + 1, details=result)
                _status(directory, "RESOURCE_BLOCKED", results=results)
                return 0
            if result.get("status") in ("stopped", "memory_admission_timeout"):
                state_name = ("STOPPED" if result.get("status") == "stopped"
                              else "RESOURCE_BLOCKED")
                _append_event(directory, "trial_wait_interrupted",
                              trial=index + 1, details=result)
                _status(directory, state_name, results=results)
                return 0
            if result.get("status") == "builder_lock_timeout":
                _append_event(directory, "builder_lock_timeout",
                              trial=index + 1)
                _status(directory, "RESOURCE_BLOCKED", results=results,
                        reason="builder lock did not become available")
                return 0
            if result.get("status") == "wall_limit":
                _status(directory, "WALL_LIMIT", results=results)
                return 0
            results.append(result)
            if result.get("status") != "completed":
                _append_event(directory, "trial_not_measured",
                              trial=index + 1,
                              status=result.get("status"))
                _status(directory, "RUNNING", trial=index + 1,
                        results=results, best_score=best_score,
                        stagnant_trials=stagnant)
                continue
            result.setdefault("trial_index", index + 1)
            result.setdefault("recipe", recipe)
            score = result.get("development", {}).get("qualified_tasks", 0)
            improved = best_score is None or score > best_score
            stagnant = 0 if improved else stagnant + 1
            if improved:
                best_score = score
            state = _status(directory, "RUNNING", trial=index + 1,
                            results=results, best_score=best_score,
                            stagnant_trials=stagnant)
            if stagnant >= config["max_stagnant_trials"]:
                break
        _finalize(directory, config, results, best_score)
    return 0


def _finalize(directory: Path, config: dict[str, Any],
              results: list[dict[str, Any]], best_score: int | None) -> None:
    manifest = json.loads((directory / "campaign.json").read_text())
    if _source_manifest() != manifest["source_sha256"]:
        _status(directory, "SOURCE_DRIFT", results=results)
        return
    successes = [item for item in results
                 if item.get("status") == "completed"]
    if not successes:
        state = ("INFRASTRUCTURE_FAILURE" if results else
                 "NO_MEASURED_GAIN")
        _status(directory, state, results=results,
                conclusion="No completed worker result.")
        return
    winner = max(successes, key=lambda item: item.get(
        "development", {}).get("qualified_tasks", 0))
    if winner.get("development", {}).get("qualified_tasks", 0) <= 0:
        _status(directory, "NO_MEASURED_GAIN", results=results,
                conclusion="No target program passed independent execution.")
        return
    recipe = winner.get("recipe")
    if not isinstance(recipe, dict):
        _status(directory, "NO_MEASURED_GAIN", results=results,
                conclusion="Worker did not return a frozen recipe.")
        return
    _atomic_json(directory / "frozen-winner.json", {
        "recipe": recipe, "recipe_sha256": _recipe_key(recipe),
        "source_sha256": _source_manifest(),
        "trial": winner.get("trial_index"), "development": winner.get(
            "development")})
    audit_ids = config.get("audit_task_ids", [])
    if not audit_ids:
        _status(directory, "LOCAL_GAIN_ONLY", results=results,
                conclusion="No fresh sealed audit tasks were configured.")
        return
    baseline = config.get("baseline_recipe", {})
    spec = {"schema_version": 1, "phase": "audit",
            "trial_index": len(results) + 1, "audit_task_ids": audit_ids,
            "baseline_recipe": baseline, "selected_recipe": recipe,
            "held_seed": config.get("held_seed", 723031),
            "source_sha256": _source_manifest()}
    if config.get("audit_tasks"):
        spec["audit_tasks"] = config["audit_tasks"]
    audit_dir = directory / "audit"
    audit = _run_trial(directory, config, spec, audit_dir, audit=True)
    _atomic_json(directory / "audit-result.json", audit)
    if audit.get("status") in ("resource_stop", "memory_admission_timeout",
                                "builder_lock_timeout"):
        _status(directory, "RESOURCE_BLOCKED", results=results, audit=audit)
        return
    if audit.get("status") == "stopped":
        _status(directory, "STOPPED", results=results, audit=audit)
        return
    if audit.get("status") == "wall_limit":
        _status(directory, "WALL_LIMIT", results=results, audit=audit)
        return
    if audit.get("status") != "completed":
        _status(directory, "INFRASTRUCTURE_FAILURE", results=results,
                audit=audit,
                conclusion="Audit did not produce a completed result.")
        return
    verdict = audit.get("verdict", "NO_MEASURED_GAIN")
    if verdict == "GENERALIZATION_GAIN" and not _valid_generalization(
            audit, config):
        verdict = "NO_MEASURED_GAIN"
    if verdict not in ("GENERALIZATION_GAIN", "LOCAL_GAIN_ONLY",
                       "NO_MEASURED_GAIN"):
        verdict = "NO_MEASURED_GAIN"
    _status(directory, verdict, results=results, audit=audit,
            conclusion=audit.get("conclusion", verdict))


def _valid_generalization(audit: dict[str, Any],
                          config: dict[str, Any]) -> bool:
    evidence = audit.get("generalization_evidence", {})
    ids = set(audit.get("fresh_task_ids", []))
    dev_ids = set(config["development_task_ids"])
    families = set(evidence.get("families", []))
    configured_ids = set(config.get("audit_task_ids", []))
    declared_families = {item.get("family_id") for item in
                         config.get("audit_tasks", [])
                         if isinstance(item, dict)}
    return (bool(ids) and ids == configured_ids and ids.isdisjoint(dev_ids)
            and len(families) >= 2 and len(declared_families) >= 2
            and families.issubset(declared_families)
            and evidence.get("unseen_family_passed") is True
            and evidence.get("cross_family_passed") is True
            and evidence.get("retention_passed") is True
            and evidence.get("actual_execution_passed") is True)


def resume(directory: Path) -> int:
    manifest = json.loads((directory / "campaign.json").read_text())
    if hashlib.sha256(_canonical(manifest["config"])).hexdigest() != \
            manifest["config_sha256"]:
        raise ValueError("immutable campaign config hash mismatch")
    if manifest["source_sha256"] != _source_manifest():
        raise ValueError("source changed since campaign start")
    (directory / "STOP").unlink(missing_ok=True)
    return _execute(directory, manifest["config"])


def stop(directory: Path) -> None:
    (directory / "STOP").write_text("stop requested\n", encoding="utf-8")
    _append_event(directory, "stop_requested")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start_parser = commands.add_parser("start")
    start_parser.add_argument("directory", type=Path)
    start_parser.add_argument("--config", type=Path)
    for name in ("status", "resume", "stop"):
        command = commands.add_parser(name)
        command.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "start":
        return start(args.directory, args.config)
    if args.command == "status":
        print((args.directory / "status.json").read_text())
        return 0
    if args.command == "stop":
        stop(args.directory)
        return 0
    return resume(args.directory)


if __name__ == "__main__":
    raise SystemExit(main())
