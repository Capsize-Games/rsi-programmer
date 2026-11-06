"""Bounded worker for array construction campaign trials.

Audit task descriptors are intentionally explicit and small. Each descriptor
has ``task_id``, ``family_id``, ``examples`` (input/output pairs),
``held_inputs``, and matching ``expected`` arrays. Held rows are consulted
only after all controller choices have been frozen.
"""
from __future__ import annotations

import json
import fcntl
import hashlib
import os
import random
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from rsi_programmer.array_policy import ArrayPolicy
from rsi_programmer.array_curriculum import probes, teachers
from rsi_programmer.array_execution import execute as interpret
from rsi_programmer.array_source import curlee_harness, curlee_source
from rsi_programmer.array_tasks import TASKS, oracle
from rsi_programmer.benchmark_policy import state_fingerprint
from rsi_programmer.program_cost import CostClock
from scripts.check_statement_curlee import passed
from scripts.check_useful_curlee import digest
from scripts.measure_useful_program import run_cli


def _recipe(value: Any) -> dict[str, Any]:
    required = {"execution_context", "seed", "width", "seconds",
                "token_choices"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("recipe must contain the five documented fields")
    if type(value["execution_context"]) is not bool:
        raise ValueError("execution_context must be boolean")
    if type(value["seed"]) is not int or value["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(value["width"]) is not int or value["width"] < 1:
        raise ValueError("width must be positive")
    if type(value["token_choices"]) is not int or value["token_choices"] < 1:
        raise ValueError("token_choices must be positive")
    if (type(value["seconds"]) not in (int, float)
            or value["seconds"] <= 0 or value["seconds"] > 60):
        raise ValueError("seconds must be in (0, 60]")
    return dict(value)


def _check_spec(spec: dict[str, Any], audit: bool) -> None:
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported campaign spec schema")
    expected_phase = "audit" if audit else "development"
    phase = spec.get("phase", spec.get("mode"))
    if phase != expected_phase:
        raise ValueError("spec phase does not match command")
    if type(spec.get("trial_index")) is not int or spec["trial_index"] < 0:
        raise ValueError("trial_index must be a nonnegative integer")
    if not audit and type(spec.get("held_seed")) is not int:
        raise ValueError("held_seed must be an integer")
    if not isinstance(spec.get("source_sha256"), dict):
        raise ValueError("source_sha256 mapping is required")
    for name, expected in spec["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"campaign source changed: {name}")


def _model(recipe: dict[str, Any]) -> tuple[ArrayPolicy, dict[str, Any], str]:
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    import torch
    torch.set_num_threads(1)
    model = ArrayPolicy(seed=recipe["seed"],
                        execution_context=recipe["execution_context"])
    fit = model.fit()
    metadata = {"seed": recipe["seed"],
                "execution_context": recipe["execution_context"]}
    fingerprint = state_fingerprint(model.state_dict(), metadata)
    return model, fit, fingerprint


def _task_descriptors(spec: dict[str, Any], audit: bool
                      ) -> list[dict[str, Any]]:
    if audit:
        descriptors = spec.get("audit_tasks")
        ids = spec.get("audit_task_ids")
        if not isinstance(descriptors, list) or not isinstance(ids, list):
            raise ValueError("audit requires audit_task_ids and audit_tasks")
        if [row.get("task_id") for row in descriptors
                if isinstance(row, dict)] != ids:
            raise ValueError("audit task descriptors must match their IDs")
        if len(set(ids)) != len(ids):
            raise ValueError("audit task IDs must be unique")
        for row in descriptors:
            examples = row.get("examples")
            if (not row.get("family_id") or not isinstance(examples, list)
                    or not examples or "held_inputs" not in row
                    or "expected" not in row):
                raise ValueError("invalid audit task descriptor")
            for pair in examples:
                if (not isinstance(pair, dict)
                        or not isinstance(pair.get("input"), list)
                        or not isinstance(pair.get("output"), list)
                        or len(pair["input"]) > 256
                        or len(pair["output"]) > 512
                        or any(type(v) is not int for v in pair["input"])
                        or any(type(v) is not int for v in pair["output"])):
                    raise ValueError("audit examples need input/output arrays")
        # Check the claimed unseen functions against every curriculum and
        # development function on the curriculum's fixed exclusion probes.
        exclusion_inputs = probes()
        known = [tuple(tuple(interpret(tokens, values))
                       for values in exclusion_inputs)
                 for tokens in teachers()]
        known.extend(tuple(tuple(oracle(name, values))
                           for values in exclusion_inputs)
                     for name in (*TASKS, "duplicate_readings"))
        for row in descriptors:
            by_input = {tuple(item["input"]): item["output"]
                        for item in row["examples"]}
            if all(tuple(values) in by_input for values in exclusion_inputs):
                signature = tuple(tuple(by_input[tuple(values)])
                                  for values in exclusion_inputs)
                if signature in known:
                    row["function_exclusion"] = "known_function"
                else:
                    row["function_exclusion"] = "probe_distinct"
            else:
                row["function_exclusion"] = "not_proven"
        return descriptors
    ids = spec.get("development_task_ids")
    allowed = (*TASKS, "duplicate_readings")
    if not isinstance(ids, list) or not ids or any(
            name not in allowed for name in ids):
        raise ValueError(
            "development_task_ids must name array development tasks")
    if len(set(ids)) != len(ids):
        raise ValueError("development task IDs must be unique")
    return [{"task_id": name, "family_id": "bounded_integer_list",
             "examples": None} for name in ids]


def _make_choices(model: ArrayPolicy, recipe: dict[str, Any],
                  descriptors: list[dict[str, Any]], audit: bool
                  ) -> list[dict[str, Any]]:
    observed = [[-3, 0, 1, 4, -1], [2, -2, 5, 1], [0, 3, -4, 2]]
    choices = []
    for task in descriptors:
        if audit:
            examples = [(item["input"], item["output"])
                        for item in task["examples"]]
        else:
            examples = [(values, oracle(task["task_id"], values))
                        for values in observed]
        start = time.monotonic()
        choice = model.synthesize(
            examples, width=recipe["width"], seconds=recipe["seconds"],
            token_choices=recipe["token_choices"])
        choices.append({"task_id": task["task_id"],
                        "family_id": task["family_id"],
                        "examples": examples, "chosen": choice,
                        "controller_seconds": time.monotonic() - start})
    return choices


def _held(task: dict[str, Any], seed: int, audit: bool
          ) -> tuple[list[list[int]], list[list[int]]]:
    if audit:
        values, expected = task["held_inputs"], task["expected"]
        if (not isinstance(values, list) or not values
                or not isinstance(expected, list)
                or len(values) != len(expected)):
            raise ValueError("audit held inputs and outputs must align")
        if any(not isinstance(row, list) or len(row) > 256
               or any(type(value) is not int for value in row)
               for row in values):
            raise ValueError("audit held rows exceed bounded integer lists")
        if any(not isinstance(row, list) or len(row) > 512
               or any(type(value) is not int for value in row)
               for row in expected):
            raise ValueError("audit expected rows exceed bounded outputs")
        example_inputs = {tuple(pair["input"]) for pair in task["examples"]}
        if any(tuple(row) in example_inputs for row in values):
            raise ValueError("held audit inputs overlap learner examples")
        return values, expected
    rng = random.Random(seed)
    values = [[], [0], [-1], [1], [100] * 256, [-100] * 256] + [
        [rng.randrange(-100, 101) for _ in range(rng.randrange(13, 101))]
        for _ in range(14)]
    return values, [oracle(task["task_id"], row) for row in values]


def _python_check(source: Path, held: list[list[int]],
                  expected: list[list[int]]) -> list[dict[str, Any]]:
    records = []
    with tempfile.TemporaryDirectory(prefix="rsi-campaign-py-") as temp:
        base = Path(temp)
        inputs, outputs = base / "inputs", base / "outputs"
        inputs.mkdir()
        outputs.mkdir()
        for index, (values, target) in enumerate(zip(held, expected)):
            name = f"{index:03}.json"
            (inputs / name).write_text(json.dumps(values) + "\n")
            try:
                proc = run_cli(source.parent, inputs, outputs, name, name)
                actual = json.loads((outputs / name).read_text())
                ok = (proc.returncode == 0 and type(actual) is list
                      and all(type(value) is int for value in actual)
                      and actual == target)
                err = proc.stderr[:2000]
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                ok, actual, err = False, None, str(exc)
                proc = subprocess.CompletedProcess([], 125, "", err)
            records.append({"case": index, "correct": ok,
                            "exit_code": proc.returncode,
                            "actual": actual, "expected": target,
                            "stderr": err})
            (source.parent / "python-records.json").write_text(
                json.dumps(records, indent=2) + "\n")
            if not ok:
                break
    return records


def _execute_task(root: Path, entry: dict[str, Any], task: dict[str, Any],
                  seed: int, compiler: Path, audit: bool
                  ) -> dict[str, Any]:
    tokens = entry["chosen"].get("tokens")
    base = {"task_id": task["task_id"], "family_id": task["family_id"],
            "tokens": tokens, "python_cases": 0, "curlee_cases": 0,
            "python_accepted": False, "curlee_accepted": False,
            "qualified": False}
    if tokens is None:
        return base
    inputs, expected = _held(task, seed, audit)
    task_dir = root / entry["label"] / task["task_id"]
    task_dir.mkdir(parents=True, exist_ok=True)
    py = task_dir / "clean_readings.py"
    cr = task_dir / "process.curlee"
    from rsi_programmer.array_source import python_source
    py.write_text(python_source(tuple(tokens)))
    cr.write_text(curlee_source(tuple(tokens)))
    records = _python_check(py, inputs, expected)
    py_ok = len(records) == len(inputs) and all(
        row["correct"] for row in records)
    base.update(python_cases=len(records), python_accepted=py_ok,
                python_records=records, python_sha256=digest(py),
                curlee_sha256=digest(cr))
    compiler_before = digest(compiler)
    curlee_rows = []
    for index, (values, target) in enumerate(zip(inputs, expected)):
        case = task_dir / f"case-{index:03}.curlee"
        case.write_text(curlee_harness(tuple(tokens), values, target))
        checked = guarded_execute(compiler, case, "check")
        run = (guarded_execute(compiler, case, "run")
               if not checked["exit_code"] else {})
        curlee_rows.append({"case": index, "check": checked, "run": run,
                            "passed": passed(run),
                            "source_sha256": digest(case)})
        (task_dir / "curlee-records.json").write_text(
            json.dumps(curlee_rows, indent=2) + "\n")
        if not curlee_rows[-1]["passed"]:
            break
    compiler_after = digest(compiler)
    cr_ok = (len(curlee_rows) == len(inputs) and all(
        row["passed"] for row in curlee_rows)
             and compiler_before == compiler_after)
    base.update(curlee_cases=len(curlee_rows), curlee_accepted=cr_ok,
                curlee_records=curlee_rows,
                compiler_sha256_before=compiler_before,
                compiler_sha256_after=compiler_after,
                compiler_stable=compiler_before == compiler_after,
                qualified=py_ok and cr_ok)
    if base["qualified"]:
        wrong = [] if expected[0] else [123]
        control = task_dir / "wrong_expected.curlee"
        control.write_text(curlee_harness(tuple(tokens), inputs[0], wrong))
        check = guarded_execute(compiler, control, "check")
        run = guarded_execute(compiler, control, "run") if not check[
            "exit_code"] else {}
        negative_passed = passed(run, 1)
        base["wrong_expected_control"] = {
            "check": check, "run": run,
            "rejected": negative_passed,
            "source_sha256": digest(control)}
        base["qualified"] = base["qualified"] and negative_passed
    return base


def _check_sources(spec: dict[str, Any]) -> dict[str, str]:
    expected = spec["source_sha256"]
    return {str(path): digest(Path(path)) for path in expected}


def guarded_execute(compiler: Path, source: Path, stage: str
                    ) -> dict[str, Any]:
    """Use CURLEE's resource guard for both compilation and VM execution."""
    guard = Path("/home/joe/Projects/curlee/scripts/run_limited.py")
    started = time.monotonic()
    try:
        result = subprocess.run(
            ["/usr/bin/python3", str(guard), str(compiler), stage,
             *( ["--fuel", "20000000"] if stage == "run" else []),
             str(source)], capture_output=True, text=True, timeout=45)
        if result.returncode == 75:
            raise RuntimeError("CURLEE resource guard refused/stopped job")
        return {"exit_code": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr[:8000],
                "wall_seconds": time.monotonic() - started}
    except subprocess.TimeoutExpired as error:
        return {"exit_code": 124, "stdout": error.stdout or "",
                "stderr": "guarded CURLEE command timed out",
                "wall_seconds": time.monotonic() - started}


def run_campaign(spec: dict[str, Any], output: Path, *, audit: bool = False,
                 baseline_recipe: dict[str, Any] | None = None,
                 selected_recipe: dict[str, Any] | None = None
                 ) -> dict[str, Any]:
    """Run choices first, then freeze and independently execute held rows."""
    _check_spec(spec, audit)
    output.mkdir(parents=True, exist_ok=True)
    descriptors = _task_descriptors(spec, audit)
    source_hashes = _check_sources(spec)
    compiler = Path("/home/joe/Projects/curlee/build/linux-debug/curlee")
    guard = Path("/home/joe/Projects/curlee/scripts/run_limited.py")
    if audit:
        baseline_recipe = _recipe(
            baseline_recipe or spec.get("baseline_recipe"))
        selected_recipe = _recipe(
            selected_recipe or spec.get("selected_recipe"))
        recipes = [("baseline", baseline_recipe),
                   ("selected", selected_recipe)]
    else:
        recipes = [("selected", _recipe(spec["recipe"]))]
    runs: list[dict[str, Any]] = []
    clock = CostClock()
    started = time.monotonic()
    pending: list[tuple[str, dict[str, Any], dict[str, Any], str,
                        list[dict[str, Any]], Path]] = []
    project_root = Path(__file__).resolve().parents[1]

    def save_checkpoint(label: str, recipe: dict[str, Any],
                        model: ArrayPolicy) -> Path:
        identity = json.dumps({"phase": spec.get("phase"),
                               "trial_index": spec["trial_index"],
                               "output": str(output.resolve()),
                               "label": label, "recipe": recipe,
                               "source_sha256": spec["source_sha256"]},
                              sort_keys=True, separators=(",", ":"))
        run_hash = hashlib.sha256(identity.encode()).hexdigest()[:24]
        folder = project_root / "weights" / "campaigns" / run_hash
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / f"{label}.pt"
        if checkpoint.exists():
            raise FileExistsError(
                f"campaign checkpoint already exists: {checkpoint}")
        temporary = checkpoint.with_suffix(".pt.tmp")
        import torch
        torch.save(model.state_dict(), temporary)
        temporary.replace(checkpoint)
        return checkpoint

    def build_choices() -> None:
        for label, recipe in recipes:
            model, fit, fingerprint = _model(recipe)
            model_path = save_checkpoint(label, recipe, model)
            choices = _make_choices(model, recipe, descriptors, audit)
            if state_fingerprint(model.state_dict(), {
                    "seed": recipe["seed"],
                    "execution_context": recipe["execution_context"]
            }) != fingerprint:
                raise RuntimeError("model weights changed during synthesis")
            pending.append((label, recipe, fit, fingerprint, choices,
                            model_path))
    if os.environ.get("RSI_CAMPAIGN_LOCK_HELD") == "1":
        build_choices()
    else:
        lock_path = Path(os.environ.get(
            "RSI_BUILDER_LOCK", "/tmp/rsi-programmer-builder.lock"))
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            build_choices()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    # Freeze choices for every condition before consulting any held row.
    frozen_at = time.monotonic()
    (output / "choices-frozen.json").write_text(json.dumps({
        "frozen_at_seconds": frozen_at - started,
        "runs": [{"label": label, "recipe": recipe,
                  "model_fingerprint": fingerprint,
                  "choices": choices}
                 for label, recipe, _, fingerprint, choices, _ in pending]
    }, indent=2) + "\n")
    for label, recipe, fit, fingerprint, choices, model_path in pending:
        executed = [_execute_task(output, item, task,
                                  spec.get("held_seed", 0),
                                  compiler, audit)
                    for item, task in zip(choices, descriptors)]
        runs.append({"label": label, "recipe": recipe, "fit": fit,
                     "model_fingerprint": fingerprint,
                     "checkpoint": str(model_path.relative_to(project_root)),
                     "checkpoint_sha256": digest(model_path),
                     "choices_frozen_at_seconds": frozen_at - started,
                     "per_task": executed,
                     "qualified_tasks": sum(x["qualified"] for x in executed),
                     "task_count": len(executed),
                     "controller_seconds": sum(
                         x["controller_seconds"] for x in choices)})
    elapsed = time.monotonic() - started
    costs = clock.elapsed()
    peak_rss_after = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         resource.getrusage(
                             resource.RUSAGE_CHILDREN).ru_maxrss)
    evidence = {"python": "standalone emitted Python CLI, isolated via bwrap",
                "curlee": "generated source checked and run using CURLEE",
                "curlee_compiler": str(compiler),
                "curlee_guard": str(guard),
                "held_seed": spec.get("held_seed"),
                "held_input_policy": "choices frozen before held rows "
                                     "generated/read"}
    scope = ("Within-family list-array construction from primitive statement "
             "tokens; Python CLI and bounded CURLEE routine are supplied "
             "lowerings, not learner-authored runtime infrastructure.")
    unsupported = ["general source synthesis", "strings", "records",
                   "functions", "libraries", "generalization beyond the "
                   "qualified task family"]
    if not audit:
        selected = runs[0]
        result = {"status": "completed", "phase": "development",
                  "trial_index": spec["trial_index"],
                  "recipe": selected["recipe"],
                  "model_fingerprint": selected["model_fingerprint"],
                  "checkpoint": selected["checkpoint"],
                  "checkpoint_sha256": selected["checkpoint_sha256"],
                  "development": {"qualified_tasks": selected[
                      "qualified_tasks"], "task_count": selected[
                      "task_count"], "failed_task_ids": [
                          row["task_id"] for row in selected["per_task"]
                          if not row["qualified"]],
                      "per_task": selected["per_task"]},
                  "execution_evidence": evidence, "wall_seconds": elapsed,
                  "cost": {**costs, "peak_rss_kib": peak_rss_after,
                           "peak_rss_scope": "worker or any completed child; "
                                             "platform resource accounting"},
                  "supported_scope": scope,
                  "unsupported_capabilities": unsupported,
                  "source_sha256": source_hashes}
    else:
        baseline, selected = runs
        fresh_ids = [task["task_id"] for task in descriptors]
        fresh_families = {task["family_id"] for task in descriptors}
        baseline_ids = set(spec.get("development_task_ids", []))
        # All current emitters and semantics are the same bounded integer-list
        # family. A caller-provided family_id cannot establish a new family.
        cross_family = False
        exclusion = {task["task_id"]: task.get("function_exclusion")
                     for task in descriptors}
        retention = {"task_ids": sorted(baseline_ids),
                     "qualified": None,
                     "status": "not_measured_without_explicit_retention_tasks"}
        selected_gain = selected["qualified_tasks"] > baseline[
            "qualified_tasks"]
        verdict = ("GENERALIZATION_GAIN" if selected_gain and fresh_ids
                   and fresh_families and cross_family
                   and retention["qualified"] else
                   "LOCAL_GAIN_ONLY" if selected_gain else
                   "NO_MEASURED_GAIN")
        result = {"status": "completed", "phase": "audit",
                  "trial_index": spec["trial_index"], "verdict": verdict,
                  "conclusion": ("No cross-family generalization claim; "
                                 "retention evidence is required."),
                  "baseline": baseline, "selected": selected,
                  "fresh_task_ids": fresh_ids,
                  "fresh_family_ids": sorted(fresh_families),
                  "function_exclusion": exclusion,
                  "cross_family_evidence": cross_family,
                  "retention_evidence": retention,
                  "execution_evidence": evidence, "wall_seconds": elapsed,
                  "cost": {**costs, "peak_rss_kib": peak_rss_after,
                           "peak_rss_scope": "worker or any completed child; "
                                             "platform resource accounting"},
                  "supported_scope": scope,
                  "unsupported_capabilities": unsupported,
                  "source_sha256": source_hashes}
    result["guard_script_exists"] = guard.is_file()
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
