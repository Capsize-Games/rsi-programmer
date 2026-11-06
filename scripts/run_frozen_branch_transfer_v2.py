"""Run one fixed v2 controller without loading the sealed held file."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
from typing import Any

from scripts.branch_transfer_v2_search import Pairs
from scripts.branch_transfer_v2_search import solve
from rsi_programmer.program_cost import CostClock, cpu_seconds
from rsi_programmer.statement_branch_library import qualify_branches
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_fragment_library import qualify
from rsi_programmer.statement_repair import refine
from rsi_programmer.statement_syntax import TOKENS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "fixtures/frozen_branch_transfer_v2"
PREVIOUS = "statement-programs-v1-execution-20261101"
CURRENT = "statement-return-v1-20261101"


def verify_bank(bank: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check the seal without opening the held-input file."""
    hashes = json.loads((bank / "sha256.json").read_text())
    seal = json.loads((bank / "seal.json").read_text())
    if seal["bank_hashes"] != hashes:
        raise ValueError("sealed bank manifest changed")
    for name, expected in seal["solver_sources"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"sealed solver source changed: {name}")
    for name in ("tasks.json", "protocol.json"):
        actual = hashlib.sha256((bank / name).read_bytes()).hexdigest()
        if actual != hashes[name]:
            raise ValueError("frozen bank metadata changed")
    spec = json.loads((bank / "protocol.json").read_text())
    tasks = json.loads((bank / "tasks.json").read_text())
    return tasks, spec


def _report_dir(root: Path, method: str, seed: int) -> Path:
    if method == "classical":
        return root / method
    return root / method / f"seed-{seed}"


def _fit_model(seed: int, label: str, names: list[str],
               variants: bool, report_root: str) -> tuple[Any, dict]:
    from rsi_programmer.statement_policy import StatementPolicy

    clock = CostClock()
    model = StatementPolicy(seed=seed, execution_context=True)
    fit = model.fit(names, epochs=30, stateful=variants,
                    return_variants=variants)
    prior = json.loads((ROOT / "artifacts" / report_root /
                        "report.json").read_text())
    metadata = prior.get("metadata", {
        "seed": seed, "max_tokens": 32,
        "tokens": TOKENS, "execution_context": True})
    return model, {"fit": fit, "prior": prior, "metadata": metadata,
                   "clock": clock, "label": label}


def _checked_record(seed: int, label: str, model: Any,
                    info: dict[str, Any]) -> dict[str, Any]:
    from rsi_programmer.benchmark_policy import state_fingerprint

    fingerprint = state_fingerprint(model.state_dict(),
                                    info["metadata"])
    if seed == 0 and fingerprint != info["prior"]["trained_hash"]:
        raise ValueError("fresh fit does not reproduce frozen model")
    return {"model": label, "fit": info["fit"],
            **info["clock"].elapsed(), "seed": seed,
            "fingerprint": fingerprint, "metadata": info["metadata"]}


def _fresh_models(seed: int) -> tuple[dict[str, Any], list[dict]]:
    import torch

    torch.set_num_threads(1)
    old = json.loads((ROOT / "fixtures/statement_tasks_v1.json"
                      ).read_text())
    new = json.loads((ROOT / "fixtures/stateful_tasks_v1.json"
                      ).read_text())
    previous = [task["id"] for task in old["tasks"]]
    current = previous + [task["id"] for task in new["tasks"]]
    configs = [("previous", previous, False, PREVIOUS),
               ("current", current, True, CURRENT)]
    models, records = {}, []
    for label, names, variants, report_root in configs:
        model, info = _fit_model(seed, label, names, variants,
                                 report_root)
        models[label] = model
        records.append(_checked_record(seed, label, model, info))
    return models, records


def _untrained_model(seed: int) -> tuple[dict[str, Any], str]:
    import torch
    from rsi_programmer.benchmark_policy import state_fingerprint
    from rsi_programmer.statement_policy import StatementPolicy

    torch.set_num_threads(1)
    model = StatementPolicy(seed=seed, execution_context=True)
    model.eval()
    fingerprint = state_fingerprint(model.state_dict(), {"seed": seed})
    return {"current": model}, fingerprint


def _qualify(method: str) -> tuple[Any, Any, list[Any], dict[str, float]]:
    setup = CostClock()
    old = (qualify(ROOT / "artifacts/statement-programs-v1"
                   "-execution-20261101") if method == "neural" else None)
    new = qualify(ROOT / "artifacts/statement-return-v1-20261101")
    base = ROOT / "artifacts/statement-context-repair-v1-20261101"
    branches = qualify_branches(base)["branches"]
    branches += qualify_branches(
        ROOT / "artifacts/positive-counter-v1-20261101")["branches"]
    return old, new, branches, setup.elapsed()


def _finish(task_id: str, result: dict[str, Any],
            cost: dict[str, float], spec: dict[str, Any]
            ) -> dict[str, Any]:
    over = cost["cpu_seconds"] > spec["request_cpu_budget_seconds"] + (
        spec["budget_overrun_tolerance_seconds"])
    chosen = result["chosen"]
    if over:
        chosen = {"tokens": None, "budget_exhausted": True,
                  "over_budget_candidate": chosen}
    print(json.dumps({"id": task_id, "emitted": chosen["tokens"]
                      is not None, "cpu_seconds": cost["cpu_seconds"]}),
          flush=True)
    return {"id": task_id, "chosen": chosen,
            "observed": result["observed"], "repair": result,
            "request_cost": cost, "over_budget": over}


def _run_task(task: dict[str, Any], queries: list[list[int]],
              method: str, models: dict[str, Any], old: Any, new: Any,
              branches: list[Any], spec: dict[str, Any]
              ) -> dict[str, Any]:
    clock = CostClock()

    def remaining() -> float:
        return max(0.0, spec["request_cpu_budget_seconds"]
                   - clock.elapsed()["cpu_seconds"])

    def attempt(examples: Pairs) -> dict[str, Any]:
        return solve(method, models, old, new, branches, examples,
                     remaining)

    reference = tuple(task["reference"])
    result = refine(attempt, task["observed"], queries,
                    lambda values: execute(reference, values),
                    rounds=spec["max_added_examples"])
    return _finish(task["id"], result, clock.elapsed(), spec)


def _check_frozen(models: dict[str, Any], fits: list[dict[str, Any]],
                  untrained_hash: str | None, seed: int) -> None:
    from rsi_programmer.benchmark_policy import state_fingerprint

    for fit in fits:
        actual = state_fingerprint(models[fit["model"]].state_dict(),
                                   fit["metadata"])
        if actual != fit["fingerprint"]:
            raise ValueError("evaluation changed trained parameters")
    if untrained_hash is not None and state_fingerprint(
            models["current"].state_dict(),
            {"seed": seed}) != untrained_hash:
        raise ValueError("evaluation changed untrained parameters")


def _setup_arm(method: str, seed: int) -> tuple[dict[str, Any], ...]:
    models: dict[str, Any] = {}
    fits: list[dict[str, Any]] = []
    untrained_hash = None
    if method == "neural":
        models, fits = _fresh_models(seed)
    elif method == "untrained":
        models, untrained_hash = _untrained_model(seed)
    old, new, branches, qualification_cost = _qualify(method)
    return models, fits, untrained_hash, old, new, branches, (
        qualification_cost)


def _write_report(folder: Path, bank: Path, method: str, seed: int,
                  records: list[dict[str, Any]], start: CostClock,
                  extra: dict[str, Any]) -> None:
    report = {"scope": "Frozen unseen branch-function evaluation v2. "
                       "No held inputs loaded; choices now final.",
              "method": method, "seed": seed,
              "tasks": len(records), "rows": records,
              "bank_hashes": json.loads((bank / "sha256.json"
                                         ).read_text()),
              "worker_cost": start.elapsed(),
              "process_cpu_seconds": cpu_seconds(),
              "peak_worker_rss_kib": resource.getrusage(
                  resource.RUSAGE_SELF).ru_maxrss,
              "runtime": {"python": sys.version.split()[0],
                          "cpu_affinity": sorted(os.sched_getaffinity(0))},
              "execution_pending": True, **extra}
    (folder / "report.json").write_text(json.dumps(report, indent=2)
                                        + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True,
                        choices=("neural", "classical", "untrained"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    start = CostClock()
    bank = args.bank.resolve()
    tasks, spec = verify_bank(bank)
    folder = _report_dir(args.output.resolve(), args.method, args.seed)
    folder.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    setup = _setup_arm(args.method, args.seed)
    models, fits, untrained_hash, old, new, branches, qual_cost = setup
    records = [_run_task(task, tasks["query_inputs"], args.method,
                         models, old, new, branches, spec)
               for task in tasks["tasks"]]
    _check_frozen(models, fits, untrained_hash, args.seed)
    _write_report(folder, bank, args.method, args.seed, records,
                  start, {"qualification_cost": qual_cost,
                          "fresh_fits": fits,
                          "untrained_fingerprint": untrained_hash})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
