"""Run one fixed controller without loading the sealed held-input file."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
from typing import Any

from rsi_programmer.program_cost import CostClock, cpu_seconds
from rsi_programmer.statement_branch_library import qualify_branches
from rsi_programmer.statement_branch_search import synthesize as compose
from rsi_programmer.statement_execution import execute
from rsi_programmer.statement_fragment_classical import synthesize as classical
from rsi_programmer.statement_fragment_library import qualify
from rsi_programmer.statement_repair import refine
from rsi_programmer.statement_syntax import TOKENS

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "fixtures/frozen_branch_transfer_v1"
Pairs = list[tuple[list[int], int]]


def _fresh_models() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from rsi_programmer.benchmark_policy import state_fingerprint
    from rsi_programmer.statement_policy import StatementPolicy

    torch.set_num_threads(1)
    old = json.loads((ROOT / "fixtures/statement_tasks_v1.json").read_text())
    new = json.loads((ROOT / "fixtures/stateful_tasks_v1.json").read_text())
    previous = [task["id"] for task in old["tasks"]]
    all_names = previous + [task["id"] for task in new["tasks"]]
    models, records = {}, []
    for label, names, variants, report_root in (
            ("previous", previous, False,
             "statement-programs-v1-execution-20261101"),
            ("current", all_names, True, "statement-return-v1-20261101")):
        clock = CostClock()
        model = StatementPolicy(seed=0, execution_context=True)
        fit = model.fit(names, epochs=30, stateful=variants,
                        return_variants=variants)
        prior = json.loads((ROOT / "artifacts" / report_root /
                            "report.json").read_text())
        metadata = prior.get("metadata", {
            "seed": 0, "max_tokens": 32,
            "tokens": TOKENS,
            "execution_context": True})
        fingerprint = state_fingerprint(model.state_dict(), metadata)
        if fingerprint != prior["trained_hash"]:
            raise ValueError("fresh fit does not reproduce the frozen model")
        models[label] = model
        records.append({"model": label, "fit": fit, **clock.elapsed(),
                        "fingerprint": fingerprint, "metadata": metadata})
    return models, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True,
                        choices=("neural", "classical", "untrained"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = CostClock()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    hashes = json.loads((BANK / "sha256.json").read_text())
    seal = json.loads((BANK / "seal.json").read_text())
    if seal["bank_hashes"] != hashes:
        raise ValueError("sealed bank manifest changed")
    for name, expected in seal["solver_sources"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("sealed solver source changed")
    # Deliberately do not open held.json in this process.
    for name in ("tasks.json", "protocol.json"):
        if hashlib.sha256((BANK / name).read_bytes()).hexdigest() != (
                hashes[name]):
            raise ValueError("frozen bank metadata changed")
    bank = json.loads((BANK / "tasks.json").read_text())
    spec = json.loads((BANK / "protocol.json").read_text())
    setup = CostClock()
    old = (qualify(
        ROOT / "artifacts/statement-programs-v1-execution-20261101")
        if args.method == "neural" else None)
    new = qualify(ROOT / "artifacts/statement-return-v1-20261101")
    base = ROOT / "artifacts/statement-context-repair-v1-20261101"
    branches = qualify_branches(base)["branches"]
    branches += qualify_branches(
        ROOT / "artifacts/positive-counter-v1-20261101")["branches"]
    qualification_cost = setup.elapsed()
    models, fits = {}, []
    untrained_hash = None
    if args.method == "neural":
        models, fits = _fresh_models()
    elif args.method == "untrained":
        import torch
        from rsi_programmer.statement_policy import StatementPolicy
        from rsi_programmer.benchmark_policy import state_fingerprint
        torch.set_num_threads(1)
        models["current"] = StatementPolicy(seed=0, execution_context=True)
        models["current"].eval()
        untrained_hash = state_fingerprint(models["current"].state_dict(),
                                           {"seed": 0})
    records = []
    for task in bank["tasks"]:
        clock = CostClock()

        def remaining() -> float:
            return max(0.0, spec["request_cpu_budget_seconds"]
                       - clock.elapsed()["cpu_seconds"])

        def solve(examples: Pairs) -> dict[str, Any]:
            budget = remaining()
            if budget <= 0:
                return {"tokens": None, "budget_exhausted": True}
            if args.method == "classical":
                first = compose(None, examples, branches,
                                budget_seconds=budget)
                if first["tokens"] is not None or remaining() <= 0:
                    return first
                budget = remaining()
                return (classical(examples, new["fragments"], budget)
                        if budget > 0 else {"tokens": None})
            from rsi_programmer.statement_fragment_search import synthesize
            route = []
            if args.method == "neural":
                first = synthesize(models["previous"], examples,
                                   old["fragments"],
                                   budget_seconds=min(1.0, budget))
                route.append(first)
                if first["tokens"] is not None or remaining() <= 0:
                    return {**first, "route": route}
            budget = remaining()
            if budget <= 0:
                return {"tokens": None, "route": route}
            second = synthesize(models["current"], examples, new["fragments"],
                                budget_seconds=min(1.0, budget))
            route.append(second)
            if second["tokens"] is not None or remaining() <= 0:
                return {**second, "route": route}
            budget = remaining()
            if budget <= 0:
                return {"tokens": None, "route": route}
            third = compose(models["current"], examples, branches,
                            budget_seconds=budget)
            return {**third, "route": [*route, third]}

        reference = tuple(task["reference"])
        result = refine(solve, task["observed"], bank["query_inputs"],
                        lambda values: execute(reference, values),
                        rounds=spec["max_added_examples"])
        cost = clock.elapsed()
        over = cost["cpu_seconds"] > spec["request_cpu_budget_seconds"] + (
            spec["budget_overrun_tolerance_seconds"])
        chosen = result["chosen"]
        if over:
            chosen = {"tokens": None, "budget_exhausted": True,
                      "over_budget_candidate": chosen}
        records.append({"id": task["id"], "chosen": chosen,
                        "observed": result["observed"], "repair": result,
                        "request_cost": cost, "over_budget": over})
        print(json.dumps({"id": task["id"], "emitted": chosen["tokens"]
                          is not None, "cpu_seconds": cost["cpu_seconds"]}),
              flush=True)
    if args.method != "classical":
        from rsi_programmer.benchmark_policy import state_fingerprint
        for fit in fits:
            actual = state_fingerprint(models[fit["model"]].state_dict(),
                                       fit["metadata"])
            if actual != fit["fingerprint"]:
                raise ValueError("evaluation changed trained parameters")
        if untrained_hash is not None and state_fingerprint(
                models["current"].state_dict(), {"seed": 0}) != untrained_hash:
            raise ValueError("evaluation changed untrained parameters")
    report = {"scope": "Frozen unseen branch-function evaluation. "
                       "No held inputs loaded; choices now final.",
              "method": args.method, "tasks": len(records), "rows": records,
              "bank_hashes": hashes, "qualification_cost": qualification_cost,
              "fresh_fits": fits, "worker_cost": start.elapsed(),
              "untrained_fingerprint": untrained_hash,
              "process_cpu_seconds": cpu_seconds(),
              "peak_worker_rss_kib": resource.getrusage(
                  resource.RUSAGE_SELF).ru_maxrss,
              "runtime": {"python": sys.version.split()[0],
                          "cpu_affinity": sorted(os.sched_getaffinity(0))},
              "execution_pending": True}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
