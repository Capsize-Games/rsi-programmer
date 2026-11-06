"""Calibration: time-to-solve on TRAIN pool only, per stratum.

One pass per arm records process-CPU seconds to solve (or cap);
solve fractions at candidate budgets are derived from those
times. Eval/dev rows are never loaded for solving: only the
train split is read. Output: calibration.json beside this file.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rsi_programmer.construction_beam import beam_construct  # noqa: E402
from rsi_programmer.construction_check import check_graph  # noqa: E402
from rsi_programmer.construction_policy import (  # noqa: E402
    ConstructionPolicy,
)
from rsi_programmer.construction_training import (  # noqa: E402
    fit_constructor,
)
from rsi_programmer.cpu_budget import CpuBudget, CpuBudgetExceeded  # noqa: E402
from rsi_programmer.pre_eval import pre_evaluation_search  # noqa: E402

CAP = 10.0
PER_STRATUM = 40
BOUND = 7
BACKSTOP = 2_000_000


def rows_of(task: dict[str, Any]) -> Any:
    return tuple(tuple(row) for row in task["observed"])


def poly_of(task: dict[str, Any]) -> Any:
    return tuple(tuple(t) for t in task["generator"]["polynomial"])


def timed(fn: Any) -> tuple[Any, float, bool]:
    started = time.process_time()
    try:
        with CpuBudget(CAP):
            return fn(), time.process_time() - started, False
    except CpuBudgetExceeded:
        return None, time.process_time() - started, True


def accepted(task: dict[str, Any], graph: Any) -> bool:
    check = check_graph(graph, rows_of(task),
                        tuple(tuple(r) for r in task["held"]),
                        poly_of(task), BOUND)
    return bool(check["accepted"])


def measure(tasks: list[dict[str, Any]], arm: str,
            policy: Any) -> list[dict[str, Any]]:
    out = []
    for task in tasks:
        observed = rows_of(task)
        if arm == "streaming":
            def search() -> Any:
                return pre_evaluation_search(
                    None, tuple((x, y) for x, y, _ in observed),
                    tuple(w for _, _, w in observed), BOUND,
                    BACKSTOP)["graph"]
        else:
            def search() -> Any:
                return beam_construct(
                    policy, observed, bound=BOUND,
                    budget=BACKSTOP, width=8, seed=0)["graph"]
        graph, cpu, capped = timed(search)
        out.append({"id": task["id"],
                    "stratum": task["generator"]["sampled_ops"],
                    "solved": accepted(task, graph),
                    "cpu_seconds": cpu, "capped": capped})
    return out


def fractions(rows: list[dict[str, Any]],
              budgets: list[float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum in (5, 6, 7, "all"):
        part = [r for r in rows
                if stratum == "all" or r["stratum"] == stratum]
        result[str(stratum)] = {
            str(b): sum(1 for r in part
                        if r["solved"] and r["cpu_seconds"] <= b)
            / len(part) for b in budgets}
    return result


def main() -> int:
    corpus = json.loads(
        (ROOT / "fixtures/construction_hard_corpus_v1.json").read_text())
    train = corpus["splits"]["train"]
    sample: list[dict[str, Any]] = []
    for stratum in (5, 6, 7):
        part = sorted(
            (t for t in train
             if t["generator"]["sampled_ops"] == stratum),
            key=lambda t: t["id"])[:PER_STRATUM]
        sample.extend(part)
    budgets = [0.5, 1.0, 2.0, 5.0]
    record: dict[str, Any] = {"sample_per_stratum": PER_STRATUM,
                              "cap_seconds": CAP, "arms": {}}
    print("streaming...", flush=True)
    rows = measure(sample, "streaming", None)
    record["arms"]["streaming"] = {"rows": rows,
                                    "fractions": fractions(rows,
                                                           budgets)}
    print("untrained...", flush=True)
    rows = measure(sample, "beam", ConstructionPolicy(seed=0))
    record["arms"]["untrained"] = {"rows": rows,
                                    "fractions": fractions(rows,
                                                           budgets)}
    print("imitation fit (100 epochs, full train)...", flush=True)
    policy = ConstructionPolicy(seed=0)
    fit = fit_constructor(policy, train, 100)
    record["imitation_fit"] = {
        k: fit[k] for k in ("complete_cpu_seconds", "final_loss",
                            "final_hash", "teacher_steps")}
    print("imitation measure...", flush=True)
    rows = measure(sample, "beam", policy)
    record["arms"]["imitation"] = {"rows": rows,
                                    "fractions": fractions(rows,
                                                           budgets)}
    out = Path(__file__).with_name("calibration.json")
    out.write_text(json.dumps(record, indent=1) + "\n")
    for arm, data in record["arms"].items():
        print(arm, json.dumps(data["fractions"]["all"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
