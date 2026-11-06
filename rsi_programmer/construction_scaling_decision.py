"""Equal-CPU truncation verdict for the sealed scaling run."""
from __future__ import annotations

from typing import Any

SeedRows = dict[int, list[dict[str, Any]]]


def credited_solves(rows: list[dict[str, Any]], fit_cpu: float,
                    budget: float) -> int:
    """Credit accepted rows while cumulative CPU stays in budget."""
    total, credited = float(fit_cpu), 0
    for row in rows:
        total += float(row["cpu_seconds"])
        if total > budget:
            break
        credited += 1 if row["accepted"] else 0
    return credited


def decide_scaling(learned: float, streaming: float,
                   untrained: float, common: float,
                   kill: str) -> dict[str, Any]:
    """Pass only when learned beats streaming and untrained at B*."""
    passed = learned > streaming and learned > untrained
    return {"passed": passed, "learned_credited_mean": learned,
            "streaming_credited": streaming,
            "untrained_credited_mean": untrained,
            "common_budget_cpu_seconds": common,
            "kill_criterion": kill}


def _mean_strata(seeds: SeedRows) -> dict[str, float]:
    """Average per-stratum solves across seeds for one arm."""
    totals: dict[str, float] = {}
    for rows in seeds.values():
        for row in rows:
            if row["accepted"]:
                key = str(row["stratum"])
                totals[key] = totals.get(key, 0.0) + 1
    return {key: value / len(seeds) for key, value in totals.items()}


def _credited_means(learned: SeedRows, untrained: SeedRows,
                    streaming: list[dict[str, Any]],
                    charges: dict[int, float]) -> tuple[Any, ...]:
    """Credit every arm at B*; also report the common budget."""
    totals = [sum(row["cpu_seconds"] for row in rows) + charges[seed]
              for seed, rows in learned.items()]
    common = min(totals + [sum(row["cpu_seconds"] for row in streaming)])
    hits = [credited_solves(rows, charges[seed], common)
            for seed, rows in learned.items()]
    misses = [credited_solves(rows, 0.0, common)
              for rows in untrained.values()]
    credited = {"learned": sum(hits) / len(hits), "streaming":
                float(credited_solves(streaming, 0.0, common)),
                "untrained": sum(misses) / len(misses)}
    return credited, common


def summarize_scaling(learned: SeedRows, untrained: SeedRows,
                      streaming: list[dict[str, Any]],
                      charges: dict[int, float], kill: str
                      ) -> dict[str, Any]:
    """Truncate every arm at the common budget; render the verdict."""
    credited, common = _credited_means(learned, untrained,
                                       streaming, charges)
    decision = decide_scaling(credited["learned"],
                              credited["streaming"],
                              credited["untrained"], common, kill)
    means = {name: sum(1 for rows in arms.values() for row in rows
                       if row["accepted"]) / len(arms)
             for name, arms in (("learned", learned),
                                ("untrained", untrained))}
    means["streaming"] = sum(1 for row in streaming if row["accepted"])
    return {"means": means, "decision": decision, "strata": {
        "learned": _mean_strata(learned),
        "untrained": _mean_strata(untrained),
        "streaming": _mean_strata({0: streaming})}}
