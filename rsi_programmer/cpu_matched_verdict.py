"""Equal-total-CPU aggregation and kill verdict for Pursue 2."""

from __future__ import annotations

from typing import Any

KILL_CRITERION = (
    "at equal total CPU the learned ranker solves no more than "
    "streaming/heuristic, or its solve advantage needs more than "
    "2x streaming's time => close line"
)


def _cells(records: list[dict[str, Any]], seed: int, arm: str
           ) -> list[dict[str, Any]]:
    """Return one seed/arm cell sequence in frozen run order."""
    return [row for row in records
            if row["seed"] == seed and row["arm"] == arm]


def _charged(cells: list[dict[str, Any]], fit_cpu: float,
             volume: int) -> list[tuple[bool, float]]:
    """Pair each cell outcome with its amortized charged CPU."""
    if volume < 1:
        raise ValueError("amortization volume must be positive")
    share = fit_cpu / volume
    return [(bool(cell["accepted"]), cell["cpu_seconds"] + share)
            for cell in cells]


def _solves_under(cells: list[tuple[bool, float]],
                  cap: float) -> int:
    """Count accepted cells within a cumulative charged-CPU cap."""
    total = 0.0
    solves = 0
    for accepted, charged in cells:
        total += charged
        if total > cap:
            break
        solves += int(accepted)
    return solves


def _totals(cells: list[tuple[bool, float]]) -> tuple[int, float]:
    """Return full solves and full charged CPU for a cell sequence."""
    solves = sum(1 for accepted, _charged_cpu in cells if accepted)
    total = sum(charged for _accepted, charged in cells)
    return solves, total


def compare_pair(charged_learned: list[list[tuple[bool, float]]],
                 charged_base: list[list[tuple[bool, float]]]
                 ) -> dict[str, Any]:
    """Compare one learned arm against one baseline at equal CPU."""
    truncated: list[tuple[float, float]] = []
    for learned_cells, base_cells in zip(charged_learned,
                                         charged_base):
        _learned_solves, learned_total = _totals(learned_cells)
        _base_solves, base_total = _totals(base_cells)
        cap = min(learned_total, base_total)
        learned_frac = _solves_under(learned_cells, cap)
        base_frac = _solves_under(base_cells, cap)
        count = len(learned_cells)
        truncated.append((learned_frac / count, base_frac / count))
    learned_mean = sum(pair[0] for pair in truncated) / len(truncated)
    base_mean = sum(pair[1] for pair in truncated) / len(truncated)
    return {"learned_truncated_mean": learned_mean,
            "baseline_truncated_mean": base_mean,
            "kill": learned_mean <= base_mean}


def summarize(records: list[dict[str, Any]],
              fit_cpu: dict[str, dict[int, float]],
              protocol: dict[str, Any]) -> dict[str, Any]:
    """Aggregate equal-CPU comparisons and return the line verdict."""
    seeds = sorted({row["seed"] for row in records})
    volume = protocol["cost"]["amortization_volume_per_seed"]
    learned = protocol["decision_rule"]["learned_arms"]
    arms: dict[str, Any] = {}
    for arm in protocol["arms"]:
        per_seed = [_charged(_cells(records, seed, arm),
                             fit_cpu[arm][seed], volume)
                    for seed in seeds]
        totals = [_totals(cells) for cells in per_seed]
        count = len(per_seed[0])
        arms[arm] = {
            "solves_mean": sum(s for s, _t in totals) / len(totals),
            "solves_fraction_mean": sum(
                s / count for s, _t in totals) / len(totals),
            "charged_cpu_mean": sum(t for _s, t in totals)
            / len(totals),
            "fit_cpu_by_seed": {str(seed): fit_cpu[arm][seed]
                                for seed in seeds},
        }
    comparisons = {}
    for name in learned:
        charged_l = [_charged(_cells(records, seed, name),
                               fit_cpu[name][seed], volume)
                     for seed in seeds]
        entry = {}
        for base in ("streaming", "heuristic"):
            charged_b = [_charged(_cells(records, seed, base),
                                   fit_cpu[base][seed], volume)
                         for seed in seeds]
            entry[base] = compare_pair(charged_l, charged_b)
        comparisons[name] = entry
    return {"arms": arms, "comparisons": comparisons,
            "verdict": decide(arms, comparisons)}


def decide(arms: dict[str, Any],
           comparisons: dict[str, Any]) -> dict[str, Any]:
    """Apply the verbatim kill criterion to the aggregated evidence."""
    streaming = arms["streaming"]
    kills = []
    for name, entry in comparisons.items():
        learned = arms[name]
        slow_win = (learned["solves_mean"]
                    > streaming["solves_mean"]
                    and learned["charged_cpu_mean"]
                    > 2.0 * streaming["charged_cpu_mean"])
        kill = (entry["streaming"]["kill"]
                and entry["heuristic"]["kill"]) or slow_win
        kills.append({"learned_arm": name,
                      "kill_vs_streaming": entry["streaming"]["kill"],
                      "kill_vs_heuristic": entry["heuristic"]["kill"],
                      "slow_win_over_2x": slow_win,
                      "close_arm": kill})
    close_line = bool(kills) and all(item["close_arm"]
                                     for item in kills)
    return {"kill_criterion": KILL_CRITERION, "arms": kills,
            "close_line": close_line,
            "line": "CLOSE" if close_line else "CONTINUE"}
