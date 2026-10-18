"""Cost curves for the predeclared complete-fallback shortlist experiment.
"""

from __future__ import annotations

from typing import Any


def configurations(
    protocol: dict[str, Any],
) -> list[tuple[str, str, list[int]]]:
    """Return the asymmetric engine/control/cap grid."""
    ev = protocol["evaluation"]
    return [("streaming", "streaming", ev["streaming_budgets"]),
            ("cheap_only", "cheap_only", ev["ranked_budgets"])] + [
        (engine, control, ev["ranked_budgets"])
        for control in ev["ranked_controls"]
        for engine in ev["ranked_engines"]
    ]


def training_charge(
    training: dict[str, Any], control: str, axis: str
) -> float:
    """Charge each learned standalone system its full supervision setup."""
    if control not in {"trained_snn", "mlp", "recurrent"}:
        return 0.0
    seeds = training["seeds"]
    own = sum(
        s["control_construction"][control][axis + "_ns"]
        + s["controls"][control]["fit_" + axis + "_ns"]
        for s in seeds
    ) / len(seeds)
    return training["training_group_" + axis + "_ns"] + own


def _work_total(rows: list[dict[str, Any]], key: str) -> int | None:
    """Sum a work axis only when every attempt measured it."""
    values = [row["work"].get(key) for row in rows]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def summarize(
    records: list[dict[str, Any]],
    training: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Keep success, CPU/wall, unrelated work axes and training separate."""
    points: list[dict[str, Any]] = []
    amort: list[dict[str, Any]] = []
    matched: list[dict[str, Any]] = []
    eval_task_count = protocol["evaluation"]["task_count"]
    seed_count = len(protocol["training"]["seeds"])
    for engine, control, budgets in configurations(protocol):
        production_cpu_charge = training_charge(training, control, "cpu")
        selection_cpu = (training["selection"]["total_cpu_ns"]
                         if engine in {"complete_shortlist", "cheap_only"}
                         else 0)
        total_cpu_charge = production_cpu_charge + selection_cpu
        production_wall_charge = training_charge(training, control, "wall")
        selection_wall = (training["selection"]["total_wall_ns"]
                          if engine in {"complete_shortlist", "cheap_only"}
                          else 0)
        total_wall_charge = production_wall_charge + selection_wall
        for repeat in (0, 1, None):
            curve = []
            for budget in budgets:
                rows = [
                    r
                    for r in records
                    if r["engine"] == engine
                    and r["control"] == control
                    and r["budget"] == budget
                    and (repeat is None or r["repeat"] == repeat)
                ]
                if not rows:
                    continue
                work = {
                    key: _work_total(rows, key)
                    for key in rows[0]["work"]
                }
                cpu = sum(
                    r["attempt_cpu_ns_including_hashes_and_validation"]
                    for r in rows
                ) / len(rows)
                wall = sum(
                    r["attempt_wall_ns_including_hashes_and_validation"]
                    for r in rows
                ) / len(rows)
                expected_attempts = seed_count * eval_task_count * (
                    2 if repeat is None else 1)
                unique_attempts = {
                    (r["seed"], r["task_id"], r["repeat"])
                    for r in rows
                }
                complete_cell = (
                    len(rows) == expected_attempts
                    and len(unique_attempts) == expected_attempts)
                censored = sum(r["censored"] for r in rows)
                point = {
                    "engine": engine,
                    "control": control,
                    "budget": budget,
                    "repeat": repeat,
                    "attempts": len(rows),
                    "accepted": sum(r["accepted"] for r in rows),
                    "success_fraction": sum(r["accepted"] for r in rows)
                    / len(rows),
                    "censored_attempts": censored,
                    "complete_matrix_cell": complete_cell,
                    "attempt_cpu_ns_mean": cpu,
                    "attempt_wall_ns_mean": wall,
                    "work_totals_by_axis": work,
                    "scan_amplification": (
                        work["scored_descriptors"]
                        / work["signature_evaluations"]
                        if work["signature_evaluations"]
                        else None
                    ),
                }
                points.append(point)
                curve.append(point)
                if repeat is None:
                    for count in protocol["amortization"]["task_counts"]:
                        amort.append(
                            {
                                "engine": engine,
                                "control": control,
                                "budget": budget,
                                "independent_task_count": count,
                                "mean_attempt_cpu_ns": cpu,
                                "mean_attempt_wall_ns": wall,
                                ("training_cpu_ns_mean_seed_"
                                 "including_full_setup"):
                                    production_cpu_charge,
                                ("training_wall_ns_mean_seed_"
                                 "including_full_setup"):
                                    production_wall_charge,
                                "selection_cpu_ns": selection_cpu,
                                "selection_wall_ns": selection_wall,
                                "total_upfront_cpu_ns": total_cpu_charge,
                                "total_upfront_wall_ns": total_wall_charge,
                                "amortized_cpu_ns_per_task": cpu
                                + total_cpu_charge / count,
                                "amortized_wall_ns_per_task": wall
                                + total_wall_charge / count,
                            }
                        )
            for target in protocol["aggregation"]["matched_success_targets"]:
                point = next(
                    (p for p in curve
                     if p["complete_matrix_cell"]
                     and p["success_fraction"] >= target
                     and (target < 1.0
                          or (p["censored_attempts"] == 0
                              and p["accepted"] == p["attempts"]))),
                    None,
                )
                matched.append(
                    {
                        "engine": engine,
                        "control": control,
                        "repeat": repeat,
                        "target": target,
                        "status": "observed" if point else "censored",
                        "budget": point["budget"] if point else None,
                        "censored_at_budget": None if point else max(budgets),
                        "success_fraction": point["success_fraction"]
                        if point
                        else None,
                        "overshoot": point["success_fraction"] - target
                        if point
                        else None,
                        "mean_attempt_cpu_ns": point["attempt_cpu_ns_mean"]
                        if point
                        else None,
                        "mean_attempt_wall_ns": point["attempt_wall_ns_mean"]
                        if point
                        else None,
                        "work_totals_by_axis": point["work_totals_by_axis"]
                        if point
                        else None,
                        "training_amortized_cpu_ns_per_task_by_N": {
                            str(n): point["attempt_cpu_ns_mean"]
                            + total_cpu_charge / n
                            for n in protocol["amortization"]["task_counts"]
                        }
                        if point
                        else {},
                        "training_amortized_wall_ns_per_task_by_N": {
                            str(n): point["attempt_wall_ns_mean"]
                            + total_wall_charge / n
                            for n in protocol["amortization"]["task_counts"]
                        }
                        if point
                        else {},
                    }
                )
    ratios = []
    for control in protocol["evaluation"]["ranked_controls"]:
        for budget in protocol["evaluation"]["ranked_budgets"]:
            for repeat in (0, 1, None):
                pair = {
                    p["engine"]: p
                    for p in points
                    if p["control"] == control
                    and p["budget"] == budget
                    and p["repeat"] == repeat
                }
                if set(pair) != {"full_cached", "complete_shortlist"}:
                    continue
                row: dict[str, Any] = {
                    "control": control,
                    "budget": budget,
                    "repeat": repeat,
                }
                for axis in ("cpu", "wall"):
                    key = "attempt_" + axis + "_ns_mean"
                    for engine in pair:
                        row[engine + "_" + axis + "_ns"] = (
                            pair[engine][key] * pair[engine]["attempts"]
                        )
                    ratio_key = (
                        "complete_shortlist_over_full_cached_"
                        + axis
                        + "_ratio"
                    )
                    row[ratio_key] = (
                        pair["complete_shortlist"][key]
                        / pair["full_cached"][key]
                    )
                ratios.append(row)
    full_success = {
        (p["engine"], p["control"]): p
        for p in matched
        if p["repeat"] is None and p["target"] == 1.0
    }
    streaming_point = full_success.get(("streaming", "streaming"))
    streaming = (streaming_point[
        "training_amortized_cpu_ns_per_task_by_N"].get("48")
        if streaming_point is not None else None)
    decisions = []
    for engine, control, _ in configurations(protocol):
        if engine not in {"complete_shortlist", "cheap_only"}:
            continue
        success_point = full_success.get((engine, control))
        value = (success_point[
            "training_amortized_cpu_ns_per_task_by_N"].get("48")
            if success_point is not None else None)
        ratio = (value / streaming
                 if value is not None and streaming is not None else None)
        decisions.append({
            "engine": engine, "control": control,
            "first_full_success_N48_cpu_ns": value,
            "streaming_first_full_success_N48_cpu_ns": streaming,
            "over_streaming_cpu_ratio": ratio,
            "supported": ratio is not None and ratio <= 0.9,
        })
    all_complete = all(d["first_full_success_N48_cpu_ns"] is not None
                       for d in decisions)
    snn_point = full_success.get(("complete_shortlist", "trained_snn"))
    snn = (snn_point["training_amortized_cpu_ns_per_task_by_N"].get("48")
           if snn_point is not None else None)
    competing = [
        (full_success[engine, control][
            "training_amortized_cpu_ns_per_task_by_N"].get("48")
         if (engine, control) in full_success else None)
        for engine, control in (
            ("complete_shortlist", "mlp"),
            ("complete_shortlist", "recurrent"),
            ("complete_shortlist", "heuristic"),
            ("cheap_only", "cheap_only"),
            ("streaming", "streaming"),
        )
    ]
    snn_supported = snn is not None and all(
        value is not None and snn <= 0.9 * value for value in competing
    )
    return {
        "points": points,
        "matched_success_cpu_cost": matched,
        "amortization": amort,
        "paired_cost_ratios": ratios,
        "decision_controls": decisions,
        "all_new_controls_full_success": all_complete,
        "decision_supported_on_cost": all_complete and any(
            d["supported"] for d in decisions),
        "snn_efficiency_supported_on_cost": snn_supported,
    }
