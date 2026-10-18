"""Cost curves for the predeclared bounded shortlist experiment."""

from __future__ import annotations

from typing import Any


def configurations(
    protocol: dict[str, Any],
) -> list[tuple[str, str, list[int]]]:
    """Return the asymmetric engine/control/cap grid."""
    ev = protocol["evaluation"]
    return [("streaming", "streaming", ev["streaming_budgets"])] + [
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


def summarize(
    records: list[dict[str, Any]],
    training: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Keep success, CPU/wall, unrelated work axes and training separate."""
    points: list[dict[str, Any]] = []
    amort: list[dict[str, Any]] = []
    matched: list[dict[str, Any]] = []
    for engine, control, budgets in configurations(protocol):
        cpu_charge = training_charge(training, control, "cpu")
        wall_charge = training_charge(training, control, "wall")
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
                    k: sum(r["work"].get(k) or 0 for r in rows)
                    for k in rows[0]["work"]
                }
                cpu = sum(
                    r["attempt_cpu_ns_including_hashes_and_validation"]
                    for r in rows
                ) / len(rows)
                wall = sum(
                    r["attempt_wall_ns_including_hashes_and_validation"]
                    for r in rows
                ) / len(rows)
                point = {
                    "engine": engine,
                    "control": control,
                    "budget": budget,
                    "repeat": repeat,
                    "attempts": len(rows),
                    "accepted": sum(r["accepted"] for r in rows),
                    "success_fraction": sum(r["accepted"] for r in rows)
                    / len(rows),
                    "censored_attempts": sum(r["censored"] for r in rows),
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
                                 "including_full_setup"): cpu_charge,
                                ("training_wall_ns_mean_seed_"
                                 "including_full_setup"): wall_charge,
                                "amortized_cpu_ns_per_task": cpu
                                + cpu_charge / count,
                                "amortized_wall_ns_per_task": wall
                                + wall_charge / count,
                            }
                        )
            for target in protocol["aggregation"]["matched_success_targets"]:
                point = next(
                    (p for p in curve if p["success_fraction"] >= target), None
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
                            + cpu_charge / n
                            for n in protocol["amortization"]["task_counts"]
                        }
                        if point
                        else {},
                        "training_amortized_wall_ns_per_task_by_N": {
                            str(n): point["attempt_wall_ns_mean"]
                            + wall_charge / n
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
                if set(pair) != {"full_cached", "bounded"}:
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
                    row["bounded_over_full_cached_" + axis + "_ratio"] = (
                        pair["bounded"][key] / pair["full_cached"][key]
                    )
                ratios.append(row)
    full_success = {
        (p["engine"], p["control"]): p
        for p in matched
        if p["repeat"] is None and p["target"] == 1.0
    }
    decisions = []
    for control in protocol["decision_rule"]["required_controls"]:
        values = {}
        for engine in ("full_cached", "bounded"):
            point = full_success[engine, control]
            values[engine] = point[
                "training_amortized_cpu_ns_per_task_by_N"
            ].get("48")
        ratio = (
            values["bounded"] / values["full_cached"]
            if all(v is not None for v in values.values())
            else None
        )
        decisions.append(
            {
                "control": control,
                "first_full_success_N48_cpu_ns": values,
                "bounded_over_full_cached_cpu_ratio": ratio,
                "supported": ratio is not None and ratio <= 0.8,
            }
        )
    snn = full_success["bounded", "trained_snn"][
        "training_amortized_cpu_ns_per_task_by_N"
    ].get("48")
    competing = [
        full_success[engine, control][
            "training_amortized_cpu_ns_per_task_by_N"
        ].get("48")
        for engine, control in (
            ("bounded", "mlp"),
            ("bounded", "recurrent"),
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
        "decision_supported_on_cost": all(d["supported"] for d in decisions),
        "snn_efficiency_supported_on_cost": snn_supported,
    }
