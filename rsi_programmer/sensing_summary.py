"""Summaries and prospective gates for the first sensing comparison."""
from __future__ import annotations

from typing import Any


ARMS = ("plastic_snn", "readout_snn", "frozen_snn", "plastic_ann",
        "persistence", "ema", "nlms")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep positive recall and sham false alarms on distinct denominators."""
    positive = [row for row in rows if row["kind"] == "positive"]
    shams = [row for row in rows if row["kind"] == "sham"]
    events = sum(row["metrics"]["events"] for row in positive)
    detected = sum(row["metrics"]["detected"] for row in positive)
    sham_fp = sum(row["metrics"]["false_alarms"] for row in shams)
    sham_seconds = sum(row["metrics"]["stable_seconds"] for row in shams)
    targets = sum(row["metrics"]["scored_windows"] for row in rows)
    return {
        "streams": len(rows), "events": events, "detected": detected,
        "recall": detected / events if events else None,
        "sham_false_alarms": sham_fp,
        "sham_stable_seconds": sham_seconds,
        "sham_false_alarms_per_minute": (
            sham_fp * 60.0 / sham_seconds if sham_seconds else None),
        "positive_false_alarms": sum(
            row["metrics"]["false_alarms"] for row in positive),
        "delay_seconds": [delay for row in positive
                          for delay in row["metrics"]["delay_seconds"]],
        "mse": sum(row["metrics"]["squared_error_sum"] for row in rows)
        / targets if targets else None,
        "processing_cpu_seconds": sum(row["processing_cpu_seconds"]
                                      for row in rows),
        "provenance_cpu_seconds": sum(row["provenance_cpu_seconds"]
                                      for row in rows),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report all seeds/speeds and apply the unchanged frozen gates."""
    expected = {(arm, seed, speed, kind, order) for arm in ARMS
                for seed in (0, 1, 2)
                for speed in (25, 30, 40, 45, 55, 60, 70, 75, 85, 90, 100)
                for kind in ("positive", "sham") for order in (0, 1)}
    actual = [(r["arm"], r["seed"], r["speed"], r["kind"], r["order"])
              for r in rows]
    if len(actual) != 924 or set(actual) != expected:
        raise ValueError("incomplete or duplicate sensing matrix")
    per_seed = []
    per_speed_seed = []
    for arm in ARMS:
        for seed in (0, 1, 2):
            selected = [r for r in rows if r["arm"] == arm
                        and r["seed"] == seed]
            per_seed.append({"arm": arm, "seed": seed,
                             **aggregate(selected)})
            for speed in sorted({r["speed"] for r in selected}):
                subset = [r for r in selected if r["speed"] == speed]
                per_speed_seed.append({"arm": arm, "seed": seed,
                                       "speed": speed, **aggregate(subset)})
    by_key = {(r["arm"], r["seed"]): r for r in per_seed}
    gates = []
    for seed in (0, 1, 2):
        ours = by_key[("plastic_snn", seed)]
        frozen = by_key[("frozen_snn", seed)]
        ann = by_key[("plastic_ann", seed)]
        practical = (ours["recall"] >= 0.8
                     and ours["sham_false_alarms_per_minute"] <= 1.0)
        eligible = [arm for arm in ("persistence", "ema", "nlms")
                    if by_key[(arm, seed)][
                        "sham_false_alarms_per_minute"] <= 1.0]
        gates.append({
            "seed": seed, "practical": practical,
            "adaptation": practical
            and ours["recall"] >= frozen["recall"] + 0.10
            and ours["sham_false_alarms_per_minute"]
            <= frozen["sham_false_alarms_per_minute"],
            "snn_specific": practical
            and ours["recall"] >= ann["recall"] + 0.10
            and ours["sham_false_alarms_per_minute"]
            <= ann["sham_false_alarms_per_minute"],
            "competitive": practical and all(
                ours["recall"] >= by_key[(arm, seed)]["recall"]
                for arm in eligible),
            "eligible_cheap_baselines": eligible,
        })
    return {
        "stream_rows": len(rows), "per_seed": per_seed,
        "per_speed_seed": per_speed_seed,
        "pooled": [{"arm": arm, **aggregate(
            [row for row in rows if row["arm"] == arm])} for arm in ARMS],
        "gates_by_seed": gates,
        "all_seed_gates": {key: all(g[key] for g in gates) for key in (
            "practical", "adaptation", "snn_specific", "competitive")},
        "device_gate": "unmeasured",
        "interpretation": "Stitched clips from one fan; correlated "
        "repetitions, not independent recording or device replicates. "
        "A competitive gate with no eligible cheap baseline is vacuous.",
        "streams": rows,
    }
